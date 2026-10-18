"""
Agent Memory - MemAgent coordinator

Unified entry point that coordinates and schedules the other agents (Extractor, Reflector).

Capabilities:
- Receive processing requests
- Dispatch the right agent per task type
- Aggregate processing results
- Error handling and degradation

Example:
    agent = MemAgent()
    
    # Handle an add-memory request
    result = await agent.process_add(
        content="The user said they like Sichuan food, especially mapo tofu",
        context={"uid": "user_456"}
    )
    
    # The result carries the extracted information
    print(result.extracted_info)  # {"preferences": {"food": "Sichuan food"}}
    print(result.suggested_layer)  # "profile"
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .extractor import Extractor
from .reflector import Reflector
from .llm_provider import LLMProvider
from ..config import MemoryConfig

logger = logging.getLogger(__name__)


class ProcessMode(Enum):
    """Processing modes."""
    FULL = "full"           # Full processing (extract + summary + reflect)
    EXTRACT_ONLY = "extract"  # Extraction only
    SKIP = "skip"           # Skip Agent processing


@dataclass
class AgentResult:
    """
    Agent processing result
    """
    success: bool
    
    # Extraction results
    extracted_info: Dict[str, Any] = field(default_factory=dict)
    suggested_layer: Optional[str] = None
    
    # Summary results
    summary: Optional[str] = None  # raw LLM summary text (already stripped), stored directly
    
    # Reflection results
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    should_merge: bool = False
    merge_target_id: Optional[str] = None
    
    # Metadata
    processing_time_ms: float = 0
    tokens_used: int = 0
    error: Optional[str] = None
    error_code: Optional[str] = None        # "EMPTY_RESPONSE" / "JSON_PARSE_FAILED" / "LLM_ERROR"
    extract_raw_response: Optional[str] = None  # raw LLM response (kept on failure)

    # Fine-grained stats (tokens and latency per step)
    extract_tokens_used: int = 0
    extract_prompt_tokens: int = 0
    extract_completion_tokens: int = 0
    extract_elapsed_ms: float = 0
    summary_tokens_used: int = 0
    summary_prompt_tokens: int = 0
    summary_completion_tokens: int = 0
    summary_elapsed_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "extracted_info": self.extracted_info,
            "suggested_layer": self.suggested_layer,
            "summary": self.summary,
            "conflicts": self.conflicts,
            "should_merge": self.should_merge,
            "processing_time_ms": self.processing_time_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "extract_tokens_used": self.extract_tokens_used,
            "extract_prompt_tokens": self.extract_prompt_tokens,
            "extract_completion_tokens": self.extract_completion_tokens,
            "extract_elapsed_ms": self.extract_elapsed_ms,
            "summary_tokens_used": self.summary_tokens_used,
            "summary_prompt_tokens": self.summary_prompt_tokens,
            "summary_completion_tokens": self.summary_completion_tokens,
            "summary_elapsed_ms": self.summary_elapsed_ms,
        }


class MemAgent:
    """
    Memory agent coordinator
    
    Coordinates the Extractor and Reflector to carry out memory processing.
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize MemAgent
        
        Args:
            config: the configuration object
            llm_provider: the LLM provider
        """
        self.config = config or MemoryConfig.from_env()
        self.llm_provider = llm_provider or LLMProvider(self.config)
        
        # Initialize sub-agents (llm_config is passed in to support per-scene max_tokens settings)
        llm_config = self.config.llm
        self.extractor = Extractor(self.llm_provider, llm_config)
        self.reflector = Reflector(self.llm_provider, llm_config)
        
        logger.debug("MemAgent initialized")

    async def process_add(
        self,
        content: str,
        context: Dict[str, Any],
        mode: ProcessMode = ProcessMode.FULL,
        existing_memories: List[Dict] = None,
        memory_at: Optional[datetime] = None,
        existing_tags: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_registry: Optional[Any] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        history_context: str = "",
        basic_profile_fields: Optional[Dict[str, str]] = None,
    ) -> AgentResult:
        """
        Handle an add-memory request

        Args:
            content: the memory content
            context: context information (uid, agent_id, etc.)
            mode: the processing mode
            existing_memories: similar existing memories (for conflict detection)
            memory_at: when the memory happened (the current_time baseline for the extract prompt)
            existing_tags: optional, all tags the user already has (the extractor lets the LLM prefer reusing them)
            tools:   **DEPRECATED** not forwarded to the LLM since v0.1.5.13+; kept only for backward compatibility
            tool_registry: **DEPRECATED** same as above
            tool_context:  kept (unused)
            basic_profile_fields:
                {field_name: description} field table passed through by the caller from MemoryConfig.basic_profile.fields;
                the extractor renders it into the prompt, asking the LLM to return it in the JSON basic_info field.

        Returns:
            the processing result
        """
        start_time = datetime.now()
        total_tokens = 0

        # Pick the time baseline for the extract prompt: memory_at first, otherwise an empty string (left blank in the prompt)
        current_time_str = memory_at.isoformat(timespec="seconds") if memory_at else ""

        try:
            result = AgentResult(success=True)

            # Extract structured information; the old rolling-summary path was removed.
            _do_extract = mode in (ProcessMode.FULL, ProcessMode.EXTRACT_ONLY)

            async def _run_extract():
                return await self.extractor.extract(
                    content, context,
                    current_time=current_time_str,
                    existing_tags=existing_tags,
                    tools=tools,
                    tool_registry=tool_registry,
                    tool_context=tool_context,
                    history_context=history_context,
                    basic_profile_fields=basic_profile_fields,
                )

            _t_parallel = datetime.now()

            extract_result = await _run_extract() if _do_extract else None

            # Handle the extract result
            if _do_extract and extract_result:
                _extract_elapsed = (datetime.now() - _t_parallel).total_seconds() * 1000
                result.extracted_info = extract_result.info
                result.suggested_layer = extract_result.suggested_layer
                total_tokens += extract_result.tokens_used
                result.extract_tokens_used = extract_result.tokens_used
                result.extract_prompt_tokens = extract_result.prompt_tokens
                result.extract_completion_tokens = extract_result.completion_tokens
                result.extract_elapsed_ms = _extract_elapsed
                result.extract_raw_response = extract_result.raw_response
                result._extract_result = extract_result  # keep the full result for trace logging

                # Return early when extraction failed
                if not extract_result.success:
                    result.success = False
                    result.error = extract_result.error
                    result.error_code = extract_result.error_code
                    result.processing_time_ms = (
                        datetime.now() - start_time
                    ).total_seconds() * 1000
                    result.tokens_used = total_tokens
                    logger.warning(
                        f"MemAgent: extract failed: code={extract_result.error_code} "
                        f"error={extract_result.error}"
                    )
                    return result

                # INFO: extraction result summary
                info = extract_result.info or {}
                identity = info.get("identity", []) or []
                facts = info.get("facts", []) or []
                basic_info = info.get("basic_info", {}) or {}
                basic_field_count = sum(
                    1 for k, v in basic_info.items() if v and v != "null"
                )
                logger.info(
                    f"MemAgent: extraction → "
                    f"identity={len(identity)} facts={len(facts)} "
                    f"basic_info={basic_field_count}fields"
                )
                logger.info(f"MemAgent: extracted_info={extract_result.info}")

            # 3. Conflict detection (if existing memories are present)
            if mode == ProcessMode.FULL and existing_memories:
                reflect_result = await self.reflector.check_conflicts(
                    content,
                    existing_memories,
                    context
                )
                result.conflicts = reflect_result.conflicts
                result.should_merge = reflect_result.should_merge
                result.merge_target_id = reflect_result.merge_target_id
                total_tokens += reflect_result.tokens_used
                logger.info(
                    f"MemAgent: conflict detection → "
                    f"conflicts={len(reflect_result.conflicts)} "
                    f"should_merge={reflect_result.should_merge}"
                )
            
            # Compute processing time
            result.processing_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000
            result.tokens_used = total_tokens

            logger.debug(f"MemAgent: tokens_used={total_tokens}")

            return result
            
        except Exception as e:
            logger.error(f"MemAgent process_add failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
    
    async def process_recall(
        self,
        query: str,
        memories: List[Dict],
        context: Dict[str, Any],
    ) -> AgentResult:
        """
        Handle a recall request (optional post-processing)
        
        Post-processing can:
        - reorder results
        - generate a summary
        - generate an answer
        
        Args:
            query: the query text
            memories: the recalled memory list
            context: context information
        
        Returns:
            the processing result
        """
        start_time = datetime.now()
        
        try:
            # Build the combined summary
            combined_content = "\n".join([m.get("content", "") for m in memories])
            
            if combined_content:
                abstract_result = await self.abstractor.abstract(
                    combined_content,
                    context,
                    focus=query  # focus on the query content
                )
                
                return AgentResult(
                    success=True,
                    summary=abstract_result.summary,
                    tokens_used=abstract_result.tokens_used,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                )
            
            return AgentResult(
                success=True,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
            
        except Exception as e:
            logger.error(f"MemAgent process_recall failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        return {
            "extractor": self.extractor.get_stats(),
            "reflector": self.reflector.get_stats(),
        }
