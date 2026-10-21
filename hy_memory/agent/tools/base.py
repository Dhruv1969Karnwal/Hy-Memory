"""
Tool abstract base class + registry + dispatcher.

Design principles:
- Schemas follow the OpenAI function calling shape (eases a future move to native function calling)
- ToolHandler.execute() is async and takes arguments + context (injected by the caller, carrying user_id / agent_id
  and other business isolation info; the handler must never assume on its own where they come from)
- One Tool can be shared by many agents; the caller decides which tools to register with the LLM
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ================================================================
# Data structures
# ================================================================

@dataclass
class ToolDefinition:
    """
    The tool schema definition.

    Follows the OpenAI function calling style:
      {
        "name": "my_tool",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
      }
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to one entry of the OpenAI function calling tools array."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def to_prompt_snippet(self) -> str:
        """
        Convert to a compact prompt-embeddable description (the prompt-simulation fallback when function-calling is unavailable).
        """
        params_json = json.dumps(self.parameters or {"type": "object", "properties": {}}, ensure_ascii=False, indent=2)
        return f"- **{self.name}**: {self.description}\n  parameters schema:\n{params_json}"


@dataclass
class ToolCall:
    """One tool-call request issued by the LLM."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""   # call id from the LLM function-calling API; empty under prompt simulation


@dataclass
class ToolResult:
    """The tool execution result."""
    success: bool
    tool_name: str = ""
    call_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ================================================================
# Abstract interface
# ================================================================

class ToolHandler(ABC):
    """
    Abstract interface for a tool executor.

    Subclasses must implement:
    - definition() → ToolDefinition (describes the schema)
    - execute(arguments, context) → ToolResult

    The execute context is business context injected by the caller, typically containing:
      user_id / agent_id / session_id / request_id
    the handler takes whatever it needs.
    """

    @abstractmethod
    def definition(self) -> ToolDefinition:
        ...

    @property
    def name(self) -> str:
        return self.definition().name

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        ...


# ================================================================
# Registry + dispatcher
# ================================================================

class ToolRegistry:
    """
    The tool registry.

    Each agent creates a ToolRegistry instance on demand, registers the tools it needs,
    passes the schemas to the LLM, and executes the returned tool_calls via dispatch().
    """

    def __init__(self):
        self._handlers: Dict[str, ToolHandler] = {}

    def register(self, handler: ToolHandler) -> None:
        name = handler.name
        if name in self._handlers:
            logger.warning(f"[tools] handler '{name}' already registered, overriding")
        self._handlers[name] = handler

    def get(self, name: str) -> Optional[ToolHandler]:
        return self._handlers.get(name)

    def names(self) -> List[str]:
        return list(self._handlers.keys())

    def definitions(self) -> List[ToolDefinition]:
        return [h.definition() for h in self._handlers.values()]

    def openai_schemas(self) -> List[Dict[str, Any]]:
        return [d.to_openai_schema() for d in self.definitions()]

    def prompt_snippets(self) -> str:
        """Join the prompt snippets of all tools, for the prompt-simulation path."""
        if not self._handlers:
            return ""
        return "\n".join(d.to_prompt_snippet() for d in self.definitions())

    async def dispatch(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        """
        Execute one tool_call.

        An unknown tool name -> a ToolResult with success=False instead of an exception;
        the caller decides whether to feed the error back to the LLM or swallow the warning.
        """
        handler = self._handlers.get(call.name)
        if handler is None:
            msg = f"tool '{call.name}' not registered"
            logger.warning(f"[tools] dispatch failed: {msg}")
            return ToolResult(
                success=False,
                tool_name=call.name,
                call_id=call.call_id,
                error=msg,
            )
        try:
            result = await handler.execute(call.arguments or {}, context or {})
            result.tool_name = call.name
            result.call_id = call.call_id
            return result
        except Exception as e:
            logger.error(f"[tools] dispatch '{call.name}' raised: {e}", exc_info=True)
            return ToolResult(
                success=False,
                tool_name=call.name,
                call_id=call.call_id,
                error=str(e),
            )


# ================================================================
# Parsing helper: extract tool_calls from LLM output
# ================================================================

def parse_tool_calls_from_json(raw: Any) -> List[ToolCall]:
    """
    Parse ToolCall objects from the `tool_calls` field of LLM JSON output.

    Supports two common formats:
    1) OpenAI style: [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
       where arguments may be a JSON string
    2) Simplified style: [{"name": "...", "arguments": {...}}]
    """
    calls: List[ToolCall] = []
    if not raw:
        return calls
    if not isinstance(raw, list):
        return calls
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # OpenAI style
        if "function" in entry and isinstance(entry["function"], dict):
            fn = entry["function"]
            name = str(fn.get("name") or "").strip()
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw.strip() else {}
                except json.JSONDecodeError:
                    logger.warning(f"[tools] failed to parse arguments as JSON: {args_raw[:120]}")
                    args = {}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            call_id = str(entry.get("id") or "")
            if name:
                calls.append(ToolCall(name=name, arguments=args, call_id=call_id))
            continue
        # Simplified style
        name = str(entry.get("name") or "").strip()
        args = entry.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if name:
            calls.append(ToolCall(name=name, arguments=args, call_id=str(entry.get("id") or "")))
    return calls
