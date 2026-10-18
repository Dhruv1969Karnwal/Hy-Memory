"""
HY Memory Agent Tools

Generic LLM "tool" mechanism (OpenAI function-calling style). The SDK no longer uses it internally
for basic user profile updates (expressed directly via the extractor prompt + the JSON `basic_info` field,
to avoid weak models filling it in arbitrarily). This module is kept for future agent scenarios.

Core components:
- ToolDefinition: schema description of a tool (OpenAI function calling style)
- ToolCall:       a tool call request issued by the LLM
- ToolResult:     result after dispatcher execution
- ToolHandler:    tool execution logic (abstract interface)
- ToolRegistry:   tool registry + dispatcher

Usage (custom scenarios)::

    from .tools import ToolRegistry, ToolHandler, ToolDefinition, ToolResult

    class MyTool(ToolHandler):
        def definition(self) -> ToolDefinition:
            return ToolDefinition(name="my_tool", description="...", parameters={...})
        async def execute(self, arguments, context) -> ToolResult:
            return ToolResult(success=True, data={...})

    registry = ToolRegistry()
    registry.register(MyTool())

    # pass to the LLM
    tool_schemas = registry.openai_schemas()

    # after the LLM returns tool_calls
    for call in tool_calls:
        result = await registry.dispatch(call, context={...})
"""

from .base import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolHandler,
    ToolRegistry,
)

__all__ = [
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolHandler",
    "ToolRegistry",
]
