"""Optional runtime capabilities established by startup configuration."""

_enabled_optional_tools: frozenset[str] = frozenset()
_OPTIONAL_TOOLS = frozenset({"web_search"})


def configure_optional_tools(enabled: frozenset[str]) -> None:
    global _enabled_optional_tools
    _enabled_optional_tools = frozenset(enabled) & _OPTIONAL_TOOLS


def tool_available(name: str) -> bool:
    return name not in _OPTIONAL_TOOLS or name in _enabled_optional_tools


def available_tools(tools):
    return [tool for tool in tools if tool_available(tool["function"]["name"])]
