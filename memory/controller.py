"""Tool schemas and dispatch for LLM memory access."""

from memory.graph import MemoryGraph

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search memory by cue keyword or entity. Returns matching stored content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cue": {"type": "string", "description": "Keyword or entity to look up"}
                },
                "required": ["cue"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_by_tag",
            "description": "Retrieve stored content by semantic topic/tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "Topic or semantic category"}
                },
                "required": ["tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cues",
            "description": "List all cue keywords currently in memory. Use to discover what's stored.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tags",
            "description": "List all semantic tags currently in memory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def dispatch(graph: MemoryGraph, name: str, args: dict, registry: dict | None = None) -> str:
    """Run a tool call against the graph.

    *registry* (optional), if given, is a shared ``{full_content_text: citation_number}``
    dict — each distinct content chunk returned gets assigned a stable [N] the
    first time it's seen, prefixed to the result line, so the LLM can cite it
    inline instead of quoting it verbatim.
    """
    def _cite(text: str) -> str:
        if registry is None:
            return ""
        if text not in registry:
            registry[text] = len(registry) + 1
        return f"[{registry[text]}] "

    if name == "search_memory":
        results = graph.search_cue(args.get("cue", ""))
        if not results:
            return f"No memories found for cue: {args.get('cue')}"
        lines = [f"{_cite(r['content'])}({r['cue']} → {r['tag']}) {r['content'][:600]}" for r in results[:4]]
        return "\n".join(lines)

    if name == "get_by_tag":
        results = graph.get_by_tag(args.get("tag", ""))
        if not results:
            return f"No memories found for tag: {args.get('tag')}"
        lines = [f"{_cite(r['content'])}({r['tag']}) {r['content'][:600]}" for r in results[:4]]
        return "\n".join(lines)

    if name == "list_cues":
        cues = graph.all_cues()
        return f"Cues: {', '.join(cues)}" if cues else "Memory is empty."

    if name == "list_tags":
        tags = graph.all_tags()
        return f"Tags: {', '.join(tags)}" if tags else "Memory is empty."

    return f"Unknown tool: {name}"
