#!/usr/bin/env python3
"""mragent — multi-role LLM agent with graph memory."""

import argparse
import sys

from llmwrapper import LLMWrapper, Prompt
from memory.graph import MemoryGraph
from agent.agent import MemoryAgent

MEMORY_FILE = "memory.json"


def _agent(llm: LLMWrapper) -> MemoryAgent:
    return MemoryAgent(llm, MemoryGraph(MEMORY_FILE))


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_ask(llm: LLMWrapper, args) -> None:
    question = " ".join(args.question)
    print(f"[prompt] {question}\n")
    print(llm.chat(question))


def cmd_chat(llm: LLMWrapper, args) -> None:
    system = args.system
    history: list[dict] = []
    print("Interactive chat — type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            break
        prompt = Prompt(user=user_input, system=system, history=history)
        reply = llm.call(prompt)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        print(f"\nagent> {reply}\n")


def cmd_summarize(llm: LLMWrapper, args) -> None:
    if args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        print("Reading from stdin (Ctrl-D to end)...")
        text = sys.stdin.read()
    print(f"\n[summarizing {len(text)} chars]\n")
    print(llm.call(Prompt(
        user=f"Summarize the following text concisely:\n\n{text}",
        system="You are a concise summarizer. Output 3-5 bullet points.",
    )))


def cmd_research(llm: LLMWrapper, args) -> None:
    topic = " ".join(args.topic)
    print(f"[research] topic: {topic}\n")

    print("--- step 1: brainstorm ---")
    ideas = llm.call(Prompt(
        user=f"List 8 interesting facts or angles about: {topic}",
        system="You are a research assistant. Be specific and factual.",
    ))
    print(ideas)

    print("\n--- step 2: distill ---")
    distilled = llm.call(Prompt(
        user=f"From these ideas, pick the 3 most insightful and explain why:\n\n{ideas}",
        system="You are an editor who selects the most valuable insights.",
    ))
    print(distilled)

    print("\n--- step 3: briefing ---")
    briefing = llm.call(Prompt(
        user=f"Turn these insights into a tight 3-paragraph briefing on '{topic}':\n\n{distilled}",
        system="You are a journalist writing a crisp executive briefing.",
    ))
    print(briefing)


def cmd_remember(llm: LLMWrapper, args) -> None:
    """Store a piece of text in the memory graph (auto-extracts cues and tags)."""
    if args.file:
        with open(args.file) as f:
            content = f.read()
    else:
        content = " ".join(args.text)

    agent = _agent(llm)
    if args.cues or args.tags:
        cues = [c.strip() for c in args.cues.split(",")] if args.cues else []
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
        cid = agent.store(content, cues, tags)
    else:
        print("[memory] extracting cues and tags with LLM...")
        cid = agent.store_auto(content)

    print(f"[memory] saved as {cid}")


def cmd_recall(llm: LLMWrapper, args) -> None:
    """Answer a question from memory."""
    query = " ".join(args.query)
    print(f"[recall] {query}\n")
    agent = _agent(llm)
    answer = agent.recall(query, deep=args.deep)
    print(f"\n{answer}")


def cmd_ingest(llm: LLMWrapper, args) -> None:
    """Ingest a PDF into the memory graph."""
    agent = _agent(llm)
    count = agent.ingest_pdf(args.file, engine=args.engine, chunk_size=args.chunk_size, workers=args.workers)
    print(f"[ingest] done — {count} chunk(s) stored from {args.file}")


def cmd_memory_show(_llm: LLMWrapper, _args) -> None:
    """Print a summary of what is stored in memory."""
    graph = MemoryGraph(MEMORY_FILE)
    cues = graph.all_cues()
    tags = graph.all_tags()
    content_nodes = [n for n in graph.nodes.values() if n.type == "content"]
    print(f"Memory: {len(content_nodes)} item(s), {len(cues)} cue(s), {len(tags)} tag(s)")
    if tags:
        print(f"Tags : {', '.join(tags)}")
    if cues:
        print(f"Cues : {', '.join(cues)}")
    for node in content_nodes:
        preview = node.text[:120].replace("\n", " ")
        print(f"  [{node.id}] {preview}{'...' if len(node.text) > 120 else ''}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="mragent", description="Multi-role LLM agent with graph memory")
    parser.add_argument("--model", default=None, help="Override OpenRouter model")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="One-shot question")
    p_ask.add_argument("question", nargs="+")

    p_chat = sub.add_parser("chat", help="Interactive conversation")
    p_chat.add_argument("--system", default=None)

    p_sum = sub.add_parser("summarize", help="Summarize text from file or stdin")
    p_sum.add_argument("file", nargs="?")

    p_res = sub.add_parser("research", help="Multi-step research pipeline")
    p_res.add_argument("topic", nargs="+")

    p_rem = sub.add_parser("remember", help="Store text in memory graph")
    p_rem.add_argument("text", nargs="*", help="Text to store (or use --file)")
    p_rem.add_argument("--file", help="Read content from file")
    p_rem.add_argument("--cues", help="Comma-separated cue keywords (skip LLM extraction)")
    p_rem.add_argument("--tags", help="Comma-separated topic tags (skip LLM extraction)")

    p_rec = sub.add_parser("recall", help="Answer a question from memory")
    p_rec.add_argument("query", nargs="+")
    p_rec.add_argument("--deep", action="store_true",
                       help="Iterative LLM-guided graph traversal (slower, more thorough)")

    p_ing = sub.add_parser("ingest", help="Ingest a PDF into memory")
    p_ing.add_argument("file", help="Path to PDF file")
    p_ing.add_argument(
        "--engine",
        default="mistral-ocr",
        choices=["mistral-ocr", "cloudflare-ai", "native", "mistral-ocr-4"],
        help="OCR engine (default: mistral-ocr via OpenRouter at $2/1000 pages)",
    )
    p_ing.add_argument("--chunk-size", type=int, default=800, metavar="N",
                       help="Max chars per memory chunk (default: 800)")
    p_ing.add_argument("--workers", type=int, default=5, metavar="N",
                       help="Parallel LLM workers for cue/tag extraction (default: 5)")

    sub.add_parser("memory", help="Show memory graph summary")

    args = parser.parse_args()
    llm = LLMWrapper(model=args.model) if args.model else LLMWrapper()

    dispatch = {
        "ask": cmd_ask,
        "chat": cmd_chat,
        "summarize": cmd_summarize,
        "research": cmd_research,
        "remember": cmd_remember,
        "recall": cmd_recall,
        "ingest": cmd_ingest,
        "memory": cmd_memory_show,
    }
    dispatch[args.command](llm, args)


if __name__ == "__main__":
    main()
