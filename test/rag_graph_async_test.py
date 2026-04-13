import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.generation.rag_graph import build_async_graph  # noqa: E402


async def main():
    graph = build_async_graph()

    while True:
        q = input("\nAsync Query: ")
        if q == "exit":
            break

        result = await graph.ainvoke({"query": q})
        print("\nAnswer:\n", result.get("answer", ""))


if __name__ == "__main__":
    asyncio.run(main())
