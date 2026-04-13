import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.generation.rag_graph import build_graph  # noqa: E402

graph = build_graph()

while True:
    q = input("\nQuery: ")
    if q == "exit":
        break

    result = graph.invoke({"query": q})
    print("\nAnswer:\n", result["answer"])
