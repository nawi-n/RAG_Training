import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.generation.rag_chain import RAGChain  # noqa: E402

rag = RAGChain()

while True:
    q = input("\nQuery: ")
    if q == "exit":
        break

    print("\nAnswer:\n", rag.run(q))
