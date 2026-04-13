import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.logger import get_logger  # noqa: E402
from app.retrieval.query_engine import QueryEngine  # noqa: E402

logger = get_logger()


def main():
    engine = QueryEngine()

    while True:
        query = input("\nEnter your query (or 'exit'): ")

        if query.lower() == "exit":
            break

        results = engine.query(query, top_k=3)

        print("\n🔍 Top Results:\n")

        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc[:200]}")
            print("-" * 50)


if __name__ == "__main__":
    main()
