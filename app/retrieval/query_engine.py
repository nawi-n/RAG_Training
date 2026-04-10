from typing import List

from app.embedding.hf_embedder import HFEmbedder
from app.logger import get_logger
from app.vectorstore.chroma_store import ChromaStore

logger = get_logger()


class QueryEngine:
    def __init__(self):
        self.embedder = HFEmbedder()
        self.vector_store = ChromaStore()

    def query(self, query_text: str, top_k: int = 3) -> List[str]:
        logger.info(f"Processing query: {query_text}")

        # 🔹 Step 1: Embed query
        query_embedding = self.embedder.embed([query_text])[0]

        # 🔹 Step 2: Search in Chroma
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]

        logger.info(f"Retrieved {len(documents)} results")

        return documents
