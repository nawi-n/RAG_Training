from app.generation.llm import get_llm
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever


class RAGChain:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = get_llm()

    def run(self, query: str) -> str:
        docs = self.retriever.retrieve(query)

        compressed_docs = self.reranker.rerank(query, docs)

        context = "\n\n".join(compressed_docs)

        prompt = f"""
You are a helpful AI assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

        response = self.llm.invoke(prompt)

        return response.content
