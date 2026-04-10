from typing import List, TypedDict

from langgraph.graph import StateGraph

from app.generation.llm import get_llm
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever


class RAGState(TypedDict):
    query: str
    docs: List[str]
    compressed_docs: List[str]
    answer: str


retriever = Retriever()
reranker = Reranker()
llm = get_llm()


def retrieve(state: RAGState):
    docs = retriever.retrieve(state["query"])
    return {"docs": docs}


def rerank(state: RAGState):
    compressed = reranker.rerank(state["query"], state["docs"])
    return {"compressed_docs": compressed}


def generate(state: RAGState):
    context = "\n\n".join(state["compressed_docs"])

    prompt = f"""
Use the context below to answer:

{context}

Question: {state['query']}
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")

    return graph.compile()
