import asyncio
from typing import List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.agent import build_agent
from app.generation.llm import get_llm, traced_llm_call, traced_llm_call_async
from app.generation.retry_utils import retry_with_exception
from app.observability.langfuse_client import langfuse
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.retrieval.router import select_retrieval_strategy
from app.retrieval.web_search import web_search


class RAGState(TypedDict, total=False):
    query: str
    docs: List[str]
    compressed_docs: List[str]
    answer: str
    error: str
    agent_output: str
    retrieval_strategy: str
    web_results: List[str]


retriever = Retriever()
reranker = Reranker()
llm = get_llm()
agent = build_agent(llm)


def retrieve(state: RAGState):
    docs = retriever.retrieve(state["query"])
    return {"docs": docs}


def retrieve_hybrid(state: RAGState):
    vector_docs = retriever.retrieve(state["query"])
    web_docs = web_search(state["query"], max_results=5)
    return {
        "docs": vector_docs + web_docs,
        "web_results": web_docs,
    }


def rerank(state: RAGState):
    compressed = reranker.rerank(state["query"], state["docs"])
    return {"compressed_docs": compressed}


async def retrieve_async(state: RAGState):
    docs = await asyncio.to_thread(retriever.retrieve, state["query"])
    return {"docs": docs}


async def retrieve_hybrid_async(state: RAGState):
    vector_docs = await asyncio.to_thread(retriever.retrieve, state["query"])
    web_docs = await asyncio.to_thread(web_search, state["query"], 5)
    return {
        "docs": vector_docs + web_docs,
        "web_results": web_docs,
    }


async def rerank_async(state: RAGState):
    compressed = await asyncio.to_thread(reranker.rerank, state["query"], state["docs"])
    return {"compressed_docs": compressed}


def agent_node(state: RAGState):
    query = state["query"]

    if not query.strip():
        return {
            "query": query,
            "agent_output": "",
            "retrieval_strategy": "skip",
        }

    response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    strategy = select_retrieval_strategy(query)

    return {
        "query": query,
        "agent_output": str(response),
        "retrieval_strategy": strategy,
    }


async def agent_node_async(state: RAGState):
    query = state["query"]

    if not query.strip():
        return {
            "query": query,
            "agent_output": "",
            "retrieval_strategy": "skip",
        }

    response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

    strategy = select_retrieval_strategy(query)

    return {
        "query": query,
        "agent_output": str(response),
        "retrieval_strategy": strategy,
    }


def route_after_agent(
    state: RAGState,
) -> Literal["retrieve", "retrieve_hybrid", "response_generic"]:
    strategy = state.get("retrieval_strategy", "vector")
    if strategy == "skip":
        return "response_generic"
    if strategy == "hybrid":
        return "retrieve_hybrid"
    return "retrieve"


@retry_with_exception
def response_generic(state: RAGState):
    error_context = state.get("error", "")
    prompt = f"""
Answer the user directly and concisely.

Previous error (if any):
{error_context}

Question: {state['query']}
"""

    response = traced_llm_call(llm, prompt)
    return {"answer": response.content}


@retry_with_exception
async def response_generic_async(state: RAGState):
    error_context = state.get("error", "")
    prompt = f"""
Answer the user directly and concisely.

Previous error (if any):
{error_context}

Question: {state['query']}
"""

    response = await traced_llm_call_async(llm, prompt)
    return {"answer": response.content}


@retry_with_exception
def generate(state: RAGState):
    context = "\n\n".join(state.get("compressed_docs", []))
    error_context = state.get("error", "")

    prompt = f"""
Use the context below to answer:

{context}

Previous error (if any):
{error_context}

Question: {state['query']}
"""

    trace = (
        langfuse.trace(name="generate")
        if (langfuse and hasattr(langfuse, "trace"))
        else None
    )
    response = traced_llm_call(llm, prompt)
    if trace:
        trace.update(output=response.content)

    return {"answer": response.content}


@retry_with_exception
async def generate_async(state: RAGState):
    context = "\n\n".join(state.get("compressed_docs", []))
    error_context = state.get("error", "")

    prompt = f"""
Use the context below to answer:

{context}

Previous error (if any):
{error_context}

Question: {state['query']}
"""

    trace = (
        langfuse.trace(name="generate")
        if (langfuse and hasattr(langfuse, "trace"))
        else None
    )
    response = await traced_llm_call_async(llm, prompt)
    if trace:
        trace.update(output=response.content)

    return {"answer": response.content}


def build_graph(async_mode: bool = False):
    graph = StateGraph(RAGState)

    graph.add_node("agent", agent_node_async if async_mode else agent_node)
    graph.add_node("retrieve", retrieve_async if async_mode else retrieve)
    graph.add_node(
        "retrieve_hybrid",
        retrieve_hybrid_async if async_mode else retrieve_hybrid,
    )
    graph.add_node("rerank", rerank_async if async_mode else rerank)
    graph.add_node("generate", generate_async if async_mode else generate)
    graph.add_node(
        "response_generic",
        response_generic_async if async_mode else response_generic,
    )

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "retrieve": "retrieve",
            "retrieve_hybrid": "retrieve_hybrid",
            "response_generic": "response_generic",
        },
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("retrieve_hybrid", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("response_generic", END)

    return graph.compile()


def build_async_graph():
    return build_graph(async_mode=True)
