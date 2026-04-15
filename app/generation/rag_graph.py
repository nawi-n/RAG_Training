from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agent.agent import build_agent
from app.generation.llm import get_llm


class RAGState(TypedDict):
    query: str
    answer: str
    route: str


llm = get_llm()
agent_executor = build_agent()


# 🔹 1. ROUTER NODE
def router(state: RAGState):
    prompt = f"""
    Decide routing:

    Query: {state['query']}

    Return only:
    - generic
    - agent
    """
    decision = llm.invoke(prompt).content.strip().lower()

    return {"route": "generic" if "generic" in decision else "agent"}


# 🔹 2. GENERIC RESPONSE NODE
def respond_generic(state: RAGState):
    response = llm.invoke(state["query"])
    return {"answer": response.content}


# 🔹 3. AGENT NODE (🔥 CORE)
def call_agent(state: RAGState):
    result = agent_executor.invoke(
        {"messages": [{"role": "user", "content": state["query"]}]}
    )

    return {"answer": result["messages"][-1].content}


def extract_answer(state):
    return {"answer": state.get("answer", "No answer")}


# 🔹 BUILD GRAPH
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("router", router)
    graph.add_node("generic", respond_generic)
    graph.add_node("agent", call_agent)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "generic": "generic",
            "agent": "agent",
        },
    )

    graph.add_edge("generic", END)
    graph.add_edge("agent", END)

    return graph.compile()
