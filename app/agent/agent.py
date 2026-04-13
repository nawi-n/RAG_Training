from langchain.agents import create_agent
from langchain.tools import tool

from app.retrieval.router import select_retrieval_strategy
from app.retrieval.web_search import web_search


@tool
def retriever_selector_tool(query: str) -> str:
    """Select retrieval strategy: vector, keyword, hybrid, skip"""
    return select_retrieval_strategy(query)


@tool
def web_search_tool(query: str) -> str:
    """Search the web when internal retrieval fails"""
    results = web_search(query)
    return "\n".join(results)


def build_agent(llm):
    tools = [
        retriever_selector_tool,
        web_search_tool,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
You are a financial-report RAG control agent.

Your job:
- Decide retrieval strategy: vector, keyword, hybrid, or skip
- Prefer internal PDF evidence for company metrics, statements, and commentary
- Use web search ONLY for clearly external or current-events questions
- Improve/normalize the query when helpful (e.g., quarter/year terms)

Output guidance:
- Keep answers concise
- Prioritize factual, auditable responses
- Avoid speculation when evidence is missing

""",
    )

    return agent
