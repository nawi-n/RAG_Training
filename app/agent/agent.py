from langchain.agents import create_agent

from app.agent.tools import RetrieveTool, WebSearchTool
from app.generation.llm import get_llm


def build_agent():
    llm = get_llm()

    tools = [
        RetrieveTool(),
        # LLMTool(),
        WebSearchTool(),
    ]

    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
        You are a smart assistant with access to tools.

        Rules:
        - Use a tool ONLY if needed.
        - NEVER call the same tool more than once for a query.
        - After receiving a tool result, directly produce the final answer.
        - Do NOT call tools again if you already have sufficient information.

        Tool usage:
        - retrieve_from_db → for company financial report related information.
        - web_search → for recent, external, or missing information.

        If no tool is required, answer directly.

        Be concise, factual, and avoid repetition.
        """,
        debug=True,  # helpful for logs
        name="rag_agent",
    )

    return agent_graph
