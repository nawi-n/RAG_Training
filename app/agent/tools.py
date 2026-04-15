from duckduckgo_search import DDGS
from langchain.tools import BaseTool

from app.generation.llm import get_llm
from app.retrieval.retriever import Retriever

retriever = Retriever()
llm = get_llm()


# 🔹 1. RAG TOOL
class RetrieveTool(BaseTool):
    name: str = "retrieve_from_db"
    description: str = (
        "Use this to answer questions related to financial reports "
        "such as revenue, profit, filings, metrics, and members.\n"
        "Examples: 'What was Apple's revenue in 2023?', 'Who are the directors?'"
    )

    def _run(self, query: str):
        docs = retriever.retrieve(query)
        context = "\n".join(docs)

        return context


"""
# 🔹 2. DIRECT LLM TOOL
class LLMTool(BaseTool):
    name : str=  "ask_llm"
    description : str = ("General or casual questions not tied to financial reports.\n"
                   "Examples: 'What is inflation?', 'Explain EBITDA', 'Hi'")

    def _run(self, query: str):
        return llm.invoke(query).content
"""


# 🔹 3. WEB SEARCH TOOL
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Fetch recent or external info not in reports.\n"
        "Examples: 'latest Tesla news', 'current interest rates 2026'"
    )

    def _run(self, query: str):
        results = DDGS().text(query, max_results=5)

        output = ""
        for r in results:
            output += f"{r['title']} - {r['body']}\n"

        return output
