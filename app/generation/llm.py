import os

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langfuse.decorators import observe
except ModuleNotFoundError:
    from langfuse import observe


@observe()
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


@observe(name="traced_llm_call")
def traced_llm_call(llm, prompt: str):
    return llm.invoke(prompt)


@observe(name="traced_llm_call_async")
async def traced_llm_call_async(llm, prompt: str):
    return await llm.ainvoke(prompt)
