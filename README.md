# RAG Training

This is a small end-to-end RAG practice project.

What is implemented:
- Data ingestion from PDF, HTML, and CSV (pymupdf4llm, BeautifulSoup+lxml, pandas)
- Text chunking and embedding generation (custom chunker with fixed overlap, Hugging Face InferenceClient `BAAI/bge-m3`)
- Chroma vector store persistence (ChromaDB)
- Retrieval + reranking flow (custom retriever over Chroma + sentence-transformers CrossEncoder `BAAI/bge-reranker-base`)
- Two generation paths: a simple RAG chain and a LangGraph-based graph (ChatGoogleGenerativeAI `gemini-2.5-flash-lite`, LangGraph StateGraph)
- Basic interactive test scripts for querying the system (Python CLI scripts in `test/`)
- Pre-Commits
