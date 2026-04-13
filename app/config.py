import os

from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "BAAI/bge-m3"
HF_TOKEN = os.getenv("HF_TOKEN")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
