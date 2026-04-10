import os

from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
