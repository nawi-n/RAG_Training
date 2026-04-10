# app/chunking/text_chunker.py

from typing import List

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.logger import get_logger

logger = get_logger()


def chunk_text(text: str) -> List[str]:
    logger.info("Starting text chunking")

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    logger.info(f"Generated {len(chunks)} chunks")

    return chunks
