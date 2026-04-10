import hashlib
import os
import pickle
from typing import List, Tuple

from app.chunking.text_chunker import chunk_text
from app.embedding.hf_embedder import HFEmbedder
from app.logger import get_logger
from app.parsers.csv_parser import parse_csv
from app.parsers.html_parser import parse_html
from app.parsers.pdf_parser import parse_pdf
from app.vectorstore.chroma_store import ChromaStore

logger = get_logger()

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class IngestionPipeline:
    def __init__(self):
        self.embedder = HFEmbedder()
        self.vector_store = ChromaStore()

    # 🔐 Generate file hash (detects file changes)
    def _generate_file_hash(self, file_path: str) -> str:
        logger.debug(f"Generating hash for {file_path}")

        hasher = hashlib.md5()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        file_hash = hasher.hexdigest()
        logger.debug(f"Hash: {file_hash}")

        return file_hash

    # 📦 Cache path using hash
    def _get_cache_path(self, file_path: str) -> str:
        file_hash = self._generate_file_hash(file_path)
        filename = os.path.basename(file_path)

        cache_file = f"{filename}_{file_hash}.pkl"
        return os.path.join(CACHE_DIR, cache_file)

    # 🧹 Remove old cache versions
    def _cleanup_old_cache(self, file_path: str):
        filename = os.path.basename(file_path)

        for file in os.listdir(CACHE_DIR):
            if file.startswith(filename) and file.endswith(".pkl"):
                full_path = os.path.join(CACHE_DIR, file)
                logger.debug(f"Removing old cache: {full_path}")
                os.remove(full_path)

    # 📄 Parser selector
    def parse(self, file_path: str) -> str:
        logger.info(f"Parsing file: {file_path}")

        if file_path.endswith(".csv"):
            return parse_csv(file_path)

        if file_path.endswith(".html"):
            return parse_html(file_path)

        if file_path.endswith(".pdf"):
            return parse_pdf(file_path)

        raise ValueError(f"Unsupported format: {file_path}")

    # 🧠 Generate deterministic IDs (prevents duplicates)
    def _generate_ids(self, file_path: str, num_chunks: int) -> List[str]:
        base = os.path.basename(file_path)
        return [f"{base}_{i}" for i in range(num_chunks)]

    # 🚀 Main pipeline
    def run(self, file_path: str) -> List[Tuple[str, List[float]]]:
        logger.info(f"Pipeline started for {file_path}")

        cache_path = self._get_cache_path(file_path)

        # ✅ CASE 1: Load from cache
        if os.path.exists(cache_path):
            logger.info(f"Loading from cache: {cache_path}")

            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)

            chunks = [item[0] for item in cached_data]
            embeddings = [item[1] for item in cached_data]

            logger.info("Pushing cached data to Chroma")

            ids = self._generate_ids(file_path, len(chunks))

            # ✅ use upsert → avoids duplicates
            self.vector_store.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
            )

            logger.info("Cache loaded and stored in Chroma")

            return cached_data

        # ❗ CASE 2: Fresh processing
        logger.info("No cache found, processing file")

        text = self.parse(file_path)

        chunks = chunk_text(text)
        logger.debug(f"Chunks created: {len(chunks)}")

        embeddings = self.embedder.embed(chunks)

        ids = self._generate_ids(file_path, len(chunks))

        # ✅ store in Chroma (upsert safe)
        self.vector_store.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
        )

        results = list(zip(chunks, embeddings))

        # 🧹 Remove old cache versions
        self._cleanup_old_cache(file_path)

        # 💾 Save cache
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Cache saved: {cache_path}")
        logger.info("Pipeline completed successfully")

        return results
