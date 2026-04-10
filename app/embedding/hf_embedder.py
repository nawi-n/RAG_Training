# app/embedding/hf_embedder.py

from typing import List

from huggingface_hub import InferenceClient

from app.config import HF_API_KEY, HF_MODEL
from app.logger import get_logger

logger = get_logger()


class HFEmbedder:
    def __init__(self):
        # ✅ Using official HF client (no manual URL handling)
        self.client = InferenceClient(
            provider="hf-inference",  # default provider (can change if needed)
            api_key=HF_API_KEY,
        )
        self.model = HF_MODEL

    def embed(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} chunks")

        all_embeddings = []

        # 🔥 Batch for stability
        batch_size = 5

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}")

            try:
                result = self.client.feature_extraction(
                    batch,
                    model=self.model,
                )

                # HF returns:
                # - List[List[float]] for batch
                # - List[float] for single input

                if isinstance(batch, list) and isinstance(result[0], list):
                    batch_embeddings = result
                else:
                    batch_embeddings = [result]

                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Embedding error: {str(e)}")
                raise

        logger.info("Embedding generation complete")

        return all_embeddings
