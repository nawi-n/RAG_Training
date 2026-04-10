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

        for i, text in enumerate(texts):
            logger.debug(f"Embedding chunk {i + 1}/{len(texts)}")

            try:
                result = self.client.feature_extraction(
                    text,
                    model=self.model,
                )

                # ✅ Normalize output
                if hasattr(result, "tolist"):  # numpy array
                    embedding = result.tolist()

                elif isinstance(result, list):
                    embedding = result

                else:
                    raise ValueError(f"Unexpected embedding format: {type(result)}")

                # ✅ Validate final format
                if not isinstance(embedding, list):
                    raise ValueError("Embedding is not a list after conversion")

                all_embeddings.append(embedding)

            except Exception as e:
                logger.error(f"Embedding failed for chunk {i}: {e}")
                raise

        logger.info("Embedding generation complete")

        return all_embeddings
