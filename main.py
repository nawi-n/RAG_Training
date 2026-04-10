# main.py

from app.logger import get_logger
from app.pipeline.ingestion_pipeline import IngestionPipeline

logger = get_logger()


def main():
    pipeline = IngestionPipeline()

    file_path = "data/sample.pdf"  # change as needed

    results = pipeline.run(file_path)

    for i, (chunk, embedding) in enumerate(results[:3]):
        logger.info(f"Chunk {i}: {chunk[:80]}")
        logger.info(f"Embedding dim: {len(embedding)}")


if __name__ == "__main__":
    main()
