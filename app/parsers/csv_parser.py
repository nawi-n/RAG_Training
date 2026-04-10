# app/parsers/csv_parser.py
import pandas as pd

from app.logger import get_logger

logger = get_logger()


def parse_csv(file_path: str) -> str:
    logger.info(f"Parsing CSV: {file_path}")

    df = pd.read_csv(file_path)

    # Better than raw join → preserves row meaning
    rows = df.astype(str).to_dict(orient="records")

    text = "\n".join([str(row) for row in rows])

    logger.debug(f"CSV parsed with {len(rows)} rows")

    return text
