# app/parsers/html_parser.py
from bs4 import BeautifulSoup

from app.logger import get_logger

logger = get_logger()


def parse_html(file_path: str) -> str:
    logger.info(f"Parsing HTML: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    text = soup.get_text(separator="\n")

    logger.debug("HTML parsing complete")

    return text
