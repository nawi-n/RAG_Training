from typing import List

import requests


def web_search(query: str, max_results: int = 5) -> List[str]:
    """Search DuckDuckGo instant answers and return compact text snippets."""
    if not query.strip():
        return []

    response = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    results: List[str] = []

    abstract = data.get("AbstractText", "").strip()
    if abstract:
        results.append(abstract)

    for item in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break

        if isinstance(item, dict) and "Text" in item:
            text = item.get("Text", "").strip()
            if text:
                results.append(text)

        topics = item.get("Topics") if isinstance(item, dict) else None
        if topics:
            for sub in topics:
                if len(results) >= max_results:
                    break
                text = sub.get("Text", "").strip()
                if text:
                    results.append(text)

    return results[:max_results]
