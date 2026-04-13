def select_retrieval_strategy(query: str) -> str:
    """Choose a retrieval strategy from: vector, keyword, hybrid, skip."""
    q = query.strip().lower()

    if not q:
        return "skip"

    keyword_markers = [
        "exact",
        "keyword",
        "verbatim",
        "match",
        "regex",
        "id:",
        "code:",
    ]
    broad_markers = [
        "latest",
        "news",
        "today",
        "current",
        "recent",
        "web",
        "online",
    ]

    if any(marker in q for marker in broad_markers):
        return "hybrid"

    if any(marker in q for marker in keyword_markers):
        return "keyword"

    # Default to semantic retrieval for normal QA.
    return "vector"
