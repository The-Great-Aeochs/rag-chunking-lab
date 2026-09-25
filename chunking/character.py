"""
Character Text Splitter

Splits only on a single separator (default: paragraph break).
Merges small pieces up toward chunk_size. Falls back to fixed-width
character slices when a piece exceeds the limit.
"""

from shared.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def chunk(
    pages,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    separator="\n\n",
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2

    raw_limit = chunk_size - chunk_overlap
    results = []
    for page in pages:
        text = page["page_content"]
        raw_chunks = []
        current = ""
        for i, piece in enumerate(text.split(separator)):
            segment = (separator if i else "") + piece
            if current and len(current) + len(segment) > raw_limit:
                raw_chunks.append(current)
                current = ""
            while len(segment) > raw_limit:
                raw_chunks.append(segment[:raw_limit])
                segment = segment[raw_limit:]
            current = segment
        if current:
            raw_chunks.append(current)

        for i, raw in enumerate(raw_chunks):
            prefix = raw_chunks[i - 1][-chunk_overlap:] if i and chunk_overlap else ""
            results.append({
                "text": (prefix + raw).strip(),
                "metadata": {**page["metadata"], "chunker": "character"},
            })
    return [r for r in results if len(r["text"]) >= 20]
