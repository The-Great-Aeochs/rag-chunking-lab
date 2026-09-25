"""
Recursive Character Text Splitter

Tries separators in order: paragraph → newline → space → character.
Backs off to smaller separators only when chunks exceed the size limit.
Always respects chunk_size strictly.
"""

from shared.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def _split_on(text, separator):
    if not separator:
        return list(text)
    return text.split(separator)


def _recursive_split(text, separators, chunk_size):
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if separators else []
    pieces = _split_on(text, sep)

    chunks = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + sep + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(piece) > chunk_size and remaining_seps:
                chunks.extend(_recursive_split(piece, remaining_seps, chunk_size))
            else:
                chunks.append(piece)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _add_overlap(chunks, overlap):
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prefix = chunks[i - 1][-overlap:]
        result.append(prefix + chunks[i])
    return result


SEPARATORS = ["\n\n", "\n", " ", ""]


def chunk(pages, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    results = []
    for page in pages:
        text = page["page_content"]
        raw_chunks = _recursive_split(text, SEPARATORS, chunk_size)
        raw_chunks = _add_overlap(raw_chunks, chunk_overlap)
        for c in raw_chunks:
            c = c.strip()
            if len(c) < 20:
                continue
            results.append({
                "text": c,
                "metadata": {**page["metadata"], "chunker": "recursive"},
            })
    return results
