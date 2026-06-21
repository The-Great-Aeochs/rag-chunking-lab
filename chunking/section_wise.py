"""
Section-Wise Chunker for Research Papers

Detects section headers (Abstract, Introduction, Method, etc.) via regex.
Joins pages per-paper so sections spanning page breaks stay whole.
Re-splits long sections with recursive fallback to fit the embedding window.
"""

import re

HEADER = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(Abstract|Introduction|Related Work|Background|Method(?:s|ology)?|Approach"
    r"|Experiments?|Results|Evaluation(?:\s+Strategies)?|Discussion|Limitations"
    r"|Conclusions?|References|Acknowledgements?"
    r"|Model Architecture|Training|Attention"
    r"|(?:The\s+)?Wiki\s*Eval(?:\s+Dataset)?|Dataset"
    r"|Retrieval|Generation|Implementation|Setup|Analysis"
    r"|Preliminaries|Problem\s+(?:Setup|Statement|Definition|Formulation))\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _recursive_resplit(text, max_chars, overlap):
    if len(text) <= max_chars:
        return [text]
    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks, current = [], ""
            for part in parts:
                candidate = part if not current else current + sep + part
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part
            if current:
                chunks.append(current)
            if all(len(c) <= max_chars * 1.2 for c in chunks):
                result = []
                for i, c in enumerate(chunks):
                    if i > 0 and overlap > 0:
                        c = chunks[i - 1][-overlap:] + c
                    result.append(c)
                return result
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars - overlap)]


def chunk(pages, chunk_size=800, chunk_overlap=80):
    by_source = {}
    for p in pages:
        by_source.setdefault(p["metadata"]["source"], []).append(p["page_content"])

    # Only re-split sections longer than this. Short-to-medium sections
    # (Abstract, Conclusions, Background) stay whole — that's the point
    # of section-aware chunking. 
    resplit_threshold = max(chunk_size * 3, 2500)

    results = []
    for source, page_texts in by_source.items():
        full_text = "\n".join(page_texts)
        matches = list(HEADER.finditer(full_text))
        starts = [0] + [m.start() for m in matches]
        names = ["Front Matter"] + [m.group(1).title() for m in matches]

        for name, start, end in zip(names, starts, starts[1:] + [len(full_text)]):
            section_text = full_text[start:end].strip()
            if len(section_text) < 50:
                continue

            if len(section_text) <= resplit_threshold:
                results.append({
                    "text": section_text,
                    "metadata": {
                        "source": source,
                        "section": name,
                        "chunker": "section_wise",
                    },
                })
            else:
                for piece in _recursive_resplit(section_text, chunk_size, chunk_overlap):
                    piece = piece.strip()
                    if len(piece) < 20:
                        continue
                    results.append({
                        "text": piece,
                        "metadata": {
                            "source": source,
                            "section": name,
                            "chunker": "section_wise",
                        },
                    })
    return results
