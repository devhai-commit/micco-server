from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    pages: list[str],
    chunk_size: int = 512,
    overlap: int = 128,
) -> list[dict]:
    """Split a list of page strings into overlapping text chunks.

    Returns:
        List of dicts: [{chunk_index, content, char_start}]
    """
    if not pages:
        return []

    full_text = "\n\n".join(p for p in pages if p.strip())
    if not full_text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    docs = splitter.create_documents([full_text])

    chunks = []
    cursor = 0
    for i, doc in enumerate(docs):
        content = doc.page_content
        char_start = full_text.find(content, cursor)
        if char_start == -1:
            char_start = cursor
        chunks.append({
            "chunk_index": i,
            "content": content,
            "char_start": char_start,
        })
        cursor = char_start + len(content)

    return chunks
