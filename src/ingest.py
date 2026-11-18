import re
from pathlib import Path
from pypdf import PdfReader

def load_pdf_pages(source, name: str, max_pages: int | None = None) -> list[dict]:
    """Extract cleaned text page by page from a file path or an uploaded file."""
    reader = PdfReader(source)
    if max_pages is not None and len(reader.pages) > max_pages:
        raise ValueError("too many pages")
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"[^\S\n]+", " ", text)   # collapse spaces/tabs, keep line breaks
        if text.strip():
            pages.append({"text": text, "source": name, "page": i + 1})
    return pages

def load_pdfs(folder: str) -> list[dict]:
    """Return a list of {text, source, page} for every page of every PDF in a folder."""
    pages = []
    for pdf_path in Path(folder).glob("*.pdf"):
        pages.extend(load_pdf_pages(pdf_path, pdf_path.name))
    return pages

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def chunks_from_pages(pages: list[dict]) -> list[dict]:
    all_chunks = []
    for page in pages:
        for j, chunk in enumerate(chunk_text(page["text"])):
            all_chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"],
                "id": f'{page["source"]}-p{page["page"]}-c{j}',
            })
    return all_chunks

def build_chunks(folder: str) -> list[dict]:
    return chunks_from_pages(load_pdfs(folder))

if __name__ == "__main__":
    chunks = build_chunks("data")
    print(f"Created {len(chunks)} chunks")
    if chunks:
        print(chunks[0])