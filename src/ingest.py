import re
from pathlib import Path
from pypdf import PdfReader

def load_pdfs(folder: str) -> list[dict]:
    """Return a list of {text, source, page} for every page of every PDF."""
    pages = []
    for pdf_path in Path(folder).glob("*.pdf"):
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r"[^\S\n]+", " ", text)   # collapse any whitespace except line breaks
            if text.strip():
                pages.append({"text": text, "source": pdf_path.name, "page": i + 1})
    return pages

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def build_chunks(folder: str) -> list[dict]:
    all_chunks = []
    for page in load_pdfs(folder):
        for j, chunk in enumerate(chunk_text(page["text"])):
            all_chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"],
                "id": f'{page["source"]}-p{page["page"]}-c{j}',
            })
    return all_chunks

if __name__ == "__main__":
    chunks = build_chunks("data")
    print(f"Created {len(chunks)} chunks")
    print(chunks[0])