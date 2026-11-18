import re
import streamlit as st
from ingest import load_pdf_pages, chunks_from_pages
from store import build_memory_index, library_size
from rag import answer

MAX_MB = 10
MAX_PAGES = 200

st.set_page_config(page_title="Document Q&A", page_icon="📚")
st.title("📚 Document Q&A")
st.caption("Ask questions about your PDFs. Every answer cites its source page (the PDF viewer's page number).")
def build_index(files):
    pages, notes, used = [], [], 0
    for f in files:
        if f.size > MAX_MB * 1024 * 1024:
            notes.append(f"{f.name}: skipped, larger than {MAX_MB} MB.")
            continue
        remaining = MAX_PAGES - used
        if remaining <= 0:
            notes.append(f"{f.name}: skipped, page limit reached.")
            continue
        try:
            f.seek(0)
            file_pages = load_pdf_pages(f, f.name, max_pages=remaining)
        except ValueError:
            notes.append(f"{f.name}: skipped, too many pages (limit is {MAX_PAGES} in total).")
            continue
        except Exception:
            notes.append(f"{f.name}: could not be read.")
            continue
        if not file_pages:
            notes.append(f"{f.name}: no selectable text found. Scanned PDFs are not supported.")
            continue
        sample = " ".join(p["text"] for p in file_pages)
        letters = [c for c in sample if c.isalpha()]
        if letters and sum(c.isascii() for c in letters) / len(letters) < 0.5:
            notes.append(
                f"{f.name}: this file does not look like English. Search works best "
                "with English documents, so answers may be poor."
            )
        used += len(file_pages)
        pages += file_pages
    chunks = chunks_from_pages(pages)
    return (build_memory_index(chunks) if chunks else None), notes

def format_sources(reply: str) -> str:
    """Show only the pages the answer actually cites, e.g. [file.pdf p.8, 10]."""
    cited = []
    for group in re.findall(r"\[([^\]]*\.pdf[^\]]*)\]", reply, flags=re.I):
        m = re.match(r"\s*(.*?\.pdf)", group, flags=re.I)
        name = m.group(1)
        cited += [(name, int(p)) for p in re.findall(r"\d+", group[m.end():])]
    cited = sorted(set(cited))
    if not cited:
        return ""
    return "Sources: " + ", ".join(f"{name} p.{page}" for name, page in cited)

# ---------- sidebar: upload ----------
with st.sidebar:
    st.header("Your documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    st.caption(
        f"Up to {MAX_MB} MB per file and {MAX_PAGES} pages in total. "
        "Files are kept in memory for this session only, but the passages used to answer "
        "are sent to Google's Gemini API, so do not upload confidential documents."
    )

signature = tuple(sorted((f.name, f.size) for f in uploads))
if signature != st.session_state.get("signature"):
    with st.spinner("Reading and indexing your files..."):
        idx, notes = build_index(uploads)
    st.session_state.signature = signature
    st.session_state.index = idx
    st.session_state.notes = notes

for note in st.session_state.get("notes", []):
    st.sidebar.warning(note)

index = st.session_state.get("index")
has_library = library_size() > 0

if index is not None:
    n_files = len({c["source"] for c in index["chunks"]})
    st.info(f"Searching your {n_files} uploaded file(s).")
elif has_library:
    st.info("Searching the built-in documents. Upload PDFs in the sidebar to ask about your own.")
else:
    st.info("Upload a PDF in the sidebar to get started.")

# ---------- chat ----------
if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        st.write(item["a"])
        if item["s"]:
            st.caption(item["s"])

question = st.chat_input(
    "Ask a question about the documents",
    disabled=not (index is not None or has_library),
)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                reply, _retrieved = answer(question, 6, index=index)
            except Exception:
                st.error("The AI service is busy right now. Please try again in a minute.")
                st.stop()
        src_text = format_sources(reply)
        st.write(reply)
        if src_text:
            st.caption(src_text)
    st.session_state.history.append({"q": question, "a": reply, "s": src_text})