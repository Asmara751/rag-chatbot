import re
import streamlit as st
from rag import answer

st.set_page_config(page_title="Document Q&A", page_icon="📚")
st.title("📚 Document Q&A")
st.caption(
    "Ask about the VMware ESX memory management paper or the web security slides. "
    "Every answer cites its source page."
)

if "history" not in st.session_state:
    st.session_state.history = []

def format_sources(reply: str) -> str:
    """Show only the pages the answer actually cites, e.g. [file.pdf p.8]."""
    cited = []
    for group in re.findall(r"\[([^\]]*\.pdf[^\]]*)\]", reply):
        name = re.match(r"\s*(.*?\.pdf)", group).group(1)
        cited += [(name, int(p)) for p in re.findall(r"p\.\s*(\d+)", group)]
    cited = sorted(set(cited))
    if not cited:
        return ""   # refusal or no citations: show nothing
    return "Sources: " + ", ".join(f"{name} p.{page}" for name, page in cited)

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        st.write(item["a"])
        if item["s"]:
            st.caption(item["s"])

question = st.chat_input("Ask a question about the documents")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                reply, _retrieved = answer(question, 6)
            except Exception:
                st.error("The AI service is busy right now. Please try again in a minute.")
                st.stop()
        src_text = format_sources(reply)
        st.write(reply)
        if src_text:
            st.caption(src_text)
    st.session_state.history.append({"q": question, "a": reply, "s": src_text})