import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from store import search

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=30000),   # give up after 30 seconds
)

# Tries these models in order if one is busy or unavailable
MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.7-flash",
]

PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the documents."
Cite the sources you used like [file p.page].

Context:
{context}

Question: {question}
Answer:"""

def generate(prompt: str) -> str:
    """Call Gemini with short retries and fallback to other models."""
    last_error = None
    for model in MODELS:
        for attempt in range(2):
            try:
                print(f"(using {model}, attempt {attempt + 1})")
                return client.models.generate_content(model=model, contents=prompt).text
            except Exception as e:
                last_error = e
                code = getattr(e, "code", None)
                print(f"   failed: code={code} {str(e)[:100]}")
                if code in (None, 429, 503):      # busy or timeout: wait a little, retry
                    time.sleep(2 * (attempt + 1))
                else:                             # e.g. 404: skip to the next model
                    break
    raise last_error

def answer(question: str, k: int = 6):
    results = search(question, k)
    context = "\n\n".join(
        f"[{meta['source']} p.{meta['page']}]\n{text}"
        for text, meta, dist in results
    )
    reply = generate(PROMPT.format(context=context, question=question))
    sources = [(meta["source"], meta["page"]) for _, meta, _ in results]
    return reply, sources

if __name__ == "__main__":
    q = input("Ask a question: ")
    reply, sources = answer(q)
    print("\nANSWER:\n", reply)
    print("\nSOURCES:", sources)