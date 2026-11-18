from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag import answer

app = FastAPI(title="RAG Chatbot API")

class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(6, ge=1, le=10)

class Source(BaseModel):
    source: str
    page: int

class Answer(BaseModel):
    answer: str
    sources: list[Source]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask(body: Question):
    try:
        reply, sources = answer(body.question, body.k)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {str(e)[:150]}")
    unique = list(dict.fromkeys(sources))   # remove duplicate pages, keep order
    return Answer(answer=reply, sources=[Source(source=s, page=p) for s, p in unique])