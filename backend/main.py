"""FastAPI service exposing retrieval inspection and full RAG answering."""

from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .llm_client import generate_answer
from .rag_engine import RAGEngine

app = FastAPI(
    title="BGE-FAISS RAG API",
    description="Retrieval-augmented QA over a small technical corpus.",
    version="1.0.0",
)

_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    """Load the index once, on first request."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(config.TOP_K, ge=1, le=20, description="Number of chunks to retrieve")


class RetrievedChunk(BaseModel):
    score: float
    text: str
    meta: dict


class RetrieveResponse(BaseModel):
    results: List[RetrievedChunk]


class AskResponse(BaseModel):
    question: str
    top_k: int
    answer: str
    retrieved: List[RetrievedChunk]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: AskRequest) -> RetrieveResponse:
    """Return the evidence only — useful for debugging retrieval on its own."""
    try:
        results = get_engine().retrieve(req.question, top_k=req.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RetrieveResponse(results=results)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Retrieve evidence, build a grounded prompt and generate an answer."""
    try:
        engine = get_engine()
        retrieved = engine.retrieve(req.question, top_k=req.top_k)
        prompt = engine.build_prompt(req.question, retrieved)
        answer = generate_answer(prompt)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        question=req.question, top_k=req.top_k, answer=answer, retrieved=retrieved
    )
