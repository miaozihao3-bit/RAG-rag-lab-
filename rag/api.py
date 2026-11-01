from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import Pipeline

app = FastAPI(title="RAG Lab", version="0.1.0")
_pipeline = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.load_or_build()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class SourceModel(BaseModel):
    source: str
    heading: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceModel]
    refused: bool
    latency_ms: int


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {"status": "ok", "chunks": len(pipeline.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    answer = get_pipeline().ask(request.question, request.top_k)
    return AskResponse(
        answer=answer.text,
        sources=[SourceModel(**source.__dict__) for source in answer.sources],
        refused=answer.refused,
        latency_ms=answer.latency_ms,
    )
