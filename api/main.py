"""FastAPI surface."""

from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import answer
from app.store import ingest_paths

app = FastAPI(title="Self-Correcting RAG")


class Query(BaseModel):
    question: str
    include_trace: bool = False


class IngestRequest(BaseModel):
    paths: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: Query):
    result = answer(req.question)
    if not req.include_trace:
        result.pop("trace", None)
    return result


@app.post("/ingest")
def ingest(req: IngestRequest):
    from pathlib import Path

    n = ingest_paths([Path(p) for p in req.paths])
    return {"chunks_indexed": n}
