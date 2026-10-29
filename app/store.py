"""pgvector-backed document store.

Postgres rather than a dedicated vector DB because the corpus here is small
enough that pgvector's performance is irrelevant, and keeping documents and
embeddings in the same database you already run removes an entire service from
the deployment. At tens of millions of vectors this choice flips.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings
from app.llm import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.collection_name,
        connection=settings.database_url,
        use_jsonb=True,
    )


def get_retriever(k: int | None = None):
    return get_vectorstore().as_retriever(
        search_kwargs={"k": k or settings.retrieve_k}
    )


def ingest_paths(paths: list[Path]) -> int:
    """Chunk and embed text files. Returns the number of chunks written."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))

    # Overlap matters: a fact that straddles a chunk boundary is invisible to
    # retrieval without it, because neither chunk contains the whole statement.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    get_vectorstore().add_documents(chunks)
    return len(chunks)
