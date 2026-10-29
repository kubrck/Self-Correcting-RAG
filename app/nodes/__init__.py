"""Graph nodes.

Each node is a plain function taking GraphState and returning a partial update.
No node knows what runs after it -- routing lives entirely in graph.py. That
separation is what lets you add a cycle without rewriting the nodes.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.config import settings
from app.llm import (
    answer_grader,
    generator,
    groundedness_grader,
    query_rewriter,
    relevance_grader,
)
from app.state import GraphState
from app.store import get_retriever


def _trace(state: GraphState, node: str, **fields) -> list[dict]:
    return [*state.get("trace", []), {"node": node, **fields}]


def _format(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


# --------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    """Pull candidates from the local corpus."""
    question = state.get("working_question") or state["question"]
    docs = get_retriever().invoke(question)
    return {
        "documents": docs,
        "source": "local",
        "trace": _trace(state, "retrieve", query=question, retrieved=len(docs)),
    }


def grade_documents(state: GraphState) -> dict:
    """Drop irrelevant retrievals before they reach the generator.

    This is the cheapest high-value step in the whole graph. Vector similarity
    returns the k nearest chunks whether or not any of them are actually
    relevant -- there is no "no results" in a similarity search. Filtering here
    stops the generator from being handed plausible-looking noise.
    """
    question = state.get("working_question") or state["question"]
    grader = relevance_grader()

    kept: list[Document] = []
    verdicts: list[dict] = []
    for doc in state.get("documents", []):
        grade = grader.invoke({"question": question, "document": doc.page_content})
        verdicts.append({"relevant": grade.relevant, "reason": grade.reason})
        if grade.relevant:
            kept.append(doc)

    return {
        "documents": kept,
        "trace": _trace(
            state,
            "grade_documents",
            kept=len(kept),
            dropped=len(state.get("documents", [])) - len(kept),
            verdicts=verdicts,
        ),
    }


def web_search(state: GraphState) -> dict:
    """Fall back to the open web when the local corpus comes up short.

    Results are appended to whatever survived grading rather than replacing it,
    which is why the source becomes 'hybrid' when local docs existed.
    """
    from langchain_community.tools.tavily_search import TavilySearchResults

    question = state.get("working_question") or state["question"]
    existing = state.get("documents", [])

    tool = TavilySearchResults(
        max_results=settings.web_results, api_key=settings.tavily_api_key or None
    )
    results = tool.invoke({"query": question})
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"source": r.get("url", "web"), "origin": "web"},
        )
        for r in results
    ]

    return {
        "documents": [*existing, *web_docs],
        "source": "hybrid" if existing else "web",
        "trace": _trace(state, "web_search", added=len(web_docs)),
    }


def generate(state: GraphState) -> dict:
    question = state.get("working_question") or state["question"]
    docs = state.get("documents", [])
    out = generator().invoke(
        {"context": _format(docs), "question": question}
    )
    return {
        "generation": out.content,
        "trace": _trace(state, "generate", context_docs=len(docs)),
    }


def transform_query(state: GraphState) -> dict:
    """Rewrite the query and send it back around the loop."""
    original = state["question"]
    rewritten = query_rewriter().invoke({"question": original}).query
    return {
        "working_question": rewritten,
        "attempts": state.get("attempts", 0) + 1,
        "trace": _trace(state, "transform_query", rewritten=rewritten),
    }


def abstain(state: GraphState) -> dict:
    """Terminal node for questions the system cannot answer.

    Having this as an explicit node rather than letting a bad answer through is
    the whole point of the grading loop. A RAG system that always answers is
    indistinguishable from one that sometimes hallucinates.
    """
    return {
        "generation": (
            "I could not find enough supporting information to answer that "
            "confidently. The source material does not appear to cover it."
        ),
        "abstained": True,
        "trace": _trace(state, "abstain", attempts=state.get("attempts", 0)),
    }


# --------------------------------------------------------------------------
# Conditional edges -- these return the name of the next node
# --------------------------------------------------------------------------

def decide_after_grading(state: GraphState) -> str:
    """Enough relevant local context, or do we need the web?"""
    docs = state.get("documents", [])
    trace = state.get("trace", [])
    retrieved = next(
        (e["retrieved"] for e in reversed(trace) if e["node"] == "retrieve"), 0
    )

    if retrieved == 0:
        ratio = 0.0
    else:
        ratio = len(docs) / retrieved

    if ratio >= settings.sufficiency_ratio and docs:
        return "generate"
    if settings.web_search_enabled:
        return "web_search"
    # No web fallback configured: generate from whatever survived, and let the
    # groundedness check catch it if the answer is unsupported.
    return "generate" if docs else "abstain"


def decide_after_generation(state: GraphState) -> str:
    """Two checks in sequence, and they route to different places.

    Ungrounded -> regenerate from the same documents. The retrieval was fine;
    the model overreached.

    Grounded but off-target -> rewrite the query. The model answered honestly
    from what it was given, so the problem is upstream in retrieval.

    Collapsing these into one check is the most common mistake in this design,
    and it makes the loop retry the wrong stage.
    """
    docs = state.get("documents", [])
    generation = state.get("generation", "")
    attempts = state.get("attempts", 0)

    grounded = groundedness_grader().invoke(
        {"documents": _format(docs), "generation": generation}
    )
    if not grounded.grounded:
        if attempts >= settings.max_attempts:
            return "abstain"
        return "generate"

    answers = answer_grader().invoke(
        {"question": state["question"], "generation": generation}
    )
    if answers.addresses_question:
        return "done"

    if attempts >= settings.max_attempts:
        return "abstain"
    return "transform_query"
