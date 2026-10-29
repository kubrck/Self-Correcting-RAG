"""The state object that flows through the graph.

Every node receives this dict and returns a partial update. LangGraph merges
the update into the running state. Keeping all state in one typed structure is
what makes the cycles tractable -- a node can look at `attempts` and decide to
stop without needing to know who called it.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # --- input ---
    question: str
    # The question as currently phrased. transform_query rewrites this; the
    # original stays in `question` so we can show the user what they asked.
    working_question: str

    # --- retrieval ---
    documents: list[Document]
    source: Literal["local", "hybrid", "web"]

    # --- generation ---
    generation: str

    # --- loop control ---
    # Incremented by transform_query. Without a ceiling, a question the corpus
    # genuinely cannot answer will loop forever: retrieval returns nothing
    # relevant, the grader rejects it, the query is rewritten, repeat. The
    # abstain path exists precisely for that case.
    attempts: int
    # Set when we give up and tell the user we don't know, rather than
    # generating something unsupported.
    abstained: bool

    # --- observability ---
    # Append-only record of which nodes ran and what they decided. This is the
    # difference between a demo and something you can debug.
    trace: list[dict]
