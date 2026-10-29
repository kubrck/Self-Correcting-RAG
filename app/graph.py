"""Graph topology.

    retrieve
       |
    grade_documents
       |
       +-- enough relevant? --> generate
       |
       +-- not enough? ------> web_search --> generate
                                                |
                                   +------------+------------+
                                   |            |            |
                              ungrounded    off-target     good
                                   |            |            |
                                generate   transform_query  END
                                                |
                                            retrieve   (cycle)

The two cycles are the point. A linear RAG pipeline answers whatever retrieval
happened to return; this one can reject its own output and try a different
approach, and gives up explicitly when it runs out of attempts.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.nodes import (
    abstain,
    decide_after_generation,
    decide_after_grading,
    generate,
    grade_documents,
    retrieve,
    transform_query,
    web_search,
)
from app.state import GraphState


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate)
    g.add_node("transform_query", transform_query)
    g.add_node("abstain", abstain)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")

    g.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"generate": "generate", "web_search": "web_search", "abstain": "abstain"},
    )
    g.add_edge("web_search", "generate")

    g.add_conditional_edges(
        "generate",
        decide_after_generation,
        {
            "generate": "generate",          # ungrounded -> regenerate
            "transform_query": "transform_query",  # off-target -> new query
            "abstain": "abstain",
            "done": END,
        },
    )

    # The cycle: a rewritten query goes back through retrieval.
    g.add_edge("transform_query", "retrieve")
    g.add_edge("abstain", END)

    return g.compile()


def answer(question: str) -> dict:
    graph = build_graph()
    # recursion_limit is a hard backstop independent of max_attempts -- if a
    # routing bug ever creates a cycle the attempt counter does not guard,
    # this stops it rather than burning tokens forever.
    final = graph.invoke(
        {"question": question, "working_question": question, "attempts": 0, "trace": []},
        config={"recursion_limit": 25},
    )
    return {
        "question": question,
        "answer": final.get("generation", ""),
        "abstained": final.get("abstained", False),
        "source": final.get("source"),
        "attempts": final.get("attempts", 0),
        "trace": final.get("trace", []),
    }
