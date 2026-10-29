"""Routing tests.

These exercise the decision functions directly with fabricated state, so they
run without an LLM, a database, or network. The routing logic is where the
bugs live -- the nodes themselves are thin wrappers around model calls.
"""

from langchain_core.documents import Document

from app.config import settings
from app.nodes import decide_after_grading


def _doc(text="x"):
    return Document(page_content=text)


def test_sufficient_local_docs_go_straight_to_generate():
    state = {
        "documents": [_doc(), _doc(), _doc()],
        "trace": [{"node": "retrieve", "retrieved": 4}],
    }
    assert decide_after_grading(state) == "generate"


def test_mostly_filtered_out_falls_back_to_web():
    # 1 of 6 survived -> ratio 0.17, well under the 0.5 threshold
    state = {
        "documents": [_doc()],
        "trace": [{"node": "retrieve", "retrieved": 6}],
    }
    assert decide_after_grading(state) == "web_search"


def test_nothing_relevant_falls_back_to_web():
    state = {"documents": [], "trace": [{"node": "retrieve", "retrieved": 5}]}
    assert decide_after_grading(state) == "web_search"


def test_abstains_when_web_disabled_and_nothing_relevant(monkeypatch):
    monkeypatch.setattr(settings, "web_search_enabled", False)
    state = {"documents": [], "trace": [{"node": "retrieve", "retrieved": 5}]}
    assert decide_after_grading(state) == "abstain"


def test_empty_retrieval_does_not_divide_by_zero():
    state = {"documents": [], "trace": [{"node": "retrieve", "retrieved": 0}]}
    assert decide_after_grading(state) == "web_search"
