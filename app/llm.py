"""Models and graders.

Every grader returns a structured object rather than free text. Asking an LLM
"is this relevant? answer yes or no" and parsing the string works until it
answers "Yes, because..." -- structured output removes that failure mode
entirely, which matters when the grader's verdict controls a branch.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings


# --------------------------------------------------------------------------
# Structured verdicts
# --------------------------------------------------------------------------

class RelevanceGrade(BaseModel):
    """Is one retrieved document relevant to the question?"""

    relevant: bool = Field(description="True if the document helps answer the question")
    reason: str = Field(description="One short sentence justifying the verdict")


class GroundednessGrade(BaseModel):
    """Is the generated answer supported by the retrieved documents?

    This is the hallucination check. It deliberately asks only about support,
    not about correctness -- an answer can be factually true and still be
    ungrounded, which for a RAG system is a failure.
    """

    grounded: bool = Field(
        description="True if every claim in the answer is supported by the documents"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the answer not supported by the documents",
    )


class AnswerGrade(BaseModel):
    """Does the answer actually address what was asked?

    Separate from groundedness because the two fail independently. An answer
    can be perfectly grounded in retrieved text and still be about the wrong
    thing -- that is a retrieval failure, and it needs a query rewrite rather
    than a regeneration.
    """

    addresses_question: bool = Field(
        description="True if the answer resolves the user's question"
    )
    reason: str = Field(description="One short sentence justifying the verdict")


class RewrittenQuery(BaseModel):
    query: str = Field(description="The rewritten search query")


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _chat(temperature: float):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=settings.openai_api_key or None,
    )


@lru_cache(maxsize=None)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
    )


def _grader(schema, system: str, human: str):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)]
    )
    return prompt | _chat(settings.grader_temperature).with_structured_output(schema)


# --------------------------------------------------------------------------
# Graders
# --------------------------------------------------------------------------

def relevance_grader():
    return _grader(
        RelevanceGrade,
        (
            "You grade whether a retrieved document is relevant to a user question.\n"
            "Be lenient: a document is relevant if it contains information that "
            "contributes to an answer, even partially. The goal is to filter out "
            "clearly unrelated retrievals, not to demand a complete answer from "
            "every chunk."
        ),
        "Question:\n{question}\n\nDocument:\n{document}",
    )


def groundedness_grader():
    return _grader(
        GroundednessGrade,
        (
            "You check whether an answer is supported by the given documents.\n"
            "Mark it ungrounded if it asserts anything the documents do not "
            "state or clearly imply. Do not judge whether the answer is true in "
            "general -- only whether these documents support it. An answer that "
            "correctly says the documents do not contain the information IS "
            "grounded."
        ),
        "Documents:\n{documents}\n\nAnswer:\n{generation}",
    )


def answer_grader():
    return _grader(
        AnswerGrade,
        (
            "You check whether an answer actually resolves the user's question.\n"
            "An answer that is on-topic but does not resolve the question fails. "
            "An explicit, honest 'the source material does not cover this' "
            "counts as resolving it."
        ),
        "Question:\n{question}\n\nAnswer:\n{generation}",
    )


def query_rewriter():
    return _grader(
        RewrittenQuery,
        (
            "You rewrite a question into a better retrieval query.\n"
            "The previous attempt retrieved nothing useful, so rephrasing alone "
            "will not help. Change the angle: expand abbreviations, add likely "
            "synonyms and domain terms, make implicit subjects explicit, and "
            "strip conversational framing. Output the query only."
        ),
        "Original question:\n{question}\n\nThis retrieved nothing relevant. Rewrite it.",
    )


def generator():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only the context below.\n"
                "If the context does not contain the answer, say so plainly "
                "instead of guessing -- an honest 'not covered here' is a "
                "correct response. Cite nothing you cannot point to in the "
                "context. Be concise.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    return prompt | _chat(settings.generator_temperature)
