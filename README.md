# Self-Correcting RAG

A retrieval-augmented generation pipeline built on LangGraph that grades its own
retrieval and its own output, and loops back when either fails — instead of
answering from whatever the vector search happened to return.

## The problem this solves

A standard RAG pipeline is linear: embed the question, retrieve the top *k*
chunks, stuff them into a prompt, generate. It has no way to notice that
retrieval failed, because a similarity search always returns *k* results. There
is no "no match" — there is only "the nearest *k*, however far away they are."

So when the corpus doesn't contain the answer, the generator receives *k* chunks
of unrelated text and writes a confident answer anyway. That is the single most
common failure mode in production RAG, and it is invisible: nothing errors,
nothing logs a warning.

This pipeline adds three checks and two cycles.

## Graph

```
         retrieve
            |
      grade_documents
            |
     +------+------------------+
     |                         |
  enough relevant?        not enough?
     |                         |
     |                    web_search
     |                         |
     +-----------+-------------+
                 |
              generate
                 |
      +----------+-----------+
      |          |           |
 ungrounded  off-target    good
      |          |           |
   generate  transform_query END
                 |
             retrieve  (cycle)
```

### 1. Document grading

Each retrieved chunk is graded for relevance before it reaches the generator.
Irrelevant chunks are dropped. If fewer than `SUFFICIENCY_RATIO` of them survive,
the local corpus is judged insufficient and the pipeline falls back to web search.

The grader is deliberately lenient — it filters out clearly unrelated retrievals
rather than demanding a complete answer from each chunk. A strict grader
discards partial evidence that would have combined into a good answer.

### 2. Groundedness check

After generation, the answer is checked against the retrieved documents. Does
every claim trace back to the context? This is the hallucination gate.

It asks only about *support*, not *truth*. An answer can be factually correct
and still ungrounded — for a RAG system that is a failure, because the system
got lucky rather than being right for the right reason.

### 3. Answer check

Separately: does the answer actually resolve what was asked?

**Keeping these two checks separate is the central design decision.** They fail
independently and need different responses:

| Failure | Cause | Response |
|---|---|---|
| Ungrounded | Model overreached past its context | Regenerate from the same docs |
| Grounded but off-target | Retrieval surfaced the wrong material | Rewrite the query, retrieve again |

Collapsing them into one "is this answer good?" check makes the loop retry the
wrong stage — regenerating when the problem was retrieval, or re-retrieving when
the model simply needed to stay closer to its source.

### 4. Abstention

After `MAX_ATTEMPTS` rewrites, the pipeline stops and says it doesn't know.

This is not a fallback, it is the point. A system that always produces an answer
is indistinguishable from one that sometimes hallucinates. The abstain path is
what makes the grading meaningful.

## Design notes

**Why structured output for the graders.** Asking a model "is this relevant,
yes or no" and parsing the string works until it replies "Yes, because…".
Every grader returns a Pydantic schema via function calling, so a verdict that
controls a branch can never be a parsing accident.

**Why graders run at temperature 0.** A grader that gives different verdicts on
identical input makes the entire loop non-reproducible — you cannot debug a
cycle whose exit condition is stochastic.

**Why `attempts` is bounded.** A question the corpus genuinely cannot answer
will loop forever otherwise: retrieve, reject, rewrite, repeat. `MAX_ATTEMPTS`
bounds cost and latency; `recursion_limit` in `graph.py` is a separate hard
backstop in case a routing bug creates a cycle the counter doesn't guard.

**Why pgvector and not a dedicated vector DB.** At this corpus size the
performance difference is irrelevant, and keeping documents and embeddings in a
Postgres you already run removes a service from the deployment. At tens of
millions of vectors this choice reverses.

**Why chunks overlap.** A fact spanning a chunk boundary is invisible to
retrieval without overlap, because neither chunk contains the complete
statement.

## Running it

```bash
docker compose up -d                      # Postgres with pgvector
pip install -r requirements.txt
cp .env.example .env                      # add your OPENAI_API_KEY

python scripts/ingest.py data/            # index .md / .txt files
uvicorn api.main:app --reload
```

```bash
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "...", "include_trace": true}'
```

`include_trace` returns the node-by-node record: what was retrieved, which
chunks were dropped and why, whether web search fired, how many rewrites
happened. Without it you cannot tell a cache hit from a lucky guess.

## Tests

```bash
pytest tests/ -q
```

The routing tests exercise the decision functions with fabricated state, so they
run with no LLM, database, or network. The routing logic is where bugs live —
the nodes themselves are thin wrappers around model calls.

## Tuning

`SUFFICIENCY_RATIO` (default 0.5) is the main knob. Raise it and the pipeline
reaches for web search more eagerly, costing latency and API calls but widening
coverage. Lower it and it trusts a thinner local result set. It is a starting
point, not a derived value — tune it against your own corpus and watch what it
does to the web-search rate.
