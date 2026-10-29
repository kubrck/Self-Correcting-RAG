from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- models ---
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # Graders run at temperature 0. A grader that gives different verdicts on
    # the same input makes the whole loop non-reproducible.
    grader_temperature: float = 0.0
    generator_temperature: float = 0.0

    # --- embeddings / store ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/scrag"
    collection_name: str = "scrag_docs"

    # --- retrieval ---
    retrieve_k: int = 6
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Fraction of retrieved docs that must pass the relevance grader for the
    # local corpus to be considered sufficient. Below this we fall back to web
    # search. 0.5 is a starting point, not a derived value -- tune it against
    # your own corpus and watch what it does to the web-search rate.
    sufficiency_ratio: float = 0.5

    # --- web fallback ---
    tavily_api_key: str = ""
    web_results: int = 3
    web_search_enabled: bool = True

    # --- loop control ---
    # Each attempt is a full retrieve + grade + generate + grade cycle, so this
    # bounds both latency and cost. 2 rewrites is usually enough; if it isn't,
    # the corpus probably doesn't contain the answer.
    max_attempts: int = 3


settings = Settings()
