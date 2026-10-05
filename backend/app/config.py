"""Application settings, read from the environment (and ../.env or .env for host runs)."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Request
from psycopg.conninfo import make_conninfo
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

APP_ROLE = "rag_app"


def fingerprint(
    model: str,
    doc_prefix: str = "",
    doc_extra_body: dict[str, Any] | None = None,
    query_prefix: str = "",
    query_extra_body: dict[str, Any] | None = None,
) -> str:
    """`model|doc_prefix|json(doc_extra_body)|query_prefix|json(query_extra_body)`.

    The query side is included because the KB's stored query canary is bound to it: changing only the query
    prefix would otherwise leave every question failing the canary check with no reindex to clear it.
    """
    doc_json = json.dumps(doc_extra_body or {}, sort_keys=True)
    query_json = json.dumps(query_extra_body or {}, sort_keys=True)
    return f"{model}|{doc_prefix}|{doc_json}|{query_prefix}|{query_json}"


class Settings(BaseSettings):
    # env_ignore_empty: `FOO=` in .env or compose counts as unset (matters for the JSON settings below).
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore", env_ignore_empty=True)

    # Database. Host defaults match the compose port mapping; the api container overrides them.
    db_host: str = "127.0.0.1"
    db_port: int = 5433
    postgres_db: str = "chatai"
    postgres_user: str = "postgres"  # owner role, used for migrations only
    postgres_password: SecretStr | None = (
        None  # only `app.cli migrate` needs it (the migrate service); never api/worker
    )
    rag_app_password: SecretStr  # password of the rag_app role the app connects as
    embedding_dim: int = Field(gt=0, le=16000)  # pgvector's vector type caps at 16000

    # Embeddings (OpenAI-compatible /embeddings). The prefixes are text prepended to every input of that mode;
    # the extra bodies are JSON merged into the request body (e.g. NVIDIA {"input_type": "passage"} / "query").
    embedding_base_url: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = ""
    embedding_doc_prefix: str = ""
    embedding_query_prefix: str = ""
    embedding_doc_extra_body: Annotated[dict[str, Any], NoDecode] = {}
    embedding_query_extra_body: Annotated[dict[str, Any], NoDecode] = {}
    embedding_batch: int = Field(default=16, gt=0, le=256)  # texts per request (the canary rides along as one more)

    # LLM (answer generation).
    llm_protocol: Literal["openai", "anthropic"] = "openai"
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_max_tokens: int = 2048  # reasoning models spend completion tokens before they answer
    llm_extra_body: Annotated[dict[str, Any], NoDecode] = {}  # JSON in the env var, passed as the SDK's extra_body
    # Merged over LLM_EXTRA_BODY (top-level keys replace) for the follow-up question rewrite only, which needs no
    # reasoning; e.g. NVIDIA Nemotron {"chat_template_kwargs": {"enable_thinking": false}}. Empty: same as answers.
    llm_rewrite_extra_body: Annotated[dict[str, Any], NoDecode] = {}

    # Optional OpenAI-style /rerank endpoint; unset RERANK_MODEL keeps the fused order.
    rerank_base_url: str = ""
    rerank_api_key: SecretStr = SecretStr("")
    rerank_model: str = ""

    # Retrieval and limits.
    top_k: int = 8
    # Hybrid ranking: weight of the normalised vector score against the normalised lexical score. Tied to the
    # embedding model; re-tune with eval/run_eval.py when the model changes.
    fusion_vector_weight: float = Field(default=0.7, ge=0, le=1)
    min_similarity: float = 0.15  # calibration knob, tune from the eval run (the defaults match .env.example)
    max_question_chars: int = 1000
    max_upload_mb: int = 50
    max_pdf_pages: int = 1000  # a PDF with more pages fails its job (the corpus' longest has 236)
    max_ocr_pages: int = 300  # a PDF needing OCR on more pages fails its job; OCR can take seconds per page

    # Files. Uploads live under <storage_dir>/uploads/<kb_id>/.
    storage_dir: Path = Path("/data/knowledges")
    static_dir: Path = Path("/app/static")

    cookie_secure: bool = True  # Secure session cookie; set COOKIE_SECURE=false only for local http development

    @field_validator(
        "embedding_doc_extra_body",
        "embedding_query_extra_body",
        "llm_extra_body",
        "llm_rewrite_extra_body",
        mode="before",
    )
    @classmethod
    def _json_object(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = json.loads(v) if v.strip() else {}
        if not isinstance(v, dict):
            raise ValueError("must be a JSON object")  # noqa: TRY004 - pydantic only turns ValueError into a 422/validation error
        return v

    @property
    def embedding_fingerprint(self) -> str:
        """What a knowledge base's vectors depend on: model and the document-side and query-side text/body knobs.

        Stored per KB and per finished job; a KB whose stored fingerprint differs from this one needs a reindex.
        """
        return fingerprint(
            self.embedding_model,
            self.embedding_doc_prefix,
            self.embedding_doc_extra_body,
            self.embedding_query_prefix,
            self.embedding_query_extra_body,
        )

    def _dsn(self, user: str, password: SecretStr) -> str:
        return make_conninfo(
            host=self.db_host,
            port=self.db_port,
            dbname=self.postgres_db,
            user=user,
            password=password.get_secret_value(),
        )

    @property
    def owner_dsn(self) -> str:
        if self.postgres_password is None:
            raise RuntimeError("POSTGRES_PASSWORD is not set; only `python -m app.cli migrate` needs it (owner role)")
        return self._dsn(self.postgres_user, self.postgres_password)

    @property
    def app_dsn(self) -> str:
        return self._dsn(APP_ROLE, self.rag_app_password)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment


def app_settings(request: Request) -> Settings:
    """FastAPI dependency: the Settings the app was created with."""
    return request.app.state.settings
