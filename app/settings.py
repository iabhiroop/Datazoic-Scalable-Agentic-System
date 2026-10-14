"""Typed configuration; infrastructure and secret validation comes at startup."""

from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import UUID

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # dotenv loads the environment; Pydantic validates and masks the resulting settings.
    model_config = SettingsConfigDict(env_file=None, env_ignore_empty=True, extra="ignore")

    def __init__(self, _env_file: str | Path | None = ".env", **values: Any) -> None:
        if _env_file is not None:
            load_dotenv(dotenv_path=_env_file, override=False, encoding="utf-8")
        super().__init__(_env_file=None, **values)

    @model_validator(mode="after")
    def validate_versions(self):
        supported_pairs = {
            ("v1", "v1"), ("v2", "v1"), ("v2", "v2"), ("v3", "v2"), ("v4", "v2")
        }
        if (self.graph_version, self.prompt_version) not in supported_pairs:
            raise ValueError("No handler is installed for the configured graph or prompt version")
        if self.llm_provider.strip().lower() == "ollama":
            address = urlsplit(self.ollama_base_url)
            if (
                address.scheme != "http"
                or address.hostname not in {"localhost", "127.0.0.1", "::1"}
                or address.username is not None
                or address.password is not None
                or address.path not in {"", "/"}
                or address.query
                or address.fragment
                or not self.ollama_model_name.strip()
            ):
                raise ValueError("Ollama requires a model and a local HTTP base URL")
        return self

    database_url: SecretStr | None = None
    checkpoint_database_url: SecretStr | None = None
    auth_bindings_json: SecretStr | None = None
    redis_url: SecretStr | None = None
    object_storage_endpoint: str | None = None
    object_storage_bucket: str = "agent-payloads"
    object_storage_access_key: SecretStr | None = None
    object_storage_secret_key: SecretStr | None = None
    document_max_bytes: int = Field(default=10485760, gt=0)
    document_chunk_chars: int = Field(default=1800, gt=0)
    document_chunk_overlap: int = Field(default=200, ge=0)
    document_max_chunks: int = Field(default=256, gt=0)
    model_name: str | None = None
    model_api_key: SecretStr | None = None
    llm_provider: str = "gemini"
    gemini_model_name: str | None = None
    gemini_api_key: SecretStr | None = None
    ollama_model_name: str = "llama3.2"
    ollama_base_url: str = "http://127.0.0.1:11434"
    paypal_sandbox_client_id: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("PAYPAL_SANDBOX_CLIENT_ID", "PAYPAL_CLIENT_ID"),
    )
    paypal_sandbox_client_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("PAYPAL_SANDBOX_CLIENT_SECRET", "PAYPAL_CLIENT_SECRET"),
    )
    paypal_sandbox_tenant_id: UUID | None = None
    paypal_sandbox_writes_enabled: bool = False
    endpoint_catalog_path: Path = Path("runtime/tool-drafts/endpoint-catalog.json")
    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_max_attempts: int = Field(default=3, ge=1, le=10)
    llm_retry_initial_seconds: float = Field(default=10, gt=0, le=60)
    llm_retry_max_seconds: float = Field(default=30, gt=0, le=60)
    llm_thinking_level: Literal["low", "medium", "high"] = "medium"
    embedding_model: str | None = None
    embedding_version: str | None = None
    embedding_dimension: int | None = Field(default=None, gt=0)
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str | None = None
    langfuse_tracing_enabled: bool = True
    langfuse_capture_content: bool = True
    langfuse_environment: str = "development"
    langfuse_timeout_seconds: int = Field(default=5, gt=0, le=30)
    langfuse_max_content_chars: int = Field(default=32000, gt=0)
    graph_version: str = "v4"
    llm_trace_dir: Path = Path("runtime/llm-traces")
    prompt_version: str = "v2"
    policy_version: str = "v1"
    candidate_tools_per_subgoal: int = Field(default=5, gt=0)
    retrieval_branch_limit: int = Field(default=20, gt=0)
    taxonomy_distance_threshold: float = Field(default=0.42, ge=0, le=2)
    knowledge_passages: int = Field(default=4, gt=0)
    trajectory_limit: int = Field(default=2, ge=0)
    max_plan_steps: int = Field(default=10, gt=0)
    max_replans: int = Field(default=2, ge=0)
    max_tool_attempts: int = Field(default=3, gt=0)
    worker_poll_seconds: float = Field(default=1, gt=0)
    worker_heartbeat_seconds: float = Field(default=5, gt=0)
    tool_timeout_seconds: float = Field(default=30, gt=0)
    graph_recursion_limit: int = Field(default=200, gt=0)
    payload_offload_bytes: int = Field(default=65536, gt=0)
    retention_days: int | None = Field(default=None, gt=0)
