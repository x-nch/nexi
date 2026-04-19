from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEXI_", env_file=".env")

    # xnch
    xnch_base_url: str = "http://localhost:8001"
    xnch_public_key_path: str = "~/.xnch/keys/public.pem"

    # Model adapter
    vllm_primary_url: str = "http://localhost:8000/v1"
    vllm_primary_timeout_s: float = 30.0
    vllm_secondary_url: str = ""
    vllm_secondary_timeout_s: float = 45.0
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    options_count: int = 5

    # Session
    session_ttl_s: int = 120
    clarification_ttl_s: int = 120
    execution_token_ttl_ms: int = 30_000

    # Redis (KV cache — shared with xnch)
    redis_url: str = "unix:///tmp/xnch-redis.sock"

    # Audit
    audit_events_path: str = "~/.xnch/audit/events.jsonl"


settings = Settings()
