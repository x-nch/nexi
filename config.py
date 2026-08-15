from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEXI_", env_file=".env")

    # xnch
    xnch_base_url: str = "http://localhost:8001"
    xnch_public_key_path: str = "~/.xnch/keys/public.pem"

    # Model adapter
    vllm_primary_url: str = "http://192.168.50.2:8083/v1"
    vllm_primary_timeout_s: float = 30.0
    vllm_secondary_url: str = ""
    vllm_secondary_timeout_s: float = 45.0
    model_id: str = "qwen2.5-vl-7b"
    options_count: int = 5

    # LiteLLM proxy
    litellm_proxy_url: str = "http://localhost:4000/v1"
    litellm_proxy_timeout_s: float = 60.0
    intent_classifier_model: str = "qwen2.5-vl-7b"

    # Session
    session_ttl_s: int = 120
    clarification_ttl_s: int = 120
    execution_token_ttl_ms: int = 30_000

    # Redis (KV cache — shared with xnch)
    redis_url: str = "unix:///tmp/xnch-redis.sock"

    # Execution runner (xnch stub at /execution/execute when no dedicated runner)
    execution_runner_url: str = "http://192.168.50.1:8001/execution"

    # vLLM health check endpoint (used by proactivity engine)
    vllm_health_url: str = "http://192.168.50.2:8083/health"

    # Audit
    audit_events_path: str = "~/.xnch/audit/events.jsonl"

    # Capability / infra auto-refresh
    capabilities_generated_path: str = "~/.xnch/nexi-capabilities.generated.yaml"
    mcp_servers_path: str = "~/.xnch/mcp-servers.yaml"
    infra_manifests_path: Path = Path(__file__).resolve().parents[1] / "infra" / "no-k3s"
    exec_policy_path: str = "~/.xnch/exec-policy.yaml"
    fs_policy_path: str = "~/.xnch/fs-policy.yaml"
    capability_refresh_interval_s: int = 300
    probe_interval_s: int = 60
    probe_timeout_s: float = 2.0
    xnch_tools_endpoint: str = "/nexi/tools"
    capability_auto_refresh: bool = True


settings = Settings()
