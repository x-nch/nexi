from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEXI_")

    # Prometheus instrumentation
    metrics_enabled: bool = True
    metrics_allow_cidrs: list[str] = Field(default_factory=lambda: ["127.0.0.1", "::1", "192.168.50.0/24"])

    # xnch
    xnch_base_url: str = "http://localhost:8001"
    xnch_service_key: str = ""
    xnch_public_key_path: str = "~/.xnch/keys/public.pem"

    # Provider routing — which backend serves a request.
    #   "opencode"    → OpenCode Go (hosted DeepSeek V4) via OPENCODE_GO_API_URL
    #   "openrouter"  → OpenRouter (any model) via NEXI_OPENROUTER_API_URL
    #   "nexi-default"→ alias resolved to `default_provider` at call time.
    # Nexi is the default brains: chat + internals (intent, options, reflection,
    # evaluator) route through nexi's model_router, which picks the provider.
    default_provider: str = "nexi-default"
    # The provider nexi-default resolves to (opencode | openrouter).
    nexi_default_resolves_to: str = "openrouter"
    # Dynamic model selection budget. One of: cheap | balanced | quality.
    model_budget: str = "balanced"
    # Model selection method. One of: "static" (env-based resolve) | "auto" (Redis rankings).
    model_method: str = "static"
    options_count: int = 5

    # OpenCode Go API (hosted DeepSeek V4)
    opencode_go_api_url: str = "https://opencode.ai/zen/go/v1"
    opencode_go_api_key: str = ""
    opencode_go_api_timeout_s: float = 60.0
    model_id: str = "deepseek-v4-pro"
    # Optional override of the opencode-go model catalog (list of dicts with
    # id/cost_tier/context_window/strengths/latency_ms/description). Empty = defaults.
    opencode_go_models: list = Field(default_factory=list)

    # OpenRouter API (agentic + internet-facing inference)
    openrouter_api_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    openrouter_api_timeout_s: float = 90.0
    openrouter_default_model: str = "anthropic/claude-sonnet-4"
    # Optional override of the openrouter model catalog (list of dicts with
    # id/cost_tier/context_window/strengths/latency_ms/description).
    openrouter_models: list = Field(default_factory=list)

    # Intent classification and reflection models (default to the router, so
    # "nexi-default" delegates to the default provider's router selection).
    intent_classifier_model: str = "nexi-default"
    reflection_model: str = "nexi-default"
    reflection_enabled: bool = True

    # Local provider: LiteLLM proxy in front of the local vLLM (node-b ornith).
    # When litellm_proxy_url is set, nexi-default resolves to the litellm provider
    # and chat/internals are served by the local model, with OpenRouter (free
    # models only) as cross-provider fallback.
    litellm_proxy_url: str = ""
    litellm_proxy_timeout_s: float = 60.0
    litellm_api_key: str = ""
    litellm_model_id: str = "ornith"
    litellm_models: list = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        # Explicitly read from os.environ as fallback for pydantic-settings
        self.litellm_proxy_url = os.environ.get("NEXI_LITELLM_PROXY_URL", self.litellm_proxy_url)
        self.litellm_api_key = os.environ.get("NEXI_LITELLM_API_KEY", self.litellm_api_key)
        self.nexi_default_resolves_to = os.environ.get("NEXI_NEXI_DEFAULT_RESOLVES_TO", self.nexi_default_resolves_to)
        self.xnch_base_url = os.environ.get("NEXI_XNCH_BASE_URL", self.xnch_base_url)
        self.model_method = os.environ.get("NEXI_MODEL_METHOD", self.model_method)
        val = os.environ.get("NEXI_WORKFLOW_EXECUTOR_ENABLED")
        if val is not None:
            self.workflow_executor_enabled = val.lower() == "true"
        self.xnch_service_key = os.environ.get("NEXI_XNCH_SERVICE_KEY", self.xnch_service_key)

    # Legacy: local vLLM (kept for rollback + persona probing)
    vllm_primary_url: str = ""
    vllm_primary_timeout_s: float = 30.0
    vllm_secondary_url: str = ""
    vllm_secondary_timeout_s: float = 45.0

    # OpenRouter free-model fallback (used only when the primary provider errors,
    # and only with `:free`-suffixed models when no paid key is configured).
    openrouter_free_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    openrouter_free_models: list = Field(default_factory=list)

    # --- Free model selector (CLI writes rankings; runtime module reads) ---
    model_selector_weights: dict[str, float] = Field(
        default_factory=lambda: {"quality": 0.5, "latency": 0.3, "context": 0.2}
    )
    model_selector_config_path: str = "config/model_selector.yaml"
    model_selector_redis_ttl_s: int = 86_400
    model_selector_probe_prompt: str = "Say hello in one sentence."
    model_selector_probe_timeout_s: float = 10.0
    model_selector_probe_runs: int = 3
    model_selector_rankings_key: str = "model_selector:rankings"

    # Session
    session_ttl_s: int = 120
    clarification_ttl_s: int = 120
    execution_token_ttl_ms: int = 30_000

    # Redis (KV cache — shared with xnch)
    redis_url: str = "unix:///tmp/xnch-redis.sock"

    # Execution runner (xnch stub at /execution/execute when no dedicated runner)
    execution_runner_url: str = "http://192.168.50.1:8001/execution"

    # vLLM health check endpoint (used by proactivity engine)
    vllm_health_url: str = "http://192.168.50.2:8082/health"

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

    # Persona self-description auto-refresh
    persona_auto_refresh: bool = True
    persona_generated_path: str = "~/.xnch/nexi-persona.generated.yaml"

    # Goal tracking driver loop
    goal_driver_enabled: bool = False
    goal_poll_interval_s: int = 5

    # Workflow executor (P2): claims APPROVED steps from xnch and runs them
    # through the pipeline. Requires xnch side workflow_executor_enabled=True.
    workflow_executor_enabled: bool = False
    workflow_poll_interval_s: int = 5
    goal_default_max_steps: int = 10
    goal_default_failure_threshold: int = 3
    goal_max_consecutive_step_errors: int = 3


settings = Settings()
