"""
IntelliPipe Core Configuration
===============================
Centralized, environment-driven configuration using Pydantic Settings.
Follows 12-factor app principles with full validation and secret masking.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KafkaSettings(BaseSettings):
    """Kafka cluster configuration."""

    model_config = SettingsConfigDict(env_prefix="KAFKA_")

    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Comma-separated Kafka bootstrap servers",
    )
    raw_events_topic: str = Field(default="raw_events")
    dlq_topic: str = Field(default="dead_letter_queue")
    schema_registry_url: str = Field(default="http://localhost:8081")
    consumer_group_id: str = Field(default="intellipipe-consumers")
    auto_offset_reset: str = Field(default="latest")
    enable_auto_commit: bool = Field(default=False)
    max_poll_records: int = Field(default=500)
    session_timeout_ms: int = Field(default=30000)
    security_protocol: str = Field(default="PLAINTEXT")
    sasl_mechanism: Optional[str] = Field(default=None)
    sasl_username: Optional[SecretStr] = Field(default=None)
    sasl_password: Optional[SecretStr] = Field(default=None)


class DatabaseSettings(BaseSettings):
    """PostgreSQL + pgvector configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="intellipipe")
    user: str = Field(default="intellipipe")
    password: SecretStr = Field(default=SecretStr("intellipipe"))
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=10)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=1800)
    echo: bool = Field(default=False)

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def sync_url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class RedisSettings(BaseSettings):
    """Redis configuration for alert queues and caching."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[SecretStr] = Field(default=None)
    max_connections: int = Field(default=50)
    alert_queue_key: str = Field(default="intellipipe:alerts")
    dq_scores_key: str = Field(default="intellipipe:dq_scores")
    cache_ttl_seconds: int = Field(default=300)

    @property
    def url(self) -> str:
        pwd = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{pwd}{self.host}:{self.port}/{self.db}"


class SparkSettings(BaseSettings):
    """PySpark Structured Streaming configuration."""

    model_config = SettingsConfigDict(env_prefix="SPARK_")

    app_name: str = Field(default="IntelliPipe-Streaming")
    master: str = Field(default="local[*]")
    checkpoint_location: str = Field(default="/tmp/intellipipe/checkpoints")
    trigger_interval: str = Field(default="30 seconds")
    watermark_delay: str = Field(default="10 minutes")
    shuffle_partitions: int = Field(default=200)
    max_offsets_per_trigger: int = Field(default=10000)
    starting_offsets: str = Field(default="latest")


class MLflowSettings(BaseSettings):
    """MLflow experiment tracking configuration."""

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_name: str = Field(default="intellipipe-anomaly-detection")
    model_registry_uri: Optional[str] = Field(default=None)
    artifact_root: str = Field(default="s3://intellipipe-mlflow-artifacts")


class LLMSettings(BaseSettings):
    """LLM / Claude API configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    anthropic_api_key: SecretStr = Field(description="Anthropic Claude API key")
    claude_model: str = Field(default="claude-opus-4-5")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    max_retries: int = Field(default=3)
    retry_delay_seconds: float = Field(default=2.0)
    rate_limit_rpm: int = Field(default=60)


class GitHubSettings(BaseSettings):
    """GitHub API configuration for PR automation."""

    model_config = SettingsConfigDict(env_prefix="GITHUB_")

    token: SecretStr = Field(description="GitHub personal access token")
    org: str = Field(default="intellipipe-org")
    repo: str = Field(default="data-platform")
    base_branch: str = Field(default="main")
    pr_reviewers: List[str] = Field(default_factory=list)
    require_approval: bool = Field(default=True)
    auto_merge_on_approval: bool = Field(default=False)

    @field_validator("pr_reviewers", mode="before")
    @classmethod
    def parse_reviewers(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [r.strip() for r in v.split(",") if r.strip()]
        return v


class JiraSettings(BaseSettings):
    """Jira integration configuration."""

    model_config = SettingsConfigDict(env_prefix="JIRA_")

    url: AnyHttpUrl = Field(default="https://intellipipe.atlassian.net")
    email: str = Field(default="data-platform@intellipipe.io")
    api_token: SecretStr = Field(description="Jira API token")
    project_key: str = Field(default="DQAI")
    incident_issue_type: str = Field(default="Incident")
    high_severity_label: str = Field(default="dq-critical")
    auto_assign: bool = Field(default=True)


class SlackSettings(BaseSettings):
    """Slack notification configuration."""

    model_config = SettingsConfigDict(env_prefix="SLACK_")

    bot_token: SecretStr = Field(description="Slack bot OAuth token")
    webhook_url: Optional[SecretStr] = Field(default=None)
    incident_channel: str = Field(default="#data-incidents")
    low_severity_channel: str = Field(default="#data-quality-alerts")
    emoji_critical: str = Field(default=":rotating_light:")
    emoji_warning: str = Field(default=":warning:")
    emoji_resolved: str = Field(default=":white_check_mark:")


class ObservabilitySettings(BaseSettings):
    """Prometheus, OpenTelemetry and observability configuration."""

    model_config = SettingsConfigDict(env_prefix="OTEL_")

    service_name: str = Field(default="intellipipe-api")
    service_version: str = Field(default="1.0.0")
    environment: str = Field(default="production")
    exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    prometheus_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    enable_tracing: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)


class AnomalySettings(BaseSettings):
    """Anomaly detection model configuration."""

    model_config = SettingsConfigDict(env_prefix="ANOMALY_")

    isolation_forest_contamination: float = Field(default=0.05)
    isolation_forest_n_estimators: int = Field(default=100)
    autoencoder_latent_dim: int = Field(default=16)
    autoencoder_epochs: int = Field(default=50)
    autoencoder_batch_size: int = Field(default=256)
    zscore_threshold: float = Field(default=3.5)
    ensemble_weights_if: float = Field(default=0.4)
    ensemble_weights_ae: float = Field(default=0.4)
    ensemble_weights_zscore: float = Field(default=0.2)
    min_samples_for_training: int = Field(default=1000)
    model_retrain_interval_hours: int = Field(default=24)
    shadow_eval_enabled: bool = Field(default=True)


class DataQualitySettings(BaseSettings):
    """Data quality rule engine configuration."""

    model_config = SettingsConfigDict(env_prefix="DQ_")

    freshness_sla_minutes: int = Field(default=30)
    null_spike_threshold_pct: float = Field(default=0.05)
    duplicate_threshold_pct: float = Field(default=0.01)
    schema_drift_alert_threshold: int = Field(default=1)
    min_rows_per_batch: int = Field(default=10)
    ge_data_docs_path: str = Field(default="/tmp/intellipipe/ge_docs")


class APISettings(BaseSettings):
    """FastAPI application configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    workers: int = Field(default=4)
    reload: bool = Field(default=False)
    secret_key: SecretStr = Field(description="JWT signing secret key")
    access_token_expire_minutes: int = Field(default=60)
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    rate_limit_requests: int = Field(default=1000)
    rate_limit_window_seconds: int = Field(default=60)


class Settings(BaseSettings):
    """
    Root IntelliPipe configuration.

    All sub-configs are composed here. Load from .env or environment variables.
    Secrets are masked in logs via SecretStr.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application metadata
    app_name: str = Field(default="IntelliPipe")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    tenant_id: str = Field(default="default")

    # Sub-configurations
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    spark: SparkSettings = Field(default_factory=SparkSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    jira: JiraSettings = Field(default_factory=JiraSettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    anomaly: AnomalySettings = Field(default_factory=AnomalySettings)
    data_quality: DataQualitySettings = Field(default_factory=DataQualitySettings)
    api: APISettings = Field(default_factory=APISettings)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    def safe_dict(self) -> Dict[str, Any]:
        """Return config dict with secrets masked for logging."""
        data = self.model_dump()
        # Recursively mask secret values
        def mask_secrets(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: "***MASKED***" if "secret" in k.lower() or
                        "password" in k.lower() or "token" in k.lower() or
                        "key" in k.lower() else mask_secrets(v)
                        for k, v in obj.items()}
            return obj
        return mask_secrets(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton settings instance with LRU cache.
    Use this factory everywhere — never instantiate Settings directly.
    """
    return Settings()


# Convenience alias
settings = get_settings()
