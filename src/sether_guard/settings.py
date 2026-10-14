"""Runtime configuration, read from ``SETHER_GUARD_*`` environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type DetectorPack = Literal["basic", "secrets", "identity"]


class GuardSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SETHER_GUARD_", frozen=True)

    upstream_base_url: AnyHttpUrl = AnyHttpUrl("https://api.openai.com/v1")
    upstream_timeout_seconds: float = Field(default=60.0, gt=0)
    policy_path: Path = Path("policy.example.yaml")

    detector_packs: tuple[DetectorPack, ...] = Field(default=("basic",), min_length=1)
    """Sether packs to run. ``secrets`` and ``identity`` are opt-in, as in Sether itself."""

    phone_region: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    """ISO country code. Without it, Sether only detects international-format numbers."""

    max_tokens_per_vault: int = Field(default=10_000, gt=0)
    max_runs: int = Field(default=1_000, gt=0)

    run_max_age_seconds: int = Field(default=3_600, gt=0)
    """Backstop for runs a caller never closed. Teardown is ``DELETE /v1/runs/{id}``."""
