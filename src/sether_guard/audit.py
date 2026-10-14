"""Structured audit log of guard decisions.

Events carry token *types* and counts, reason codes, and policy field names.
Never values, never tokens, never argument paths taken from model output, and
never run ids (only a one-way ``run_ref``).
"""

import logging
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

type AuditEventName = Literal[
    "chat_completion", "tool_call", "run_opened", "run_closed", "run_expired"
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: AuditEventName
    outcome: str
    timestamp: str = Field(default_factory=_utc_now)
    request_id: str | None = None
    run_ref: str | None = None
    model: str | None = None
    tool: str | None = None
    reasons: tuple[str, ...] = ()
    policy_fields: tuple[str, ...] = ()
    token_types: dict[str, int] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)


class AuditSink(Protocol):
    def emit(self, event: AuditEvent) -> None: ...


AUDIT_LOGGER = "sether_guard.audit"


class LoggingAuditSink:
    """One JSON object per line on the ``sether_guard.audit`` logger."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(AUDIT_LOGGER)

    def emit(self, event: AuditEvent) -> None:
        self._logger.info(event.model_dump_json(exclude_none=True))


def configure_audit_logging() -> None:
    """Send audit lines to stderr unless the host application already routes them.

    Without this, INFO records on an unconfigured logger are dropped by
    Python's last-resort handler, and the audit trail would silently vanish.
    """
    logger = logging.getLogger(AUDIT_LOGGER)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
