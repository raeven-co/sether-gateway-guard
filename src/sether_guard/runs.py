"""Explicitly scoped multi-turn runs.

A run keeps one vault alive across several chat completions so a tool call in
turn N can use a token issued in turn 1. Teardown is an API call
(``DELETE /v1/runs/{id}``), not a TTL. ``max_age_seconds`` only reclaims runs
a caller forgot to close, and each such expiry is reported.

The run id is a bearer capability: anyone holding it could ask the model to
echo a token and get the plaintext back. So ids are server-generated with 192
bits of entropy, bound to the caller's credential, and never logged.
"""

import hashlib
import hmac
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from .errors import RunLimitError, RunNotFoundError
from .vault import RunVault, VaultFactory


def run_ref(run_id: str) -> str:
    """Non-reversible reference for logs and traces."""
    return hashlib.sha256(run_id.encode()).hexdigest()[:16]


def _digest(owner: str) -> str:
    return hashlib.sha256(owner.encode()).hexdigest()


@dataclass
class Run:
    run_id: str
    vault: RunVault
    owner_digest: str
    created_at: float

    @property
    def ref(self) -> str:
        return run_ref(self.run_id)


class RunRegistry:
    def __init__(
        self,
        *,
        vault_factory: VaultFactory,
        max_runs: int,
        max_age_seconds: float,
        on_expire: Callable[[str], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._vault_factory = vault_factory
        self._max_runs = max_runs
        self._max_age = max_age_seconds
        self._on_expire = on_expire
        self._clock = clock
        self._runs: dict[str, Run] = {}

    def open(self, owner: str) -> Run:
        self._sweep_expired()
        if len(self._runs) >= self._max_runs:
            raise RunLimitError
        run_id = f"run_{secrets.token_urlsafe(24)}"
        run = Run(run_id, self._vault_factory(), _digest(owner), self._clock())
        self._runs[run_id] = run
        return run

    def get(self, run_id: str, owner: str) -> Run:
        """Same error for unknown, expired and foreign runs: no existence oracle."""
        self._sweep_expired()
        run = self._runs.get(run_id)
        if run is None or not hmac.compare_digest(run.owner_digest, _digest(owner)):
            raise RunNotFoundError
        return run

    def close(self, run_id: str, owner: str) -> Run:
        run = self.get(run_id, owner)
        del self._runs[run_id]
        run.vault.clear()
        return run

    def close_all(self) -> None:
        for run in self._runs.values():
            run.vault.clear()
        self._runs.clear()

    def _sweep_expired(self) -> None:
        now = self._clock()
        expired = [run for run in self._runs.values() if now - run.created_at > self._max_age]
        for run in expired:
            del self._runs[run.run_id]
            run.vault.clear()
            self._on_expire(run.ref)
