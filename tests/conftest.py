from collections.abc import Iterator, Sequence
from contextlib import ExitStack

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import AnyHttpUrl
from sether import Detector

from sether_guard import GuardPolicy, GuardSettings, create_app
from sether_guard.vault import RunVault

from .support import (
    TEST_POLICY,
    UPSTREAM_BASE_URL,
    GuardHarness,
    HarnessFactory,
    MemoryAuditSink,
    MockProvider,
    RecordingTracer,
    Responder,
)


@pytest.fixture
def make_harness() -> Iterator[HarnessFactory]:
    """Build a guard app wired to a mock provider, with spies on audit, tracing and vaults."""
    with ExitStack() as stack:

        def factory(
            respond: Responder,
            *,
            policy: GuardPolicy = TEST_POLICY,
            detectors: Sequence[Detector] | None = None,
            max_runs: int = 10,
        ) -> GuardHarness:
            vaults: list[RunVault] = []

            def vault_factory() -> RunVault:
                vault = RunVault(max_tokens=100, ttl_ms=60_000)
                vaults.append(vault)
                return vault

            provider = MockProvider(respond)
            audit = MemoryAuditSink()
            tracer = RecordingTracer()
            settings = GuardSettings(
                upstream_base_url=AnyHttpUrl(UPSTREAM_BASE_URL), max_runs=max_runs
            )
            app = create_app(
                settings,
                policy,
                detectors=detectors,
                transport=httpx.MockTransport(provider.handle),
                audit=audit,
                tracer=tracer,
                vault_factory=vault_factory,
            )
            client = stack.enter_context(TestClient(app))
            return GuardHarness(client, provider, audit, tracer, vaults)

        yield factory
