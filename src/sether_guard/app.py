"""FastAPI application: an OpenAI-compatible chat completions endpoint plus run lifecycle."""

import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated

import httpx
from fastapi import APIRouter, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sether import Detector

from .audit import AuditEvent, AuditSink, LoggingAuditSink, configure_audit_logging
from .detection import build_detectors, detector_types
from .errors import GuardError, StreamingNotSupportedError, UpstreamRejectedError
from .policy import GuardPolicy, load_policy
from .runs import RunRegistry
from .schemas import ChatCompletionRequest
from .service import ChatGuard
from .settings import GuardSettings
from .tracing import NoopTracer, Tracer
from .upstream import UpstreamClient
from .vault import RunVault, VaultFactory

Authorization = Annotated[str | None, Header()]


class RunOpened(BaseModel):
    run_id: str
    max_age_seconds: int


def create_app_from_env() -> FastAPI:
    """Entry point for ``uvicorn --factory sether_guard.app:create_app_from_env``."""
    settings = GuardSettings()
    configure_audit_logging()
    return create_app(settings, load_policy(settings.policy_path))


def create_app(
    settings: GuardSettings,
    policy: GuardPolicy,
    *,
    detectors: Sequence[Detector] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    audit: AuditSink | None = None,
    tracer: Tracer | None = None,
    vault_factory: VaultFactory | None = None,
) -> FastAPI:
    """Build the app. Keyword arguments are injection points for tests and embedding."""
    active_detectors = (
        tuple(detectors)
        if detectors is not None
        else build_detectors(settings.detector_packs, settings.phone_region)
    )
    policy.check_detector_coverage(detector_types(active_detectors))
    sink = audit or LoggingAuditSink()
    factory = vault_factory or partial(
        RunVault,
        max_tokens=settings.max_tokens_per_vault,
        ttl_ms=settings.run_max_age_seconds * 1000,
    )
    runs = RunRegistry(
        vault_factory=factory,
        max_runs=settings.max_runs,
        max_age_seconds=settings.run_max_age_seconds,
        on_expire=_report_expiry(sink),
    )
    http = httpx.AsyncClient(
        base_url=str(settings.upstream_base_url),
        timeout=settings.upstream_timeout_seconds,
        transport=transport,
    )
    guard = ChatGuard(
        policy=policy,
        detectors=active_detectors,
        upstream=UpstreamClient(http),
        audit=sink,
        tracer=tracer or NoopTracer(),
        vault_factory=factory,
        runs=runs,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            runs.close_all()
            await http.aclose()

    app = FastAPI(title="Sether gateway guard", lifespan=lifespan)
    app.include_router(_build_router(guard, settings.run_max_age_seconds))
    _install_error_handler(app)
    return app


def _build_router(guard: ChatGuard, run_max_age_seconds: int) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        authorization: Authorization = None,
        x_sether_run_id: Annotated[str | None, Header()] = None,
    ) -> JSONResponse:
        if body.stream:
            raise StreamingNotSupportedError
        run = guard.resolve_run(x_sether_run_id, owner=authorization or "")
        request_id = uuid.uuid4().hex
        payload = await guard.complete(
            body, authorization=authorization, run=run, request_id=request_id
        )
        return JSONResponse(payload, headers={"x-request-id": request_id})

    @router.post("/v1/runs", status_code=201)
    async def open_run(authorization: Authorization = None) -> RunOpened:
        run = guard.open_run(owner=authorization or "")
        return RunOpened(run_id=run.run_id, max_age_seconds=run_max_age_seconds)

    @router.delete("/v1/runs/{run_id}", status_code=204)
    async def close_run(run_id: str, authorization: Authorization = None) -> Response:
        guard.close_run(run_id, owner=authorization or "")
        return Response(status_code=204)

    @router.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return router


def _install_error_handler(app: FastAPI) -> None:
    @app.exception_handler(GuardError)
    async def on_guard_error(_: Request, error: GuardError) -> Response:
        if isinstance(error, UpstreamRejectedError):
            # Tokenized body, deliberately not restored: provider errors may echo input.
            return Response(error.body, error.status_code, media_type=error.content_type)
        body = {
            "error": {"message": error.message, "type": "sether_guard_error", "code": error.code}
        }
        return JSONResponse(body, status_code=error.status_code)


def _report_expiry(sink: AuditSink) -> Callable[[str], None]:
    def report(run_ref: str) -> None:
        sink.emit(AuditEvent(event="run_expired", outcome="closed_by_backstop", run_ref=run_ref))

    return report
