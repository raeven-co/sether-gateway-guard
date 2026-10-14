"""The request pipeline: tokenize, forward, restore, gate, report."""

from collections.abc import Sequence
from dataclasses import dataclass

from sether import Detector, Sether

from .audit import AuditEvent, AuditSink
from .detection import detector_types
from .errors import GuardError
from .gate import ToolGate
from .inbound import GuardedCompletion, InboundGuard, ToolCallOutcome, render
from .outbound import Redact, RedactFor, tokenize_request
from .policy import GuardPolicy
from .runs import Run, RunRegistry
from .schemas import ChatCompletionRequest
from .tokens import TokenIndex
from .tracing import Span, Tracer
from .upstream import UpstreamClient
from .vault import RunVault, VaultFactory

UNLISTED_TOOL = "<unlisted>"


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    run_ref: str | None
    model: str


class ChatGuard:
    def __init__(
        self,
        *,
        policy: GuardPolicy,
        detectors: Sequence[Detector],
        upstream: UpstreamClient,
        audit: AuditSink,
        tracer: Tracer,
        vault_factory: VaultFactory,
        runs: RunRegistry,
    ) -> None:
        self._policy = policy
        self._detectors = detectors
        self._known_types = detector_types(detectors)
        self._upstream = upstream
        self._audit = audit
        self._tracer = tracer
        self._vault_factory = vault_factory
        self._runs = runs

    # --- runs ----------------------------------------------------------------

    def open_run(self, owner: str) -> Run:
        run = self._runs.open(owner)
        self._audit.emit(AuditEvent(event="run_opened", outcome="ok", run_ref=run.ref))
        return run

    def close_run(self, run_id: str, owner: str) -> None:
        run = self._runs.close(run_id, owner)
        self._audit.emit(AuditEvent(event="run_closed", outcome="ok", run_ref=run.ref))

    def resolve_run(self, run_id: str | None, owner: str) -> Run | None:
        return None if run_id is None else self._runs.get(run_id, owner)

    # --- chat completions -------------------------------------------------------

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        authorization: str | None,
        run: Run | None,
        request_id: str,
    ) -> dict[str, object]:
        context = RequestContext(request_id, run.ref if run else None, request.model)
        vault = run.vault if run is not None else self._vault_factory()
        with self._tracer.start_span("sether_guard.chat_completion") as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", request.model)
            span.set_attribute("sether_guard.run_scoped", run is not None)
            try:
                guarded, issued = await self._complete(request, authorization, vault)
            except GuardError as error:
                span.set_attribute("sether_guard.outcome", error.code)
                self._audit.emit(_request_event(context, error.code))
                raise
            finally:
                if run is None:
                    vault.clear()
            self._report(context, span, guarded, issued)
        return render(guarded)

    async def _complete(
        self, request: ChatCompletionRequest, authorization: str | None, vault: RunVault
    ) -> tuple[GuardedCompletion, dict[str, int]]:
        sether = Sether(detectors=self._detectors, vault=vault)
        before = vault.issued_type_counts()
        outbound = tokenize_request(request, _redact_for(sether, vault))
        issued = _counts_since(before, vault.issued_type_counts())
        completion = await self._upstream.create_chat_completion(outbound, authorization)
        tokens = TokenIndex(vault.issued_sources(), self._known_types)
        gate = ToolGate(self._policy, tokens, sether.restore_sync)
        return InboundGuard(gate, tokens, sether.restore_sync).guard(completion), issued

    def _report(
        self,
        context: RequestContext,
        span: Span,
        guarded: GuardedCompletion,
        issued: dict[str, int],
    ) -> None:
        for tool_outcome in guarded.outcomes:
            self._audit.emit(self._tool_event(context, tool_outcome))
        counts = {
            "tokens_issued": sum(issued.values()),
            "content_tokens_restored": guarded.content_tokens_restored,
            "content_tokens_unresolved": guarded.content_tokens_unresolved,
            "tool_calls": len(guarded.outcomes),
            "tool_calls_blocked": guarded.blocked_count,
        }
        outcome = "tool_calls_blocked" if guarded.blocked_count else "ok"
        self._audit.emit(_request_event(context, outcome, issued, counts))
        span.set_attribute("sether_guard.outcome", outcome)
        span.set_attribute("gen_ai.response.model", guarded.completion.model)
        for key, value in counts.items():
            span.set_attribute(f"sether_guard.{key}", value)
        _annotate_usage(span, guarded)

    def _tool_event(self, context: RequestContext, outcome: ToolCallOutcome) -> AuditEvent:
        decision = outcome.decision
        listed = decision.tool_name in self._policy.tools
        return AuditEvent(
            event="tool_call",
            outcome="allowed" if decision.allowed else "blocked",
            request_id=context.request_id,
            run_ref=context.run_ref,
            model=context.model,
            tool=decision.tool_name if listed else UNLISTED_TOOL,
            reasons=tuple(reason.value for reason in decision.reasons),
            policy_fields=decision.policy_fields,
            token_types=decision.token_types,
            counts={"fields_restored": decision.fields_restored},
        )


def _request_event(
    context: RequestContext,
    outcome: str,
    token_types: dict[str, int] | None = None,
    counts: dict[str, int] | None = None,
) -> AuditEvent:
    return AuditEvent(
        event="chat_completion",
        outcome=outcome,
        request_id=context.request_id,
        run_ref=context.run_ref,
        model=context.model,
        token_types=token_types or {},
        counts=counts or {},
    )


def _redact_for(sether: Sether, vault: RunVault) -> RedactFor:
    def bind(role: str) -> Redact:
        def redact(text: str) -> str:
            with vault.issuing_for(role):
                return sether.redact_sync(text)

        return redact

    return bind


def _annotate_usage(span: Span, guarded: GuardedCompletion) -> None:
    usage = guarded.completion.usage
    if usage is None:
        return
    if usage.prompt_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
    if usage.completion_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)


def _counts_since(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    delta = {key: count - before.get(key, 0) for key, count in after.items()}
    return {key: count for key, count in delta.items() if count > 0}
