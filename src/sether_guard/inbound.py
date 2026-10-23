"""Inbound: restore assistant text and run every tool call through the gate."""

from dataclasses import dataclass

from sether import Vault, restore_json, restore_sync

from .contract import REVIEW_TOOL_NAME
from .errors import UpstreamContractError
from .gate import GateDecision, ToolGate, review_arguments
from .schemas import ChatCompletion, Choice, FunctionCall, ToolCall
from .tokens import TokenIndex, TokenStatus


@dataclass(frozen=True)
class ToolCallOutcome:
    tool_call_id: str
    decision: GateDecision


@dataclass(frozen=True)
class GuardedCompletion:
    completion: ChatCompletion
    outcomes: tuple[ToolCallOutcome, ...]
    content_tokens_restored: int
    content_tokens_unresolved: int

    @property
    def blocked_count(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.decision.allowed)


@dataclass(frozen=True)
class _GuardedChoice:
    choice: Choice
    outcomes: tuple[ToolCallOutcome, ...]
    restored: int
    unresolved: int


class InboundGuard:
    def __init__(
        self, gate: ToolGate, tokens: TokenIndex, vault: Vault, *, json_content: bool
    ) -> None:
        self._gate = gate
        self._tokens = tokens
        self._vault = vault
        self._json_content = json_content

    def guard(self, completion: ChatCompletion) -> GuardedCompletion:
        guarded = [self._guard_choice(choice) for choice in completion.choices]
        return GuardedCompletion(
            completion=completion.model_copy(update={"choices": [g.choice for g in guarded]}),
            outcomes=tuple(outcome for g in guarded for outcome in g.outcomes),
            content_tokens_restored=sum(g.restored for g in guarded),
            content_tokens_unresolved=sum(g.unresolved for g in guarded),
        )

    def _guard_choice(self, choice: Choice) -> _GuardedChoice:
        message = choice.message
        if message.function_call is not None:
            raise UpstreamContractError("Legacy function_call responses are not supported.")
        update: dict[str, object] = {}
        restored = unresolved = 0
        if message.content is not None:
            findings = self._tokens.find(message.content)
            content = self._restore_content(message.content)
            if content is not None:
                restored = sum(1 for f in findings if f.status is TokenStatus.ISSUED)
                update["content"] = content
            unresolved = len(findings) - restored
        outcomes: list[ToolCallOutcome] = []
        if message.tool_calls is not None:
            outcomes = [
                ToolCallOutcome(c.id, self._gate.evaluate(c.function)) for c in message.tool_calls
            ]
            update["tool_calls"] = [
                _apply(call, outcome.decision)
                for call, outcome in zip(message.tool_calls, outcomes, strict=True)
            ]
        guarded_choice = choice.model_copy(update={"message": message.model_copy(update=update)})
        return _GuardedChoice(guarded_choice, tuple(outcomes), restored, unresolved)

    def _restore_content(self, content: str) -> str | None:
        """Restored content, or None when it must stay tokenized.

        In JSON mode the content is a document, so it is restored value by value and an
        original cannot add keys or break it. Content that does not parse (truncated
        output, for example) stays tokenized rather than being restored as text.
        """
        if not self._json_content:
            return restore_sync(content, self._vault)
        try:
            return restore_json(content, self._vault)
        except (ValueError, RecursionError):
            return None


def _apply(call: ToolCall, decision: GateDecision) -> ToolCall:
    if decision.allowed:
        function = call.function.model_copy(update={"arguments": decision.arguments})
        return call.model_copy(update={"function": function})
    marker = FunctionCall(name=REVIEW_TOOL_NAME, arguments=review_arguments(decision))
    return ToolCall(id=call.id, type="function", function=marker)


def render(guarded: GuardedCompletion) -> dict[str, object]:
    """The OpenAI-shaped response plus a ``sether_guard`` extension object."""
    payload: dict[str, object] = guarded.completion.model_dump(mode="json", exclude_unset=True)
    payload["sether_guard"] = {
        "tool_calls_blocked": guarded.blocked_count,
        "decisions": [_render_outcome(outcome) for outcome in guarded.outcomes],
    }
    return payload


def _render_outcome(outcome: ToolCallOutcome) -> dict[str, object]:
    decision = outcome.decision
    return {
        "tool_call_id": outcome.tool_call_id,
        "tool": decision.tool_name,
        "outcome": "allowed" if decision.allowed else "blocked",
        "reasons": [reason.value for reason in decision.reasons],
        "policy_fields": list(decision.policy_fields),
        "replaced_with": None if decision.allowed else REVIEW_TOOL_NAME,
    }
