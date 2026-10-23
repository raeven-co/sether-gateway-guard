"""Tool-call gate: decide whether restored PII may flow into a tool call.

A call is allowed only if (1) the tool is in the policy, (2) its arguments are
one strict JSON object, (3) every token-shaped string anywhere in it (keys
included) is byte-identical to a token issued in this run, and (4) every
policy ``token_fields`` entry holds exactly one issued token of an allowed
type, issued from an allowed message role. Only then are ``restore_fields``
restored, with Sether's strict restore, and the arguments are re-serialized, so
the tool receives exactly the JSON the gate inspected.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from sether import UnresolvedTokenError, Vault, restore_sync

from .contract import BlockReason
from .jsonwalk import (
    JsonObject,
    StringSite,
    dumps_json,
    iter_string_sites,
    map_string_values,
    parse_field_path,
    parse_json_object,
)
from .policy import GuardPolicy, TokenFieldRule, ToolPolicy
from .schemas import FunctionCall
from .tokens import TokenFinding, TokenIndex, TokenStatus, token_type_of

_TOKEN_STATUS_REASONS = {
    TokenStatus.ALTERED: BlockReason.ALTERED_TOKEN,
    TokenStatus.UNKNOWN: BlockReason.UNKNOWN_TOKEN,
}


@dataclass(frozen=True)
class GateDecision:
    tool_name: str
    reasons: tuple[BlockReason, ...]
    arguments: str
    """Restored, re-serialized arguments when allowed; empty when blocked."""

    token_types: dict[str, int] = field(default_factory=dict)
    fields_restored: int = 0
    policy_fields: tuple[str, ...] = ()
    """Policy field names involved in a block. Taken from the policy, never from model output."""

    @property
    def allowed(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class _Failure:
    reason: BlockReason
    policy_field: str | None = None


class ToolGate:
    def __init__(self, policy: GuardPolicy, tokens: TokenIndex, vault: Vault) -> None:
        self._policy = policy
        self._tokens = tokens
        self._vault = vault

    def evaluate(self, call: FunctionCall) -> GateDecision:
        tool_policy = self._policy.tools.get(call.name)
        if tool_policy is None:
            return _blocked(call.name, [_Failure(BlockReason.TOOL_NOT_ALLOWED)])
        arguments = parse_json_object(call.arguments)
        if arguments is None:
            return _blocked(call.name, [_Failure(BlockReason.INVALID_ARGUMENTS)])

        sites = list(iter_string_sites(arguments))
        findings = [finding for site in sites for finding in self._tokens.find(site.value)]
        token_types = dict(Counter(finding.token_type for finding in findings))
        failures = [*_token_failures(findings), *self._field_failures(tool_policy, sites)]
        if failures:
            return _blocked(call.name, failures, token_types)
        return self._allow(call.name, arguments, sites, tool_policy, token_types)

    def _field_failures(
        self, tool_policy: ToolPolicy, sites: Sequence[StringSite]
    ) -> list[_Failure]:
        failures: list[_Failure] = []
        for spec, rule in tool_policy.token_fields.items():
            path = parse_field_path(spec)
            values = [s.value for s in sites if not s.is_key and s.path == path and s.value.strip()]
            failures.extend(self._check_token_field(spec, rule, values))
        return failures

    def _check_token_field(
        self, spec: str, rule: TokenFieldRule, values: Sequence[str]
    ) -> list[_Failure]:
        if not values:
            return [_Failure(BlockReason.MISSING_REQUIRED_TOKEN, spec)] if rule.required else []
        failures: list[_Failure] = []
        for value in values:
            if not self._tokens.is_issued(value):
                failures.append(_Failure(BlockReason.VALUE_NOT_ISSUED_TOKEN, spec))
            elif token_type_of(value) not in rule.types:
                failures.append(_Failure(BlockReason.TOKEN_TYPE_NOT_ALLOWED, spec))
            elif rule.sources is not None and self._tokens.source_of(value) not in rule.sources:
                failures.append(_Failure(BlockReason.TOKEN_SOURCE_NOT_ALLOWED, spec))
        return failures

    def _allow(
        self,
        name: str,
        arguments: JsonObject,
        sites: Sequence[StringSite],
        tool_policy: ToolPolicy,
        token_types: dict[str, int],
    ) -> GateDecision:
        paths = tool_policy.restore_paths()
        try:
            restored = map_string_values(
                arguments, lambda path, value: self._restore(value) if path in paths else value
            )
        except UnresolvedTokenError:
            # Tokens were checked against a snapshot of the vault; one has expired since.
            return _blocked(name, [_Failure(BlockReason.UNKNOWN_TOKEN)], token_types)
        fields_restored = sum(
            1 for s in sites if not s.is_key and s.path in paths and self._tokens.find(s.value)
        )
        return GateDecision(
            tool_name=name,
            reasons=(),
            arguments=dumps_json(restored),
            token_types=token_types,
            fields_restored=fields_restored,
        )

    def _restore(self, value: str) -> str:
        """Restore one decoded string value; ``dumps_json`` then escapes the originals.

        Sether's ``restore_json_value`` does the same for every value, but only policy
        fields may be restored here. Strict, because the gate already requires every
        token to be issued: one left unresolved must fail the call, not reach the tool.
        """
        return restore_sync(value, self._vault, strict=True)


def review_arguments(decision: GateDecision) -> str:
    """Arguments of the ``needs_human_review`` marker that replaces a blocked call."""
    return dumps_json(
        {
            "blocked_tool": decision.tool_name,
            "reasons": [reason.value for reason in decision.reasons],
            "policy_fields": list(decision.policy_fields),
        }
    )


def _token_failures(findings: Iterable[TokenFinding]) -> list[_Failure]:
    return [
        _Failure(_TOKEN_STATUS_REASONS[finding.status])
        for finding in findings
        if finding.status is not TokenStatus.ISSUED
    ]


def _blocked(
    name: str, failures: Sequence[_Failure], token_types: dict[str, int] | None = None
) -> GateDecision:
    return GateDecision(
        tool_name=name,
        reasons=tuple(sorted({failure.reason for failure in failures})),
        arguments="",
        token_types=token_types or {},
        policy_fields=tuple(sorted({f.policy_field for f in failures if f.policy_field})),
    )
