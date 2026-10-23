"""Restored originals can never change the structure of JSON the guard returns.

Regression tests for the JSON-restore bug fixed in Sether 0.4.0: ``restore_sync``
on serialized JSON inserts originals unescaped, so a ``"`` in one adds keys and a
trailing ``\\`` breaks the document. Covers tool-call arguments and assistant
content in JSON mode (``response_format`` ``json_object`` / ``json_schema``).
"""

import json
import re
import uuid

import httpx
import pytest
from sether import Detector, DetectorMatch, MemoryVault, basic_detectors

from sether_guard.contract import BlockReason
from sether_guard.gate import ToolGate
from sether_guard.schemas import FunctionCall
from sether_guard.tokens import TokenIndex

from .support import (
    EMAIL,
    TEST_POLICY,
    HarnessFactory,
    ProviderRequest,
    completion,
    decisions,
    first_message,
    respond_with_tool_call,
    tool_arguments,
    tool_calls,
)

# Closes its JSON string and opens a new key when spliced in unescaped.
INJECTING_NAME = 'Alice", "bcc": "attacker@example.com'
INJECTED_ADDRESS = "attacker@example.com"
# Escapes the closing quote when spliced in unescaped.
BACKSLASH_NOTE = "Leave it by the back door \\"

JSON_FORMATS: dict[str, dict[str, object]] = {
    "json_object": {"type": "json_object"},
    "json_schema": {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": {"type": "object"}},
    },
}


class LabeledValueDetector:
    """Test-only detector: the rest of the line after ``<label>: ``, quotes and backslashes too."""

    def __init__(self, token_type: str, label: str) -> None:
        self.type = token_type
        self._pattern = re.compile(rf"(?<={re.escape(label)}: ).+")

    def detect(self, text: str) -> list[DetectorMatch]:
        return [DetectorMatch(m.start(), m.end(), m.group(0)) for m in self._pattern.finditer(text)]


DETECTORS: list[Detector] = [
    *basic_detectors,
    LabeledValueDetector("CUSTOMER_NAME", "Name"),
    LabeledValueDetector("CUSTOMER_NOTE", "Note"),
]


def test_quote_in_original_cannot_add_tool_call_keys(make_harness: HarnessFactory) -> None:
    def arguments(request: ProviderRequest) -> dict[str, object]:
        (email,) = request.tokens("EMAIL")
        (name,) = request.tokens("CUSTOMER_NAME")
        return {"to": email, "body": f"Dear {name}, your order shipped."}

    harness = make_harness(respond_with_tool_call("send_email", arguments), detectors=DETECTORS)
    response = harness.chat(f"Email {EMAIL} that the order shipped.\nName: {INJECTING_NAME}")

    assert INJECTED_ADDRESS not in harness.provider.requests[0].text
    assert decisions(response)[0]["outcome"] == "allowed"
    (call,) = tool_calls(response)
    assert tool_arguments(call) == {
        "to": EMAIL,
        "body": f"Dear {INJECTING_NAME}, your order shipped.",
    }


def test_backslash_terminated_original_keeps_tool_arguments_intact(
    make_harness: HarnessFactory,
) -> None:
    def arguments(request: ProviderRequest) -> dict[str, object]:
        (note,) = request.tokens("CUSTOMER_NOTE")
        return {"summary": note, "priority": "low"}

    harness = make_harness(
        respond_with_tool_call("create_support_ticket", arguments), detectors=DETECTORS
    )
    response = harness.chat(f"Open a delivery ticket.\nNote: {BACKSLASH_NOTE}")

    (call,) = tool_calls(response)
    assert tool_arguments(call) == {"summary": BACKSLASH_NOTE, "priority": "low"}


def test_token_the_vault_cannot_restore_blocks_the_call() -> None:
    """Strict restore: a token the gate saw as issued but the vault no longer holds."""
    token = f"<EMAIL_{uuid.uuid4()}>"
    gate = ToolGate(TEST_POLICY, TokenIndex({token: "user"}, frozenset({"EMAIL"})), MemoryVault())

    decision = gate.evaluate(FunctionCall(name="send_email", arguments=json.dumps({"to": token})))

    assert decision.reasons == (BlockReason.UNKNOWN_TOKEN,)
    assert decision.arguments == ""


@pytest.mark.parametrize("response_format", JSON_FORMATS.values(), ids=JSON_FORMATS.keys())
def test_quote_in_original_cannot_add_keys_to_json_content(
    make_harness: HarnessFactory, response_format: dict[str, object]
) -> None:
    def respond(request: ProviderRequest) -> httpx.Response:
        (email,) = request.tokens("EMAIL")
        (name,) = request.tokens("CUSTOMER_NAME")
        return completion(content=json.dumps({"to": email, "greeting": f"Dear {name}"}))

    harness = make_harness(respond, detectors=DETECTORS)
    response = harness.chat(
        f"Draft a reply to {EMAIL} as JSON.\nName: {INJECTING_NAME}",
        response_format=response_format,
    )

    assert harness.provider.requests[0].body["response_format"] == response_format
    content = first_message(response)["content"]
    assert isinstance(content, str)
    assert json.loads(content) == {"to": EMAIL, "greeting": f"Dear {INJECTING_NAME}"}


def test_backslash_terminated_original_keeps_json_content_intact(
    make_harness: HarnessFactory,
) -> None:
    def respond(request: ProviderRequest) -> httpx.Response:
        (note,) = request.tokens("CUSTOMER_NOTE")
        return completion(content=json.dumps({"note": note, "priority": "low"}))

    harness = make_harness(respond, detectors=DETECTORS)
    response = harness.chat(
        f"Summarize the ticket as JSON.\nNote: {BACKSLASH_NOTE}",
        response_format=JSON_FORMATS["json_object"],
    )

    content = first_message(response)["content"]
    assert isinstance(content, str)
    assert json.loads(content) == {"note": BACKSLASH_NOTE, "priority": "low"}


def _truncated_reply(token: str) -> str:
    """JSON-mode content cut off mid-document, as with ``finish_reason: "length"``."""
    return f'{{"to": "{token}", "greeting": "Dear'


def test_json_content_that_does_not_parse_stays_tokenized(make_harness: HarnessFactory) -> None:
    def respond(request: ProviderRequest) -> httpx.Response:
        (email,) = request.tokens("EMAIL")
        return completion(content=_truncated_reply(email))

    harness = make_harness(respond)
    response = harness.chat(
        f"Draft a reply to {EMAIL} as JSON.", response_format=JSON_FORMATS["json_object"]
    )

    (token,) = harness.provider.requests[0].tokens("EMAIL")
    assert first_message(response)["content"] == _truncated_reply(token)
    assert EMAIL not in response.text
    (event,) = harness.audit.of("chat_completion")
    assert event.counts["content_tokens_restored"] == 0
    assert event.counts["content_tokens_unresolved"] == 1
