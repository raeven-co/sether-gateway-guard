"""Outbound tokenization, inbound restoration, vault lifecycle, failure modes."""

import json

import httpx

from .support import (
    ATTACKER_EMAIL,
    CALLER_KEY,
    CARD,
    EMAIL,
    IBAN,
    IP_ADDRESS,
    PHONE,
    RAW_PII,
    SSN,
    SUPPORT_EMAIL,
    TOKEN_RE,
    HarnessFactory,
    ProviderRequest,
    completion,
    first_message,
    tool_call,
)


def test_assistant_text_round_trip(make_harness: HarnessFactory) -> None:
    def respond(request: ProviderRequest) -> httpx.Response:
        (token,) = request.tokens("EMAIL")
        return completion(content=f"Sure. I will send the update to {token}.")

    harness = make_harness(respond)
    response = harness.chat(f"Send the order update to {EMAIL}.")

    assert response.status_code == 200
    assert first_message(response)["content"] == f"Sure. I will send the update to {EMAIL}."
    assert response.json()["sether_guard"] == {"tool_calls_blocked": 0, "decisions": []}
    assert harness.provider.requests[0].authorization == CALLER_KEY


def test_provider_never_receives_raw_pii(make_harness: HarnessFactory) -> None:
    harness = make_harness(lambda _: completion(content="ok"))
    restored_history = json.dumps({"to": EMAIL, "subject": "Refund", "body": f"Card {CARD}."})
    messages: list[dict[str, object]] = [
        {"role": "system", "content": f"Escalations go to {SUPPORT_EMAIL}."},
        {"role": "user", "content": f"I am {EMAIL}, phone {PHONE}, IBAN {IBAN}."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("send_email", restored_history)],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": f"Sent to {EMAIL} from {IP_ADDRESS}.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"My SSN is {SSN}."},
                {"type": "image_url", "image_url": {"url": "https://example.com/receipt.png"}},
            ],
        },
    ]

    response = harness.chat(messages=messages)

    assert response.status_code == 200
    (sent,) = harness.provider.requests
    leaked = [value for value in RAW_PII if value in sent.text]
    assert leaked == []
    issued_types = {match.group(1) for match in TOKEN_RE.finditer(sent.text)}
    assert issued_types == {"EMAIL", "PHONE", "IBAN", "CC", "IPV4", "SSN"}
    history_arguments = json.loads(
        sent.body["messages"][2]["tool_calls"][0]["function"]["arguments"]
    )
    assert TOKEN_RE.fullmatch(history_arguments["to"])
    assert sent.body["messages"][3]["tool_call_id"] == "call_1"
    assert sent.body["messages"][4]["content"][1]["type"] == "image_url"


def test_per_request_vault_holds_values_only_during_the_call(make_harness: HarnessFactory) -> None:
    sizes_during_call: list[int] = []

    def respond(_: ProviderRequest) -> httpx.Response:
        sizes_during_call.append(harness.vaults[0].size())
        return completion(content="ok")

    harness = make_harness(respond)
    harness.chat(f"Reach me at {EMAIL} or {PHONE}.")

    (vault,) = harness.vaults
    assert sizes_during_call == [2]
    assert vault.size() == 0
    assert vault.issued_sources() == {}


def test_vault_is_cleared_when_upstream_is_unreachable(make_harness: HarnessFactory) -> None:
    def respond(_: ProviderRequest) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    harness = make_harness(respond)
    response = harness.chat(f"Email {EMAIL}.")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_unavailable"
    assert harness.vaults[0].size() == 0
    assert harness.audit.of("chat_completion")[0].outcome == "upstream_unavailable"


def test_streaming_is_rejected_not_faked(make_harness: HarnessFactory) -> None:
    harness = make_harness(lambda _: completion(content="unused"))
    response = harness.chat(f"Email {EMAIL}.", stream=True)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "streaming_not_supported"
    assert harness.provider.requests == []


def test_upstream_error_body_passes_through_still_tokenized(make_harness: HarnessFactory) -> None:
    def respond(request: ProviderRequest) -> httpx.Response:
        (token,) = request.tokens("EMAIL")
        return httpx.Response(400, json={"error": {"message": f"Rejected input {token}"}})

    harness = make_harness(respond)
    response = harness.chat(f"Email {EMAIL}.")

    assert response.status_code == 400
    assert EMAIL not in response.text
    assert TOKEN_RE.search(response.text)


def test_unparseable_upstream_completion_fails_closed(make_harness: HarnessFactory) -> None:
    harness = make_harness(lambda _: httpx.Response(200, text="<html>gateway timeout</html>"))
    response = harness.chat(f"Email {EMAIL}.")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_contract_violation"


def test_legacy_function_call_fails_closed(make_harness: HarnessFactory) -> None:
    def respond(_: ProviderRequest) -> httpx.Response:
        body = completion(content=None).json()
        body["choices"][0]["message"]["function_call"] = {"name": "send_email", "arguments": "{}"}
        return httpx.Response(200, json=body)

    harness = make_harness(respond)
    response = harness.chat(f"Email {EMAIL}.")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_contract_violation"


def test_audit_and_trace_carry_types_and_counts_but_no_pii(make_harness: HarnessFactory) -> None:
    def respond(request: ProviderRequest) -> httpx.Response:
        (token,) = request.tokens("EMAIL")
        return completion(
            tool_calls=[
                tool_call("send_email", {"to": token, "body": "Shipped."}, call_id="call_ok"),
                tool_call("send_email", {"to": ATTACKER_EMAIL, "body": "x"}, call_id="call_bad"),
            ]
        )

    harness = make_harness(respond)
    harness.chat(f"Email {EMAIL}, or call {PHONE}.")

    audit_text = harness.audit.as_text()
    (span,) = harness.tracer.spans
    span_text = json.dumps(span.attributes)
    for text in (audit_text, span_text):
        assert [value for value in (*RAW_PII, ATTACKER_EMAIL) if value in text] == []
        assert TOKEN_RE.search(text) is None

    allowed, blocked = harness.audit.of("tool_call")
    assert (allowed.outcome, allowed.token_types, allowed.counts) == (
        "allowed",
        {"EMAIL": 1},
        {"fields_restored": 1},
    )
    assert (blocked.outcome, blocked.reasons, blocked.policy_fields) == (
        "blocked",
        ("value_not_issued_token",),
        ("to",),
    )
    (request_event,) = harness.audit.of("chat_completion")
    assert request_event.token_types == {"EMAIL": 1, "PHONE": 1}
    assert request_event.counts["tool_calls_blocked"] == 1
    assert span.attributes["sether_guard.tool_calls_blocked"] == 1
    assert span.attributes["gen_ai.usage.input_tokens"] == 42
