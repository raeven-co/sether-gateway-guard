"""Test doubles: a scripted mock provider, an in-memory audit sink, a recording tracer."""

import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from fastapi.testclient import TestClient
from sether import Detector

from sether_guard.audit import AuditEvent
from sether_guard.policy import GuardPolicy
from sether_guard.tracing import AttributeValue
from sether_guard.vault import RunVault

# Fictional values. Each one is detected by Sether's default (basic) pack.
EMAIL = "amara.okafor@example.com"
OTHER_EMAIL = "joris.devries@example.org"
SUPPORT_EMAIL = "support@example.com"
ATTACKER_EMAIL = "attacker@evil.example"
PHONE = "+31 6 12345678"
CARD = "4111 1111 1111 1111"
IBAN = "NL91ABNA0417164300"
SSN = "123-45-6789"
IP_ADDRESS = "192.0.2.10"
RAW_PII = (EMAIL, OTHER_EMAIL, SUPPORT_EMAIL, PHONE, CARD, IBAN, SSN, IP_ADDRESS)

UPSTREAM_BASE_URL = "http://upstream.test/v1"
CALLER_KEY = "Bearer test-key"
TOKEN_RE = re.compile(r"<([A-Z][A-Z0-9_]*)_[0-9a-f-]{36}>")

TEST_POLICY = GuardPolicy.model_validate(
    {
        "version": 1,
        "tools": {
            "send_email": {
                "restore_fields": ["to", "cc[]", "subject", "body"],
                "token_fields": {
                    "to": {"types": ["EMAIL"], "sources": ["user"]},
                    "cc[]": {"types": ["EMAIL"], "required": False},
                },
            },
            "lookup_order": {"restore_fields": ["order_id"]},
            "create_support_ticket": {"restore_fields": ["summary"]},
        },
    }
)


class ProviderRequest:
    """One request as the provider received it, raw bytes included."""

    def __init__(self, request: httpx.Request) -> None:
        self.text = request.content.decode()
        self.body = json.loads(self.text)
        self.authorization = request.headers.get("authorization")

    def tokens(self, token_type: str) -> list[str]:
        return [m.group(0) for m in TOKEN_RE.finditer(self.text) if m.group(1) == token_type]


type Responder = Callable[[ProviderRequest], httpx.Response]


class MockProvider:
    """Stands in for the model provider: records each request, answers from a script."""

    def __init__(self, respond: Responder) -> None:
        self.requests: list[ProviderRequest] = []
        self._respond = respond

    def handle(self, request: httpx.Request) -> httpx.Response:
        recorded = ProviderRequest(request)
        self.requests.append(recorded)
        return self._respond(recorded)


def completion(
    *, content: str | None = None, tool_calls: list[dict[str, object]] | None = None
) -> httpx.Response:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_791_000_000,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
    }
    return httpx.Response(200, json=body)


def tool_call(
    name: str, arguments: Mapping[str, object] | str, call_id: str = "call_1"
) -> dict[str, object]:
    raw = arguments if isinstance(arguments, str) else json.dumps(dict(arguments))
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


def respond_with_tool_call(
    name: str, build_arguments: Callable[[ProviderRequest], Mapping[str, object] | str]
) -> Responder:
    return lambda request: completion(tool_calls=[tool_call(name, build_arguments(request))])


class MemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        self.events.append(event)

    def of(self, name: str) -> list[AuditEvent]:
        return [event for event in self.events if event.event == name]

    def as_text(self) -> str:
        return "\n".join(event.model_dump_json() for event in self.events)


class RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, AttributeValue] = {}

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self.attributes[key] = value


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []

    @contextmanager
    def start_span(self, name: str) -> Iterator[RecordingSpan]:
        span = RecordingSpan(name)
        self.spans.append(span)
        yield span


@dataclass
class GuardHarness:
    client: TestClient
    provider: MockProvider
    audit: MemoryAuditSink
    tracer: RecordingTracer
    vaults: list[RunVault] = field(default_factory=list)

    def chat(
        self,
        user: str | None = None,
        *,
        messages: Sequence[Mapping[str, object]] | None = None,
        run_id: str | None = None,
        authorization: str = CALLER_KEY,
        stream: bool = False,
    ) -> httpx.Response:
        payload = {
            "model": "mock-model",
            "messages": [dict(m) for m in messages or [{"role": "user", "content": user}]],
            "stream": stream,
        }
        headers = {"authorization": authorization}
        if run_id is not None:
            headers["x-sether-run-id"] = run_id
        response: httpx.Response = self.client.post(
            "/v1/chat/completions", json=payload, headers=headers
        )
        return response

    def open_run(self, authorization: str = CALLER_KEY) -> str:
        response = self.client.post("/v1/runs", headers={"authorization": authorization})
        assert response.status_code == 201
        run_id: str = response.json()["run_id"]
        return run_id

    def close_run(self, run_id: str, authorization: str = CALLER_KEY) -> httpx.Response:
        response: httpx.Response = self.client.delete(
            f"/v1/runs/{run_id}", headers={"authorization": authorization}
        )
        return response


class HarnessFactory(Protocol):
    def __call__(
        self,
        respond: Responder,
        *,
        policy: GuardPolicy = ...,
        detectors: Sequence[Detector] | None = ...,
        max_runs: int = ...,
    ) -> GuardHarness: ...


def first_message(response: httpx.Response) -> dict[str, object]:
    message: dict[str, object] = response.json()["choices"][0]["message"]
    return message


def tool_calls(response: httpx.Response) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = response.json()["choices"][0]["message"]["tool_calls"]
    return calls


def tool_arguments(call: dict[str, object]) -> object:
    function = call["function"]
    assert isinstance(function, dict)
    return json.loads(function["arguments"])


def decisions(response: httpx.Response) -> list[dict[str, object]]:
    result: list[dict[str, object]] = response.json()["sether_guard"]["decisions"]
    return result
