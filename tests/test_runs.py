"""Run scoping: tokens live as long as the run, bound to the caller, torn down explicitly."""

import httpx
import pytest

from sether_guard.errors import RunNotFoundError
from sether_guard.runs import RunRegistry
from sether_guard.vault import RunVault

from .support import (
    EMAIL,
    OTHER_EMAIL,
    HarnessFactory,
    ProviderRequest,
    completion,
    decisions,
    tool_arguments,
    tool_call,
    tool_calls,
)


def test_run_scope_spans_turns_until_explicit_teardown(make_harness: HarnessFactory) -> None:
    first_turn_tokens: list[str] = []

    def respond(request: ProviderRequest) -> httpx.Response:
        if len(harness.provider.requests) == 1:
            first_turn_tokens.extend(request.tokens("EMAIL"))
            return completion(content="Noted.")
        arguments = {"to": first_turn_tokens[0], "body": "Your order shipped."}
        return completion(tool_calls=[tool_call("send_email", arguments)])

    harness = make_harness(respond)
    run_id = harness.open_run()
    first_user_turn = {"role": "user", "content": f"My email is {EMAIL}."}
    harness.chat(messages=[first_user_turn], run_id=run_id)
    second = harness.chat(
        messages=[
            first_user_turn,
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Send me the shipping update."},
        ],
        run_id=run_id,
    )

    (call,) = tool_calls(second)
    assert tool_arguments(call) == {"to": EMAIL, "body": "Your order shipped."}
    (vault,) = harness.vaults
    assert vault.size() == 2  # the history was re-tokenized in turn 2

    assert harness.close_run(run_id).status_code == 204
    assert vault.size() == 0
    after_teardown = harness.chat("Anything else?", run_id=run_id)
    assert after_teardown.status_code == 404
    assert run_id not in harness.audit.as_text()


def test_token_from_another_run_is_unknown(make_harness: HarnessFactory) -> None:
    run_a_tokens: list[str] = []

    def respond(request: ProviderRequest) -> httpx.Response:
        if not run_a_tokens:
            run_a_tokens.extend(request.tokens("EMAIL"))
            return completion(content="ok")
        arguments = {"to": run_a_tokens[0], "body": "Hi"}
        return completion(tool_calls=[tool_call("send_email", arguments)])

    harness = make_harness(respond)
    run_a, run_b = harness.open_run(), harness.open_run()
    harness.chat(f"I am {EMAIL}.", run_id=run_a)
    response = harness.chat(f"I am {OTHER_EMAIL}.", run_id=run_b)

    assert decisions(response)[0]["reasons"] == ["unknown_token", "value_not_issued_token"]
    assert EMAIL not in response.text


def test_unscoped_requests_never_share_tokens(make_harness: HarnessFactory) -> None:
    earlier_tokens: list[str] = []

    def respond(request: ProviderRequest) -> httpx.Response:
        if not earlier_tokens:
            earlier_tokens.extend(request.tokens("EMAIL"))
            return completion(content="ok")
        arguments = {"to": earlier_tokens[0], "body": "Hi"}
        return completion(tool_calls=[tool_call("send_email", arguments)])

    harness = make_harness(respond)
    harness.chat(f"I am {EMAIL}.")
    response = harness.chat(f"I am {EMAIL}.")

    assert decisions(response)[0]["reasons"] == ["unknown_token", "value_not_issued_token"]
    assert [vault.size() for vault in harness.vaults] == [0, 0]


def test_run_is_bound_to_the_caller_credential(make_harness: HarnessFactory) -> None:
    harness = make_harness(lambda _: completion(content="ok"))
    run_id = harness.open_run(authorization="Bearer tenant-a")

    assert harness.chat("hi", run_id=run_id, authorization="Bearer tenant-b").status_code == 404
    assert harness.close_run(run_id, authorization="Bearer tenant-b").status_code == 404
    assert harness.close_run(run_id, authorization="Bearer tenant-a").status_code == 204
    assert harness.provider.requests == []


def test_unknown_run_is_rejected_before_the_provider_is_called(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(lambda _: completion(content="ok"))
    response = harness.chat(f"I am {EMAIL}.", run_id="run_guessed")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "run_not_found"
    assert harness.provider.requests == []


def test_open_runs_are_capped(make_harness: HarnessFactory) -> None:
    harness = make_harness(lambda _: completion(content="ok"), max_runs=1)
    harness.open_run()
    response = harness.client.post("/v1/runs", headers={"authorization": "Bearer test-key"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "run_limit_reached"


def test_forgotten_run_is_reclaimed_by_the_backstop_and_reported() -> None:
    now = [0.0]
    expired_refs: list[str] = []
    registry = RunRegistry(
        vault_factory=lambda: RunVault(max_tokens=10, ttl_ms=60_000),
        max_runs=10,
        max_age_seconds=60,
        on_expire=expired_refs.append,
        clock=lambda: now[0],
    )
    run = registry.open(owner="Bearer test-key")
    run.vault.set("<EMAIL_00000000-0000-4000-8000-000000000000>", EMAIL)

    now[0] = 61.0
    with pytest.raises(RunNotFoundError):
        registry.get(run.run_id, owner="Bearer test-key")
    assert run.vault.size() == 0
    assert expired_refs == [run.ref]
