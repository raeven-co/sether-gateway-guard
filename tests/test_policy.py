"""Policy validation: fail at startup, never silently at request time."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sether_guard import GuardPolicy, GuardSettings, create_app, load_policy
from sether_guard.detection import build_detectors, detector_types
from sether_guard.errors import VaultCapacityError
from sether_guard.policy import PolicyError
from sether_guard.vault import RunVault

EXAMPLE_POLICY = Path(__file__).resolve().parent.parent / "policy.example.yaml"


def test_example_policy_is_valid_for_the_default_detectors() -> None:
    policy = load_policy(EXAMPLE_POLICY)
    policy.check_detector_coverage(detector_types(build_detectors(("basic",), None)))
    assert set(policy.tools) == {"send_email", "lookup_order", "reschedule_delivery"}


@pytest.mark.parametrize("tool_name", ["needs_human_review", "send email", ""])
def test_reserved_or_invalid_tool_names_are_rejected(tool_name: str) -> None:
    with pytest.raises(ValidationError):
        GuardPolicy.model_validate({"version": 1, "tools": {tool_name: {}}})


@pytest.mark.parametrize("field_path", ["to..cc", "cc[0]", "to.", "[]", "to cc"])
def test_invalid_field_paths_are_rejected(field_path: str) -> None:
    with pytest.raises(ValidationError):
        GuardPolicy.model_validate(
            {"version": 1, "tools": {"send_email": {"restore_fields": [field_path]}}}
        )


def test_unknown_token_source_role_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GuardPolicy.model_validate(
            {
                "version": 1,
                "tools": {
                    "send_email": {"token_fields": {"to": {"types": ["EMAIL"], "sources": ["web"]}}}
                },
            }
        )


def test_misspelled_policy_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GuardPolicy.model_validate(
            {"version": 1, "tools": {"send_email": {"restore_feilds": ["to"]}}}
        )


def test_policy_naming_an_undetectable_token_type_fails_at_startup() -> None:
    policy = GuardPolicy.model_validate(
        {"version": 1, "tools": {"greet": {"token_fields": {"name": {"types": ["NAME"]}}}}}
    )
    # NAME comes from Sether's opt-in identity pack, which is not enabled by default.
    with pytest.raises(PolicyError, match="NAME"):
        create_app(GuardSettings(), policy)


def test_vault_fails_closed_at_capacity_instead_of_evicting() -> None:
    vault = RunVault(max_tokens=1, ttl_ms=60_000)
    vault.set("<EMAIL_00000000-0000-4000-8000-000000000001>", "a@example.com")

    with pytest.raises(VaultCapacityError):
        vault.set("<EMAIL_00000000-0000-4000-8000-000000000002>", "b@example.com")
    assert vault.get("<EMAIL_00000000-0000-4000-8000-000000000001>") == "a@example.com"
