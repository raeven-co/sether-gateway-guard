"""Token-shape detection: catch mangled tokens without flagging ordinary text."""

import pytest

from sether_guard.tokens import TokenFinding, TokenIndex, TokenStatus

ISSUED = "<EMAIL_3f2a9c4e-1b7d-4e8a-9c3f-2d1e0f9a8b7c>"
INDEX = TokenIndex({ISSUED: "user"}, known_types=frozenset({"EMAIL", "PHONE"}))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"write to {ISSUED}", TokenFinding("EMAIL", TokenStatus.ISSUED)),
        (
            "<EMAIL_3f2a9c4e-1b7d-4e8a-9c3f-2d1e0f9a8b7d>",
            TokenFinding("EMAIL", TokenStatus.ALTERED),
        ),
        (
            "< EMAIL_3f2a9c4e-1b7d-4e8a-9c3f-2d1e0f9a8b7c >",
            TokenFinding("EMAIL", TokenStatus.ALTERED),
        ),
        (
            "<EMAIL_00000000-0000-4000-8000-000000000000>",
            TokenFinding("EMAIL", TokenStatus.UNKNOWN),
        ),
        (
            "<SECRET_00000000-0000-4000-8000-000000000000>",
            TokenFinding("OTHER", TokenStatus.UNKNOWN),
        ),
    ],
    ids=["issued", "edited", "spaced", "unknown", "invented_type"],
)
def test_token_classification(text: str, expected: TokenFinding) -> None:
    assert INDEX.find(text) == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "doc_3f2a9c4e-1b7d-4e8a-9c3f-2d1e0f9a8b7c",
        "ORDER_3f2a9c4e-1b7d-4e8a-9c3f-2d1e0f9a8b7c",
        "<step_2>",
        "<div>a < b > c</div>",
        "Order 4532 for <customer_name>",
    ],
    ids=["lowercase_id", "unknown_bare_type", "short_id", "markup", "placeholder"],
)
def test_ordinary_text_is_not_token_shaped(text: str) -> None:
    assert INDEX.find(text) == []
