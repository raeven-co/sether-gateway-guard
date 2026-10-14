"""The gate contract callers integrate against.

When the guard refuses a tool call it keeps the call's ``id`` and replaces the
function with ``REVIEW_TOOL_NAME``. Its arguments are a JSON object::

    {"blocked_tool": "send_email",
     "reasons": ["altered_token"],
     "policy_fields": ["to"]}

An agent loop that dispatches on function name therefore cannot run the
original tool by accident. Register a handler for ``needs_human_review`` to
route the call to a review queue, or treat it as a hard stop.
"""

from enum import StrEnum

REVIEW_TOOL_NAME = "needs_human_review"


class BlockReason(StrEnum):
    """Why a tool call was refused. Values are stable and safe to log."""

    TOOL_NOT_ALLOWED = "tool_not_allowed"
    """The tool is not listed in the policy (default deny)."""

    INVALID_ARGUMENTS = "invalid_arguments"
    """Arguments are not one JSON object (or use duplicate keys, NaN, deep nesting)."""

    ALTERED_TOKEN = "altered_token"
    """A token-shaped string resembles a token issued in this run but is not identical."""

    UNKNOWN_TOKEN = "unknown_token"
    """A token-shaped string was never issued in this run."""

    MISSING_REQUIRED_TOKEN = "missing_required_token"
    """A policy ``token_fields`` entry marked required is absent or empty."""

    VALUE_NOT_ISSUED_TOKEN = "value_not_issued_token"
    """A ``token_fields`` value is not exactly one token issued in this run."""

    TOKEN_TYPE_NOT_ALLOWED = "token_type_not_allowed"
    """A ``token_fields`` value is an issued token of a type the policy does not allow."""

    TOKEN_SOURCE_NOT_ALLOWED = "token_source_not_allowed"
    """A ``token_fields`` value was issued from a message role the policy does not trust,
    for example an address that first appeared in a tool result rather than from the user."""
