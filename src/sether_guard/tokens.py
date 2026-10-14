"""Find token-shaped strings in model output and classify them against a run.

Sether 0.2.0 issues tokens shaped ``<TYPE_uuid4>`` and its restore step
silently leaves any token it does not know. The gate needs the opposite: every
token-shaped string must be accounted for. A "token-shaped" string here is
anything Sether's restore grammar would recognise, widened to also catch the
ways a model mangles tokens (changed case, inner whitespace, dropped brackets).
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

OTHER_TOKEN_TYPE = "OTHER"

_MIN_ID_LENGTH = 8  # Sether's restore grammar requires 8+ id characters.
_MAX_ID_EDITS = 3  # Random uuid4 ids differ in ~30 of 32 positions; 3 edits is drift.

_SEGMENT_RE = re.compile(r"<([^<>]*)>")
_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HEX_OR_DASH = frozenset("0123456789abcdefABCDEF-")
_BARE_TOKEN_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]*)_([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
)


class TokenStatus(StrEnum):
    ISSUED = "issued"
    ALTERED = "altered"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TokenFinding:
    token_type: str
    """Detector type, or ``OTHER`` when the model invented one. Safe to log."""

    status: TokenStatus


def token_type_of(token: str) -> str:
    return token.strip("<> \t").rpartition("_")[0].upper()


class TokenIndex:
    """Snapshot of the tokens issued in one run, used to classify model output.

    ``issued`` maps each token to the message role that issued it.
    """

    def __init__(self, issued: Mapping[str, str], known_types: frozenset[str]) -> None:
        self._issued = dict(issued)
        self._issued_ids = frozenset(_token_id(token) for token in self._issued)
        self._known_types = known_types

    def is_issued(self, text: str) -> bool:
        """True only when ``text`` is byte-identical to a token issued in this run."""
        return text in self._issued

    def source_of(self, token: str) -> str | None:
        return self._issued.get(token)

    def find(self, text: str) -> list[TokenFinding]:
        return [
            TokenFinding(self._type_label(candidate), self._classify(candidate))
            for candidate in self._candidates(text)
        ]

    def _candidates(self, text: str) -> list[str]:
        spans = [m.span() for m in _SEGMENT_RE.finditer(text) if _is_token_shape(m.group(1))]
        candidates = [text[start:end] for start, end in spans]
        for match in _BARE_TOKEN_RE.finditer(text):
            if match.group(1) in self._known_types and not _within(match.span(), spans):
                candidates.append(match.group(0))
        return candidates

    def _classify(self, candidate: str) -> TokenStatus:
        if candidate in self._issued:
            return TokenStatus.ISSUED
        candidate_id = _token_id(candidate)
        if any(_is_drift(candidate_id, issued_id) for issued_id in self._issued_ids):
            return TokenStatus.ALTERED
        return TokenStatus.UNKNOWN

    def _type_label(self, candidate: str) -> str:
        token_type = token_type_of(candidate)
        return token_type if token_type in self._known_types else OTHER_TOKEN_TYPE


def _is_token_shape(inner: str) -> bool:
    type_part, separator, id_part = inner.strip().rpartition("_")
    return (
        bool(separator)
        and len(id_part) >= _MIN_ID_LENGTH
        and set(id_part) <= _HEX_OR_DASH
        and _TYPE_RE.fullmatch(type_part) is not None
    )


def _token_id(token: str) -> str:
    return token.strip("<> \t").rpartition("_")[2].replace("-", "").lower()


def _is_drift(candidate_id: str, issued_id: str) -> bool:
    """Same id with different framing, a truncated id, or up to 3 substituted characters.

    Diagnostic only: altered and unknown tokens are both blocked. Insertions or
    deletions inside an id are reported as unknown.
    """
    if candidate_id == issued_id:
        return True
    if len(candidate_id) >= _MIN_ID_LENGTH and issued_id.startswith(candidate_id):
        return True
    if len(candidate_id) != len(issued_id):
        return False
    edits = sum(a != b for a, b in zip(candidate_id, issued_id, strict=True))
    return edits <= _MAX_ID_EDITS


def _within(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(outer_start <= start and end <= outer_end for outer_start, outer_end in spans)
