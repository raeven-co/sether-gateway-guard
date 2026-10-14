"""Tool policy: which tools may run and where restored PII may flow.

Unknown keys are rejected, so a typo such as ``restore_feilds`` fails at
startup instead of silently changing what reaches a tool.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contract import REVIEW_TOOL_NAME
from .jsonwalk import JsonPath, parse_field_path

_FIELD_PATH_RE = re.compile(r"[A-Za-z_]\w*(?:\[\])?(?:\.[A-Za-z_]\w*(?:\[\])?)*", re.ASCII)
_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_TOKEN_TYPE_RE = re.compile(r"[A-Z][A-Z0-9_]*")


class PolicyError(ValueError):
    """The policy cannot be enforced with the configured detectors."""


def _check_field_paths(specs: Iterable[str]) -> None:
    for spec in specs:
        if _FIELD_PATH_RE.fullmatch(spec) is None:
            raise ValueError(f"invalid field path {spec!r}; use keys, dots and [] only")


type MessageRole = Literal["system", "developer", "user", "assistant", "tool"]


class TokenFieldRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    types: frozenset[str] = Field(min_length=1)
    required: bool = True
    sources: frozenset[MessageRole] | None = Field(default=None, min_length=1)
    """Message roles a token must have been issued from. None accepts any role."""

    @field_validator("types")
    @classmethod
    def _uppercase_types(cls, types: frozenset[str]) -> frozenset[str]:
        invalid = sorted(t for t in types if _TOKEN_TYPE_RE.fullmatch(t) is None)
        if invalid:
            raise ValueError(f"token types must be upper-case detector types, got {invalid}")
        return types


class ToolPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    restore_fields: frozenset[str] = frozenset()
    token_fields: dict[str, TokenFieldRule] = Field(default_factory=dict)

    @field_validator("restore_fields")
    @classmethod
    def _valid_restore_fields(cls, fields: frozenset[str]) -> frozenset[str]:
        _check_field_paths(fields)
        return fields

    @field_validator("token_fields")
    @classmethod
    def _valid_token_fields(cls, fields: dict[str, TokenFieldRule]) -> dict[str, TokenFieldRule]:
        _check_field_paths(fields.keys())
        return fields

    def restore_paths(self) -> frozenset[JsonPath]:
        return frozenset(parse_field_path(spec) for spec in self.restore_fields)

    def referenced_token_types(self) -> frozenset[str]:
        return frozenset(t for rule in self.token_fields.values() for t in rule.types)


class GuardPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    tools: dict[str, ToolPolicy]

    @field_validator("tools")
    @classmethod
    def _valid_tool_names(cls, tools: dict[str, ToolPolicy]) -> dict[str, ToolPolicy]:
        for name in tools:
            if _TOOL_NAME_RE.fullmatch(name) is None:
                raise ValueError(f"invalid tool name {name!r}")
            if name == REVIEW_TOOL_NAME:
                raise ValueError(f"{REVIEW_TOOL_NAME!r} is reserved for the review marker")
        return tools

    def check_detector_coverage(self, detector_types: frozenset[str]) -> None:
        """Fail at startup if a rule names a token type no detector can issue."""
        referenced = {t for tool in self.tools.values() for t in tool.referenced_token_types()}
        missing = sorted(referenced - detector_types)
        if missing:
            raise PolicyError(f"policy references token types no detector emits: {missing}")


def load_policy(path: Path) -> GuardPolicy:
    with path.open(encoding="utf-8") as handle:
        raw: object = yaml.safe_load(handle)
    return GuardPolicy.model_validate(raw)
