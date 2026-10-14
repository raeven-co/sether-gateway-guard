"""Strict JSON parsing plus path-aware walking of string values.

A path is a tuple of segments: a ``str`` is an object key and ``LIST_ITEM``
(``None``) is any list element. Policy paths such as ``cc[]`` or
``customer.email`` parse to the same shape, so an object key that is literally
named ``"cc[]"`` can never alias the list path ``cc[]``.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import NoReturn, cast

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]
type JsonPath = tuple[str | None, ...]

LIST_ITEM = None
MAX_DEPTH = 32


@dataclass(frozen=True)
class StringSite:
    """One string in a JSON document: a value, or an object key."""

    path: JsonPath
    value: str
    is_key: bool


def parse_json_object(text: str) -> JsonObject | None:
    """Parse ``text`` as one JSON object, or return None.

    Rejects duplicate keys and NaN/Infinity so the value the gate inspects is
    exactly the value a downstream parser would see.
    """
    try:
        value = json.loads(
            text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict) or _exceeds_depth(value, MAX_DEPTH):
        return None
    return cast(JsonObject, value)


def dumps_json(value: object) -> str:
    """Serialize compactly. Quotes and control characters in values are escaped."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse_field_path(spec: str) -> JsonPath:
    """``"cc[]"`` -> ``("cc", None)``; ``"customer.email"`` -> ``("customer", "email")``."""
    segments: list[str | None] = []
    for part in spec.split("."):
        if part.endswith("[]"):
            segments.extend((part[:-2], LIST_ITEM))
        else:
            segments.append(part)
    return tuple(segments)


def iter_string_sites(node: JsonValue, path: JsonPath = ()) -> Iterator[StringSite]:
    if isinstance(node, str):
        yield StringSite(path, node, is_key=False)
    elif isinstance(node, list):
        for item in node:
            yield from iter_string_sites(item, (*path, LIST_ITEM))
    elif isinstance(node, dict):
        for key, value in node.items():
            yield StringSite(path, key, is_key=True)
            yield from iter_string_sites(value, (*path, key))


def map_string_values(
    node: JsonValue, transform: Callable[[JsonPath, str], str], path: JsonPath = ()
) -> JsonValue:
    """Rebuild ``node`` with every string value transformed. Keys are kept as is."""
    if isinstance(node, str):
        return transform(path, node)
    if isinstance(node, list):
        return [map_string_values(item, transform, (*path, LIST_ITEM)) for item in node]
    if isinstance(node, dict):
        return {
            key: map_string_values(value, transform, (*path, key)) for key, value in node.items()
        }
    return node


def map_all_strings(node: JsonValue, transform: Callable[[str], str]) -> JsonValue:
    """Rebuild ``node`` with every string transformed, object keys included."""
    if isinstance(node, str):
        return transform(node)
    if isinstance(node, list):
        return [map_all_strings(item, transform) for item in node]
    if isinstance(node, dict):
        return {transform(key): map_all_strings(value, transform) for key, value in node.items()}
    return node


def _reject_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {name}")


def _exceeds_depth(node: JsonValue, limit: int) -> bool:
    stack: list[tuple[JsonValue, int]] = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > limit:
            return True
        if isinstance(current, dict):
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
    return False
