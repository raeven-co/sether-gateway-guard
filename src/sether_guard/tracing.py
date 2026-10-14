"""A minimal tracing hook shaped like the OpenTelemetry API.

An OpenTelemetry tracer fits with a three-line adapter whose ``start_span``
returns ``otel_tracer.start_as_current_span(name)``. Attribute names follow the
OTel GenAI conventions where one exists (``gen_ai.*``) and use ``sether_guard.*``
otherwise. Attributes are counts and identifiers, never message content.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol

type AttributeValue = str | int | float | bool


class Span(Protocol):
    def set_attribute(self, key: str, value: AttributeValue) -> None: ...


class Tracer(Protocol):
    def start_span(self, name: str) -> AbstractContextManager[Span]: ...


class _NoopSpan:
    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """Discard the attribute."""


class NoopTracer:
    @contextmanager
    def start_span(self, name: str) -> Iterator[Span]:  # noqa: ARG002 - protocol signature
        yield _NoopSpan()
