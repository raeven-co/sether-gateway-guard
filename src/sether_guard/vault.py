"""A Sether ``Vault`` scoped to one request or one run."""

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from sether import MemoryVault

from .errors import VaultCapacityError
from .tokens import token_type_of

UNATTRIBUTED_SOURCE = "unattributed"


class RunVault:
    """Implements Sether's ``Vault`` protocol on top of ``sether.MemoryVault``.

    Adds what the gate needs:

    * a ledger of issued tokens and the message role that issued each one, so
      "was this token issued in this run, and from where?" has an exact answer;
    * a hard cap that fails closed. MemoryVault evicts the least recently used
      entry at its cap, which would silently turn a valid token into an
      unknown one mid-run. Here the request fails instead.

    Holds plaintext in process memory only. Callers must ``clear()`` it.
    """

    def __init__(self, max_tokens: int, ttl_ms: int) -> None:
        self._store = MemoryVault(max_entries=max_tokens, ttl_ms=ttl_ms)
        self._max_tokens = max_tokens
        self._sources: dict[str, str] = {}
        self._type_counts: Counter[str] = Counter()
        self._current_source = UNATTRIBUTED_SOURCE

    # --- sether.Vault protocol ---------------------------------------------

    def set(self, token: str, value: str) -> None:
        if token not in self._sources:
            if len(self._sources) >= self._max_tokens:
                raise VaultCapacityError
            self._sources[token] = self._current_source
            self._type_counts[token_type_of(token)] += 1
        self._store.set(token, value)

    def get(self, token: str) -> str | None:
        return self._store.get(token)

    def has(self, token: str) -> bool:
        return self._store.has(token)

    def delete(self, token: str) -> bool:
        self._sources.pop(token, None)
        return self._store.delete(token)

    def size(self) -> int:
        return self._store.size()

    def clear(self) -> None:
        self._store.clear()
        self._sources.clear()
        self._type_counts.clear()

    # --- ledger --------------------------------------------------------------

    @contextmanager
    def issuing_for(self, source: str) -> Iterator[None]:
        """Attribute tokens issued inside this block to ``source`` (a message role)."""
        previous, self._current_source = self._current_source, source
        try:
            yield
        finally:
            self._current_source = previous

    def issued_sources(self) -> dict[str, str]:
        """Restorable (unexpired) issued tokens mapped to the role that issued them."""
        return {token: src for token, src in self._sources.items() if self._store.has(token)}

    def issued_type_counts(self) -> dict[str, int]:
        return dict(self._type_counts)


type VaultFactory = Callable[[], RunVault]
