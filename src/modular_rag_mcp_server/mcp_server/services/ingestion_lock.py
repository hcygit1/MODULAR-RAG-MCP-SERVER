"""Per-document locking for concurrent ingestion requests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator


@dataclass
class _LockEntry:
    lock: Lock = field(default_factory=Lock)
    references: int = 0


class KeyedLockRegistry:
    """Serialize requests for the same file, collection and profile."""

    def __init__(self) -> None:
        self._guard = Lock()
        self._entries: dict[tuple[str, str, str], _LockEntry] = {}

    @contextmanager
    def hold(self, key: tuple[str, str, str]) -> Iterator[None]:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _LockEntry()
                self._entries[key] = entry
            entry.references += 1

        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.references -= 1
                if entry.references == 0 and self._entries.get(key) is entry:
                    del self._entries[key]

    def size(self) -> int:
        with self._guard:
            return len(self._entries)


INGESTION_LOCKS = KeyedLockRegistry()
