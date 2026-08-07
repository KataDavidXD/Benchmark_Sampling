"""Deterministic uniform sampling helpers shared by experiment runtimes."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def random_permutation(items: Sequence[T], *, seed: int) -> tuple[T, ...]:
    """Return a seeded uniform permutation without mutating ``items``."""

    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


__all__ = ["random_permutation"]
