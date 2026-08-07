"""Uniform random sampling without replacement across the loaded pool."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from benchmark.sampling.samplers.base import BaseSampler
from benchmark.sampling.types import BenchmarkItem, ItemRealization, StratumStats


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


class RandomSampler(BaseSampler):
    """Draw a uniform subset of the complete population without replacement.

    Strata are used only to report the realized allocation. They do not bias
    the draw; every benchmark item has the same inclusion probability.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._step = 0

    def select(
        self,
        strata: dict[str, list[BenchmarkItem]],
        strata_stats: dict[str, StratumStats],
        budget: int,
    ) -> ItemRealization:
        del strata_stats
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative integer")

        labels = sorted(strata)
        population: list[tuple[str, BenchmarkItem]] = [
            (label, item)
            for label in labels
            for item in strata[label]
        ]
        if len({item.item_id for _label, item in population}) != len(population):
            raise ValueError("random sampling requires globally unique item IDs")

        rng_state_before = self._rng.getstate()
        self._rng.shuffle(population)
        selected = population[: min(budget, len(population))]
        counts = {label: 0 for label in labels}
        for label, _item in selected:
            counts[label] += 1
        self._step += 1

        return ItemRealization(
            allocation=[counts[label] for label in labels],
            realized_items=[item.item_id for _label, item in selected],
            rng_state_before=rng_state_before,
        )

    def get_state(self) -> dict[str, Any]:
        return {"rng_state": self._rng.getstate(), "step": self._step}

    def set_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("random sampler state must be a mapping")
        self._rng.setstate(_tuple_tree(state["rng_state"]))
        step = state["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("random sampler step must be a non-negative integer")
        self._step = step


__all__ = ["RandomSampler"]
