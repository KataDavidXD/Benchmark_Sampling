from __future__ import annotations

import json

from benchmark.sampling.engine import SamplingEngine
from benchmark.sampling.random import random_permutation
from benchmark.sampling.samplers.random import RandomSampler
from benchmark.sampling.types import BenchmarkItem, SamplingState, StratumStats


def _item(item_id: str, stratum: str) -> BenchmarkItem:
    return BenchmarkItem(
        item_id=item_id,
        benchmark="synthetic",
        stratum=stratum,
        payload={},
        target=None,
        metadata={},
    )


class _Adapter:
    name = "ultradomain"

    def __init__(self) -> None:
        self._items = [
            *(_item(f"a-{index}", "a") for index in range(10)),
            *(_item(f"b-{index}", "b") for index in range(10)),
        ]

    def load_items(self) -> list[BenchmarkItem]:
        return list(self._items)


def _strata() -> tuple[
    dict[str, list[BenchmarkItem]],
    dict[str, StratumStats],
]:
    strata = {
        "a": [_item(f"a-{index}", "a") for index in range(6)],
        "b": [_item(f"b-{index}", "b") for index in range(4)],
    }
    stats = {
        label: StratumStats(label, len(items))
        for label, items in strata.items()
    }
    return strata, stats


def test_sampling_engine_defaults_to_uniform_random_without_replacement() -> None:
    first = SamplingEngine(adapter=_Adapter(), budget=12, seed=42).run()
    second = SamplingEngine(adapter=_Adapter(), budget=12, seed=42).run()

    assert first.state.sampler_type == "random"
    assert first.realization.realized_items == second.realization.realized_items
    assert len(first.realization.realized_items) == 12
    assert len(set(first.realization.realized_items)) == 12
    assert sum(first.realization.allocation) == 12


def test_random_sampler_state_round_trip_continues_exact_rng_stream() -> None:
    strata, stats = _strata()
    uninterrupted = RandomSampler(seed=7)
    uninterrupted.select(strata, stats, 5)
    saved = json.loads(json.dumps(uninterrupted.get_state()))
    expected = uninterrupted.select(strata, stats, 5)

    restored = RandomSampler(seed=999)
    restored.set_state(saved)
    actual = restored.select(strata, stats, 5)

    assert actual.realized_items == expected.realized_items
    assert actual.allocation == expected.allocation


def test_sampling_state_json_restores_nested_sampler_rng_state() -> None:
    result = SamplingEngine(adapter=_Adapter(), budget=8, seed=3).run()
    restored = SamplingState.from_json(result.state.to_json())

    sampler = RandomSampler(seed=999)
    sampler.set_state(restored.sampler_state)
    assert sampler.get_state()["rng_state"] == result.state.sampler_state["rng_state"]


def test_random_permutation_matches_seeded_shuffle_and_preserves_input() -> None:
    values = tuple(range(20))
    ordered = random_permutation(values, seed=123)

    assert values == tuple(range(20))
    assert sorted(ordered) == list(values)
    assert ordered == random_permutation(values, seed=123)
    assert ordered != random_permutation(values, seed=124)
