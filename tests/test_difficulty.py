"""Unit tests for difficulty judging helpers.

Covers the pure-Python parsing and logprob-expectation logic. The
vLLM-dependent ``DifficultyJudge`` class itself is exercised end-to-end
on a GPU box, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import pytest

# difficulty.py imports ``vllm`` at module level. On CPU-only hosts where
# vllm isn't installed, skip the whole module cleanly.
pytest.importorskip("vllm")

from guac.judge.difficulty import (  # noqa: E402, I001 — importorskip must run before this import
    compute_continuous_difficulty,
    parse_difficulty_score,
    shard_records,
)


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #

@dataclass
class MockLogprob:
    """Stand-in for ``vllm.Logprob`` — only ``.logprob`` is read."""
    logprob: float


class DigitTokenizer:
    """Minimal tokenizer that decodes token ids via a supplied map.

    Accepts a ``{token_id: decoded_str}`` dict. Tokens not in the map
    decode to ``"?"`` so the helper's digit-matching branch treats them
    as non-digit noise.
    """

    def __init__(self, mapping: Dict[int, str]):
        self._mapping = mapping

    def decode(self, ids):
        return self._mapping.get(int(ids[0]), "?")


def _lp(p: float) -> MockLogprob:
    """Build a MockLogprob from a raw probability."""
    return MockLogprob(logprob=math.log(p))


# --------------------------------------------------------------------------- #
# compute_continuous_difficulty
# --------------------------------------------------------------------------- #

def test_expectation_basic():
    """{1,2,3} with p={0.1,0.5,0.4} -> E = 1*0.1 + 2*0.5 + 3*0.4 = 2.3."""
    tok = DigitTokenizer({11: "1", 12: "2", 13: "3"})
    lp_dict = {11: _lp(0.1), 12: _lp(0.5), 13: _lp(0.4)}
    exp, probs = compute_continuous_difficulty(lp_dict, tok, score_max=10)

    assert exp == pytest.approx(2.3, rel=1e-9)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert set(probs.keys()) == {"1", "2", "3"}


def test_missing_digits_renormalize():
    """Only digits 2 and 4 present — subset-softmax should renormalize.

    Raw 0.2 and 0.6 normalize to 0.25 and 0.75 → E = 2*0.25 + 4*0.75 = 3.5.
    """
    tok = DigitTokenizer({12: "2", 14: "4"})
    lp_dict = {12: _lp(0.2), 14: _lp(0.6)}
    exp, probs = compute_continuous_difficulty(lp_dict, tok, score_max=10)

    assert exp == pytest.approx(3.5, rel=1e-9)
    assert probs["2"] == pytest.approx(0.25, rel=1e-9)
    assert probs["4"] == pytest.approx(0.75, rel=1e-9)


def test_duplicate_digit_tokens_keep_max_logprob():
    """Two token ids decoding to the same digit: larger logprob wins."""
    tok = DigitTokenizer({12: "2", 99: " 2"})
    lp_dict = {12: _lp(0.1), 99: _lp(0.6)}
    exp, probs = compute_continuous_difficulty(lp_dict, tok, score_max=10)

    assert set(probs.keys()) == {"2"}
    assert probs["2"] == pytest.approx(1.0, rel=1e-9)
    assert exp == pytest.approx(2.0, rel=1e-9)


def test_score_max_10_excludes_10():
    """For ``score_max=10`` the multi-token ``"10"`` must be ignored."""
    tok = DigitTokenizer({12: "2", 13: "3", 100: "10"})
    lp_dict = {12: _lp(0.3), 13: _lp(0.5), 100: _lp(0.2)}
    exp, probs = compute_continuous_difficulty(lp_dict, tok, score_max=10)

    assert "10" not in probs
    assert set(probs.keys()) == {"2", "3"}
    # renormalised over {2, 3}: p(2)=0.375, p(3)=0.625
    assert exp == pytest.approx(2 * 0.375 + 3 * 0.625, rel=1e-9)


def test_score_max_5_limits_digit_range():
    """``score_max=5`` should drop digits above 5 (e.g., ``"7"``)."""
    tok = DigitTokenizer({11: "1", 12: "2", 15: "5", 17: "7"})
    lp_dict = {11: _lp(0.1), 12: _lp(0.3), 15: _lp(0.3), 17: _lp(0.3)}
    exp, probs = compute_continuous_difficulty(lp_dict, tok, score_max=5)

    assert set(probs.keys()) == {"1", "2", "5"}
    assert sum(probs.values()) == pytest.approx(1.0, rel=1e-9)
    z = 0.1 + 0.3 + 0.3
    assert probs["1"] == pytest.approx(0.1 / z, rel=1e-9)
    assert probs["5"] == pytest.approx(0.3 / z, rel=1e-9)
    assert exp == pytest.approx(
        1 * (0.1 / z) + 2 * (0.3 / z) + 5 * (0.3 / z), rel=1e-9
    )


def test_empty_or_none_logprobs_returns_none():
    tok = DigitTokenizer({})
    assert compute_continuous_difficulty(None, tok, score_max=10) == (None, None)
    assert compute_continuous_difficulty({}, tok, score_max=10) == (None, None)


def test_non_digit_tokens_yield_none():
    """Top-k with no digit tokens at all -> (None, None)."""
    tok = DigitTokenizer({50: "the", 51: "a", 52: "score"})
    lp_dict = {50: _lp(0.5), 51: _lp(0.3), 52: _lp(0.2)}
    exp, probs = compute_continuous_difficulty(lp_dict, tok, score_max=10)

    assert exp is None
    assert probs is None


# --------------------------------------------------------------------------- #
# parse_difficulty_score
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "response,expected",
    [
        ("7", 7.0),
        ("Level 3", 3.0),
        ("Difficulty: 8", 8.0),
        ("5/10", 5.0),
        ("7.5", 7.5),
        ("Difficulty: 7.2", 7.2),
        ("10", 10.0),
        ("", None),
        ("   ", None),
        ("no digits here", None),
    ],
)
def test_parse_difficulty_score_default_range(response, expected):
    assert parse_difficulty_score(response) == expected


def test_parse_difficulty_score_respects_score_max():
    # Default score_max=10 accepts the 10 bucket.
    assert parse_difficulty_score("10") == 10.0
    # score_max=5 rejects 7 (out of rubric).
    assert parse_difficulty_score("7", score_max=5) is None
    # score_max=5 accepts the boundary.
    assert parse_difficulty_score("5", score_max=5) == 5.0
    # Float values work too.
    assert parse_difficulty_score("4.7", score_max=5) == 4.7
    # Skips an OOR leading integer and finds a later valid one.
    assert parse_difficulty_score("15 ... 3", score_max=5) == 3.0


def test_parse_difficulty_score_first_valid_wins():
    assert parse_difficulty_score("scores: 3, 7, 9") == 3.0


# --------------------------------------------------------------------------- #
# shard_records (data-parallel judging helper)
# --------------------------------------------------------------------------- #

def test_shard_records_partitions_and_covers_all():
    """Every record ends up in exactly one shard; shards are roughly balanced."""
    records = [{"id": i} for i in range(17)]
    shards = [shard_records(records, rank=r, world_size=4) for r in range(4)]

    all_ids = sorted(r["id"] for s in shards for r in s)
    assert all_ids == list(range(17))
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1


def test_shard_records_no_sharding_returns_copy():
    """``world_size<=1`` should return a copy of the original list unchanged."""
    records = [{"id": 1}, {"id": 2}]
    out = shard_records(records, rank=0, world_size=1)
    assert out == records
    assert out is not records  # defensive copy, not the same list object


def test_shard_records_index_modulo_rule():
    """Rank r should receive exactly the records where index % world_size == r."""
    records = [{"id": i} for i in range(12)]
    W = 3
    for r in range(W):
        shard = shard_records(records, rank=r, world_size=W)
        assert [rec["id"] for rec in shard] == [i for i in range(12) if i % W == r]


def test_shard_records_rejects_invalid_rank():
    with pytest.raises(ValueError):
        shard_records([{"id": 1}], rank=5, world_size=4)
    with pytest.raises(ValueError):
        shard_records([{"id": 1}], rank=-1, world_size=4)
