"""scripts/decode_expected.py — 기대 부분점수 디코딩의 손계산 대조 (CPU 전용)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from decode_expected import decode_expected, score_matrix  # noqa: E402

from snuai import perm


def _scores_from_probs(probs_by_rank: dict) -> list[float]:
    """rank→확률 dict를 24-way 로그점수로 (softmax하면 원래 확률 복원)."""
    p = np.full(24, 1e-12)
    for rank, prob in probs_by_rank.items():
        p[perm.index_of(rank)] = prob
    return list(np.log(p / p.sum()))


def test_peaked_distribution_matches_argmax():
    # 고마진(뾰족한 분포)이면 expected 디코딩 = argmax — 기존 예측 보존
    mat = score_matrix("pairwise")
    scores = _scores_from_probs({(2, 0, 3, 1): 0.9, (0, 1, 2, 3): 0.1})
    assert decode_expected(scores, mat) == (2, 0, 3, 1)


def test_spread_distribution_moves_to_median_perm():
    # 손계산: q1=(1,0,2,3) 0.35, q2=(0,1,3,2) 0.35, 항등 0.30일 때
    #   E[항등] = 0.35·(5/6)·2 + 0.30·1 = 0.8833…
    #   E[q1]   = 0.35·1 + 0.35·(4/6) + 0.30·(5/6) = 0.8333…
    # → argmax(q1 또는 q2)와 달리 expected는 항등을 선택해야 한다
    mat = score_matrix("pairwise")
    scores = _scores_from_probs(
        {(1, 0, 2, 3): 0.35, (0, 1, 3, 2): 0.35, (0, 1, 2, 3): 0.30})
    assert decode_expected(scores, mat) == perm.IDENTITY


def test_score_matrix_diagonal_and_symmetry():
    for obj in ("pairwise", "position"):
        m = score_matrix(obj)
        assert np.allclose(np.diag(m), 1.0)
        assert np.allclose(m, m.T)
