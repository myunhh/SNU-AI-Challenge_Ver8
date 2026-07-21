"""BALANCED8 (균형 2-코셋 8뷰) 속성 테스트.

핵심 속성: ① 각 입력이 각 슬롯을 정확히 2회 방문(균형) ② 역원 닫힘(셔플 방향
규약 무관) ③ 비대합 뷰(4-cycle)를 통과해도 리맵·집계가 정답을 보존 ④ 기존
경로(BALANCED4·시드셔플 TTA8) 불변.
"""

import numpy as np

import pytest

from snuai import perm
from snuai.infer.tta import (BALANCED4, BALANCED8, TTAConfig, tta_scores,
                             tta_shuffles)


def test_balanced8_shape_and_members():
    assert len(BALANCED8) == 8
    assert len(set(BALANCED8)) == 8
    assert BALANCED8[0] == perm.IDENTITY
    assert set(BALANCED4) <= set(BALANCED8)
    for s in BALANCED8:
        assert perm.is_perm(s)


def test_balanced8_each_input_each_slot_exactly_twice():
    counts = {(img, slot): 0 for img in range(4) for slot in range(4)}
    for s in BALANCED8:
        placed = perm.apply_shuffle([0, 1, 2, 3], s)
        for slot, img in enumerate(placed):
            counts[(img, slot)] += 1
    assert all(c == 2 for c in counts.values()), counts


def test_balanced4_each_input_each_slot_exactly_once():
    counts = {(img, slot): 0 for img in range(4) for slot in range(4)}
    for s in BALANCED4:
        placed = perm.apply_shuffle([0, 1, 2, 3], s)
        for slot, img in enumerate(placed):
            counts[(img, slot)] += 1
    assert all(c == 1 for c in counts.values()), counts


def test_balanced8_inverse_closed():
    assert {perm.inverse(s) for s in BALANCED8} == set(BALANCED8)


def test_tta_shuffles_balanced8_paths():
    assert tta_shuffles(TTAConfig(n_views=8, balanced8=True)) == list(BALANCED8)
    # BudgetGuard 뷰 축소(n_views=1) 경로는 항등만
    assert tta_shuffles(TTAConfig(n_views=1, balanced8=True)) == [perm.IDENTITY]
    with pytest.raises(ValueError):
        tta_shuffles(TTAConfig(n_views=5, balanced8=True))


def test_legacy_paths_unchanged():
    # TTA4 = Klein, TTA8(기존) = 항등+7 시드셔플 — balanced8 기본값이 이를 바꾸면 안 됨
    assert tta_shuffles(TTAConfig(n_views=4)) == list(BALANCED4)
    legacy8 = tta_shuffles(TTAConfig(n_views=8))
    assert len(legacy8) == 8 and legacy8[0] == perm.IDENTITY
    assert len(set(legacy8)) == 8
    assert legacy8 != list(BALANCED8)


def _make_scorer(truth_seq):
    """셔플된 배열의 실제 rank에 가까울수록 높은 점수를 주는 합성 스코어러."""
    def scorer_fn(shuffled):
        true_rank = tuple(truth_seq.index(v) for v in shuffled)
        return np.array([-float(perm.kendall_tau_distance(r, true_rank))
                         for r in perm.PERMS24])
    return scorer_fn


@pytest.mark.parametrize("cfg", [
    TTAConfig(n_views=8, balanced8=True),
    TTAConfig(n_views=8),          # 기존 시드셔플 경로도 동일 속성 유지
    TTAConfig(n_views=4),
])
def test_tta_roundtrip_preserves_answer(cfg):
    # 4-cycle 뷰(비대합)를 포함해도 리맵 후 argmax가 원본 공간 정답과 일치해야 한다.
    images = [10, 20, 30, 40]
    truth_seq = [30, 10, 40, 20]   # 시간순: 30 → 10 → 40 → 20
    true_rank = tuple(truth_seq.index(v) for v in images)
    scores, per_view = tta_scores(images, _make_scorer(truth_seq), cfg)
    assert len(per_view) == cfg.n_views
    assert int(np.argmax(scores)) == perm.index_of(true_rank)
    # 모든 뷰가 리맵 후 같은 정답을 가리켜야 함(합성 스코어러는 뷰-불변 진실 사용)
    for v in per_view:
        assert int(np.argmax(v)) == perm.index_of(true_rank)
