"""BALANCED24(전체 S4) 속성 테스트 — Klein V(4차)→Sylow-2(8차) 계열의 자연스러운
종점. 핵심 속성: ① S4 전체(24개 전부, 중복 없음) ② 각 입력이 각 슬롯을 정확히
6회 방문 ③ 부분군(당연히 — 군 전체이므로 폐쇄·역원 자명) ④ n_views==24는
balanced8 플래그·시드와 무관하게 무조건 이 세트를 반환 ⑤ 리맵·집계가 24개
전수(4-cycle 등 비대합 다수 포함)에서도 정답을 보존.
"""

import numpy as np

import pytest

from snuai import perm
from snuai.infer.tta import (BALANCED4, BALANCED8, BALANCED24, TTAConfig,
                             tta_scores, tta_shuffles)


def test_balanced24_is_full_s4():
    assert len(BALANCED24) == 24
    assert set(BALANCED24) == set(perm.PERMS24)
    assert BALANCED24[0] == perm.IDENTITY
    assert set(BALANCED4) <= set(BALANCED24)
    assert set(BALANCED8) <= set(BALANCED24)


def test_balanced24_each_input_each_slot_exactly_six_times():
    counts = {(img, slot): 0 for img in range(4) for slot in range(4)}
    for s in BALANCED24:
        placed = perm.apply_shuffle([0, 1, 2, 3], s)
        for slot, img in enumerate(placed):
            counts[(img, slot)] += 1
    assert all(c == 6 for c in counts.values()), counts


def test_balanced24_is_subgroup_of_s4():
    s = set(BALANCED24)
    for a in s:
        assert perm.inverse(a) in s
        for b in s:
            assert perm.compose(a, b) in s


def test_n_views_24_unconditional_and_ignores_balanced8_flag():
    # balanced8=True/False 무관 — n_views==24는 무조건 완전균형 세트
    assert tta_shuffles(TTAConfig(n_views=24)) == list(BALANCED24)
    assert tta_shuffles(TTAConfig(n_views=24, balanced8=True)) == list(BALANCED24)
    assert tta_shuffles(TTAConfig(n_views=24, seed=999)) == list(BALANCED24)


def test_legacy_paths_unchanged_by_24_addition():
    assert tta_shuffles(TTAConfig(n_views=4)) == list(BALANCED4)
    assert tta_shuffles(TTAConfig(n_views=8, balanced8=True)) == list(BALANCED8)
    legacy8 = tta_shuffles(TTAConfig(n_views=8))
    assert len(legacy8) == 8 and legacy8 != list(BALANCED8)


def _make_scorer(truth_seq):
    def scorer_fn(shuffled):
        true_rank = tuple(truth_seq.index(v) for v in shuffled)
        return np.array([-float(perm.kendall_tau_distance(r, true_rank))
                         for r in perm.PERMS24])
    return scorer_fn


def test_tta24_roundtrip_preserves_answer():
    images = [10, 20, 30, 40]
    truth_seq = [30, 10, 40, 20]
    true_rank = tuple(truth_seq.index(v) for v in images)
    scores, per_view = tta_scores(images, _make_scorer(truth_seq), TTAConfig(n_views=24))
    assert len(per_view) == 24
    assert int(np.argmax(scores)) == perm.index_of(true_rank)
    for v in per_view:
        assert int(np.argmax(v)) == perm.index_of(true_rank)
