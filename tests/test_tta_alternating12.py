"""ALTERNATING12(A4, 짝순열 전부) 속성 테스트 — V(4)→D4(8)→S4(24) 계열과는 별개의
가지: "홀순열이 모델을 헷갈리게 하는가"를 격리 검증하기 위한 12뷰. 핵심 속성:
① 짝순열 12개 전부(중복 없음, 홀순열 없음) ② 부분군(폐쇄·역원 보유) ③ 각 입력이
각 슬롯을 정확히 3회 방문(균형) ④ n_views==12는 무조건 이 세트(다른 플래그·시드
무관) ⑤ 리맵·집계가 비대합 다수(3-사이클 8개) 포함에서도 정답을 보존.
"""

import numpy as np

import pytest

from snuai import perm
from snuai.infer.tta import (ALTERNATING12, BALANCED4, BALANCED8, BALANCED24,
                             TTAConfig, tta_scores, tta_shuffles)


def _is_even(p):
    return sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0


def test_alternating12_is_even_permutations_of_s4():
    assert len(ALTERNATING12) == 12
    assert len(set(ALTERNATING12)) == 12
    assert ALTERNATING12[0] == perm.IDENTITY
    assert set(ALTERNATING12) <= set(perm.PERMS24)
    assert all(_is_even(p) for p in ALTERNATING12)
    # 홀순열은 하나도 없어야 함(가설 검증의 전제)
    odd = [p for p in perm.PERMS24 if not _is_even(p)]
    assert not (set(ALTERNATING12) & set(odd))


def test_alternating12_is_subgroup_of_s4():
    s = set(ALTERNATING12)
    for a in s:
        assert perm.inverse(a) in s
        for b in s:
            assert perm.compose(a, b) in s


def test_alternating12_each_input_each_slot_exactly_three_times():
    counts = {(img, slot): 0 for img in range(4) for slot in range(4)}
    for s in ALTERNATING12:
        placed = perm.apply_shuffle([0, 1, 2, 3], s)
        for slot, img in enumerate(placed):
            counts[(img, slot)] += 1
    assert all(c == 3 for c in counts.values()), counts


def test_alternating12_not_subset_or_superset_of_balanced8():
    # A4는 D4(짝4+홀4 혼합)의 상위집합도 부분집합도 아니다 — 별개 구조라는 근거.
    a4, d4 = set(ALTERNATING12), set(BALANCED8)
    assert not (a4 <= d4)
    assert not (d4 <= a4)
    assert a4 & d4  # 교집합은 있음(짝순열 쪽)


def test_n_views_12_unconditional_and_ignores_other_flags():
    assert tta_shuffles(TTAConfig(n_views=12)) == list(ALTERNATING12)
    assert tta_shuffles(TTAConfig(n_views=12, balanced8=True)) == list(ALTERNATING12)
    assert tta_shuffles(TTAConfig(n_views=12, seed=999)) == list(ALTERNATING12)


def test_legacy_paths_unchanged_by_12_addition():
    assert tta_shuffles(TTAConfig(n_views=4)) == list(BALANCED4)
    assert tta_shuffles(TTAConfig(n_views=8, balanced8=True)) == list(BALANCED8)
    assert tta_shuffles(TTAConfig(n_views=24)) == list(BALANCED24)
    legacy8 = tta_shuffles(TTAConfig(n_views=8))
    assert len(legacy8) == 8 and legacy8 != list(BALANCED8)


def _make_scorer(truth_seq):
    def scorer_fn(shuffled):
        true_rank = tuple(truth_seq.index(v) for v in shuffled)
        return np.array([-float(perm.kendall_tau_distance(r, true_rank))
                         for r in perm.PERMS24])
    return scorer_fn


def test_tta12_roundtrip_preserves_answer():
    images = [10, 20, 30, 40]
    truth_seq = [30, 10, 40, 20]
    true_rank = tuple(truth_seq.index(v) for v in images)
    scores, per_view = tta_scores(images, _make_scorer(truth_seq), TTAConfig(n_views=12))
    assert len(per_view) == 12
    assert int(np.argmax(scores)) == perm.index_of(true_rank)
    for v in per_view:
        assert int(np.argmax(v)) == perm.index_of(true_rank)
