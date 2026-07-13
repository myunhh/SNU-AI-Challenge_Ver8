"""perm 규약 방어선 — rank/order 혼동(50.5% 함정)은 조용히 절반만 틀려서 티가 안 난다."""

import random

from snuai import perm


def test_perms24_basics():
    assert perm.PERMS24[0] == (0, 1, 2, 3)          # A = 항등
    assert perm.PERMS24[23] == (3, 2, 1, 0)         # X = 완전 역순
    assert len(set(perm.PERMS24)) == 24
    assert perm.LETTERS24 == "ABCDEFGHIJKLMNOPQRSTUVWX"


def test_worked_example_rank_class_letter():
    # 실제 시간순 fA→fB→fC→fD, 입력 [fC, fA, fD, fB]:
    # 입력1=fC는 3번째, 입력2=fA는 1번째, 입력3=fD는 4번째, 입력4=fB는 2번째
    rank0 = perm.from_1based([3, 1, 4, 2])
    assert rank0 == (2, 0, 3, 1)
    assert perm.index_of(rank0) == 13
    assert perm.letter_of_rank(rank0) == "N"
    # order는 rank의 역순열 (여기서 rank는 자기역원이 아님 — 혼동이 드러나는 케이스)
    order = perm.rank_to_order(rank0)
    assert order == (1, 3, 0, 2)
    assert order != rank0
    assert perm.order_to_rank(order) == rank0


def test_shuffle_is_gather_not_argsort():
    # new_rank[j] = rank[s[j]] (gather). argsort류 구현은 non-involution에서 어긋난다.
    r = (0, 2, 1, 3)
    s = (1, 0, 3, 2)
    assert perm.shuffled_rank(r, s) == (2, 0, 3, 1)
    # 물리 셔플과 라벨 재계산의 정합: 새 슬롯 j의 이미지 = 옛 슬롯 s[j]의 이미지
    items = ["x0", "x1", "x2", "x3"]
    shuffled = perm.apply_shuffle(items, s)
    assert shuffled == ["x1", "x0", "x3", "x2"]
    new_rank = perm.shuffled_rank(r, s)
    for j in range(4):
        old_slot = items.index(shuffled[j])
        assert new_rank[j] == r[old_slot]


def test_shuffle_roundtrip_random():
    rng = random.Random(0)
    for _ in range(100):
        r = perm.random_shuffle(rng)
        s = perm.random_shuffle(rng)
        assert perm.unshuffle_rank(perm.shuffled_rank(r, s), s) == r


def test_compose_inverse_laws():
    rng = random.Random(1)
    for _ in range(50):
        p = perm.random_shuffle(rng)
        assert perm.compose(p, perm.inverse(p)) == perm.IDENTITY
        assert perm.compose(perm.inverse(p), p) == perm.IDENTITY


def test_adjacent_swap_ranks_distance_one():
    rng = random.Random(2)
    for _ in range(20):
        r = perm.random_shuffle(rng)
        swaps = perm.adjacent_swap_ranks(r)
        assert len(swaps) == 3
        for w in swaps:
            assert perm.kendall_tau_distance(r, w) == 1


def test_partial_credit_scores():
    # 완전 일치 = 1.0, 완전 역순 = 쌍순서 0.0
    assert perm.pairwise_score((0, 1, 2, 3), (0, 1, 2, 3)) == 1.0
    assert perm.position_score((0, 1, 2, 3), (0, 1, 2, 3)) == 1.0
    assert perm.pairwise_score((0, 1, 2, 3), (3, 2, 1, 0)) == 0.0
    assert perm.position_score((0, 1, 2, 3), (3, 2, 1, 0)) == 0.0
    # 인접 스와프 1개: 쌍 1/6 어긋남, 위치 2/4 어긋남
    assert perm.pairwise_score((0, 1, 2, 3), (1, 0, 2, 3)) == 1 - 1 / 6
    assert perm.position_score((0, 1, 2, 3), (1, 0, 2, 3)) == 0.5
    # 대칭성
    rng = random.Random(3)
    for _ in range(20):
        a, b = perm.random_shuffle(rng), perm.random_shuffle(rng)
        assert perm.pairwise_score(a, b) == perm.pairwise_score(b, a)
        assert perm.position_score(a, b) == perm.position_score(b, a)


def test_soft_target_distribution_is_simplex_peaked_at_truth():
    true_rank = (2, 0, 3, 1)
    dist = perm.soft_target_distribution(true_rank, temperature=0.02)
    assert len(dist) == 24
    assert abs(sum(dist) - 1.0) < 1e-9
    assert all(p >= 0 for p in dist)
    # 낮은 온도에서는 true_rank 자신의 확률이 압도적으로 커야 함(유일한 최댓값)
    assert dist[perm.index_of(true_rank)] == max(dist)
    assert dist[perm.index_of(true_rank)] > 0.9


def test_soft_target_distribution_uniform_at_high_temperature():
    true_rank = (2, 0, 3, 1)
    dist = perm.soft_target_distribution(true_rank, temperature=1000.0)
    assert all(abs(p - 1 / 24) < 1e-3 for p in dist)


def test_soft_target_distribution_rejects_nonpositive_temperature():
    import pytest
    with pytest.raises(ValueError):
        perm.soft_target_distribution((0, 1, 2, 3), temperature=0.0)
