"""DPO 선호쌍 생성 (TODO §2a) — 전부 CPU. 인접 스와프 hard negative 선택 검증."""

import numpy as np
import pytest
from PIL import Image

from snuai import perm
from snuai.data.augment import AugmentConfig
from snuai.data.sample import Sample
from snuai.train.dpo_pairs import DPOPairConfig, build_dpo_records

# perm_mode="off"로 셔플 증강을 꺼서 aug.rank == 입력 rank를 보장(hard-negative 로직만 검증).
_NO_SHUFFLE = AugmentConfig(perm_mode="off")


def _sample(rank):
    images = [Image.new("RGB", (8, 8), color=(k, 0, 0)) for k in range(4)]
    return Sample(id="s1", caption="cap", images=images, rank=rank)


def test_rejected_is_always_an_adjacent_swap():
    rank = (2, 0, 3, 1)
    recs = build_dpo_records([_sample(rank)], DPOPairConfig(rejected_per_sample=1, augment=_NO_SHUFFLE))
    assert len(recs) == 1
    r = recs[0]
    assert r["chosen"] == perm.letter_of_rank(rank)
    assert r["rejected"] != r["chosen"]
    assert perm.kendall_tau_distance(perm.rank_of_letter(r["rejected"]), rank) == 1


def test_multiple_rejected_per_sample_are_distinct():
    rank = (2, 0, 3, 1)
    recs = build_dpo_records([_sample(rank)], DPOPairConfig(rejected_per_sample=3, augment=_NO_SHUFFLE))
    assert len(recs) == 3
    rejected_set = {r["rejected"] for r in recs}
    assert len(rejected_set) == 3   # 인접 스와프 3종 전부 소진, 중복 없음


def test_scorer_picks_hard_negative_not_random():
    rank = (2, 0, 3, 1)
    candidates = perm.adjacent_swap_ranks(rank)
    target = candidates[1]   # 이 후보를 모델이 가장 그럴듯하다고 보게 만든다

    def fake_scorer(caption, images):
        scores = np.full(24, -10.0)
        scores[perm.index_of(target)] = 5.0        # hard negative
        scores[perm.index_of(rank)] = 10.0          # 정답은 항상 최고점(스코어러 현실성)
        return scores

    recs = build_dpo_records([_sample(rank)], DPOPairConfig(rejected_per_sample=1, augment=_NO_SHUFFLE),
                             scorer=fake_scorer)
    assert len(recs) == 1
    assert recs[0]["rejected"] == perm.letter_of_rank(target)


def test_scorer_orders_top_k_by_score_descending():
    rank = (2, 0, 3, 1)
    candidates = perm.adjacent_swap_ranks(rank)

    def fake_scorer(caption, images):
        scores = np.zeros(24)
        for i, c in enumerate(candidates):
            scores[perm.index_of(c)] = float(i)   # candidates[2] 최고점, [0] 최저점
        scores[perm.index_of(rank)] = 100.0
        return scores

    recs = build_dpo_records([_sample(rank)], DPOPairConfig(rejected_per_sample=2, augment=_NO_SHUFFLE),
                             scorer=fake_scorer)
    rejected_ranks = [perm.rank_of_letter(r["rejected"]) for r in recs]
    assert rejected_ranks[0] == candidates[2]   # 1등 hard negative
    assert rejected_ranks[1] == candidates[1]   # 2등


def test_include_random_rejected_adds_extra_distinct_record():
    rank = (2, 0, 3, 1)
    recs = build_dpo_records(
        [_sample(rank)],
        DPOPairConfig(rejected_per_sample=1, include_random_rejected=True, augment=_NO_SHUFFLE))
    assert len(recs) == 2
    ranks = [perm.rank_of_letter(r["rejected"]) for r in recs]
    assert len(set(ranks)) == 2
    assert rank not in ranks


def test_unlabeled_sample_raises():
    images = [Image.new("RGB", (8, 8)) for _ in range(4)]
    unlabeled = Sample(id="s2", caption="c", images=images, rank=None)
    with pytest.raises(ValueError):
        build_dpo_records([unlabeled], DPOPairConfig(augment=_NO_SHUFFLE))
