"""DPO 선호쌍 자동 생성 — chosen=정답 순열 토큰, rejected=인접 스와프 오답.

SFT 이후 포스트트레이닝 단계(노션 권장 순서: SFT → DPO → GRPO). 별도 보상모델 없이
라벨에서 선호쌍을 결정적으로 생성한다. score24가 단일 토큰 구조라 TRL DPOTrainer의
멀티턴 생성형 전처리를 쓸 이유가 없어, 소비 측은 train_dpo.py의 자체 구현 단일토큰
DPO Trainer(다음 토큰 로짓에서 chosen/rejected 두 토큰의 log-softmax만 비교)다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .. import perm
from ..data.augment import AugmentConfig, augment_sample
from ..data.sample import Sample
from ..prompting import build_score24_messages


@dataclass(frozen=True)
class DPOPairConfig:
    rejected_per_sample: int = 1      # 인접 스와프 3종 중 몇 개를 쌍으로 만들지
    include_random_rejected: bool = False  # 스와프 외 랜덤 오답도 추가할지
    augment: AugmentConfig = field(default_factory=lambda: AugmentConfig(perm_mode="uniform"))
    video_mode: bool = False
    seed: int = 777


def build_dpo_records(samples: list[Sample], cfg: DPOPairConfig,
                      scorer: Callable[[str, list], "object"] | None = None) -> list[dict]:
    """레코드: {"prompt_messages", "chosen", "rejected", ...}.

    chosen/rejected는 단일 글자(LETTERS24) — score24 SFT와 같은 출력 공간이라
    DPO가 exact match를 직접 최적화하는 방향과 정렬된다.

    scorer(caption, images) -> scores24(len 24, PERMS24 인덱스 순)를 주면 인접
    스와프 3종 중 **현재 모델이 가장 그럴듯하다고 보는 오답**(hard negative, TODO
    §2a 1순위)을 rejected로 고른다. None이면(기본) 3종 중 무작위 선택 — SFT 어댑터
    없이도 파이프라인을 검증할 수 있게 하는 폴백이며 실제 DPO 학습은 scorer를 준다.
    """
    rng = random.Random(cfg.seed)
    out: list[dict] = []
    for s in samples:
        if s.rank is None:
            raise ValueError(f"라벨 없는 샘플: {s.id}")
        aug = augment_sample(s, cfg.augment, rng)
        msgs = build_score24_messages(aug.caption, aug.images, video_mode=cfg.video_mode)
        chosen = perm.letter_of_rank(aug.rank)

        candidates = perm.adjacent_swap_ranks(aug.rank)
        if scorer is not None:
            scores24 = scorer(aug.caption, aug.images)
            candidates = sorted(candidates, key=lambda r: scores24[perm.index_of(r)], reverse=True)
        else:
            rng.shuffle(candidates)
        rejected_ranks = candidates[: cfg.rejected_per_sample]
        if cfg.include_random_rejected:
            while True:
                r = perm.random_shuffle(rng)
                if r != aug.rank and r not in rejected_ranks:
                    rejected_ranks.append(r)
                    break
        for rj in rejected_ranks:
            out.append({
                "sample_id": s.id,
                "prompt_messages": msgs,
                "chosen": chosen,
                "rejected": perm.letter_of_rank(rj),
                "chosen_rank": aug.rank,
                "rejected_rank": rj,
            })
    return out
