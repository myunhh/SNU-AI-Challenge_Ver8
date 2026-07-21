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
    # 완전역전(d=6) rejected 추가 — ckpt600 홀드아웃 오답 분석(2026-07-18)에서 확인된
    # "확신에 찬 방향 역전"(d>=4 64건, 완전역전 12건, margin 0.98+)을 직접 벌주는 신호.
    # 인접 스와프(d=1)만으로는 이 오류 모드에 그래디언트가 실리지 않는다.
    # 결정적 생성(rng 미소비)이라 include_random_rejected와 달리 DDP 캐시 경로와 호환.
    include_reversal_rejected: bool = False
    augment: AugmentConfig = field(default_factory=lambda: AugmentConfig(perm_mode="uniform"))
    video_mode: bool = False
    seed: int = 777


def build_dpo_records(samples: list[Sample], cfg: DPOPairConfig,
                      scorer: Callable[[str, list], "object"] | None = None,
                      rejected_ranks_cache: dict[str, list] | None = None) -> list[dict]:
    """레코드: {"prompt_messages", "chosen", "rejected", ...}.

    chosen/rejected는 단일 글자(LETTERS24) — score24 SFT와 같은 출력 공간이라
    DPO가 exact match를 직접 최적화하는 방향과 정렬된다.

    scorer(caption, images) -> scores24(len 24, PERMS24 인덱스 순)를 주면 인접
    스와프 3종 중 **현재 모델이 가장 그럴듯하다고 보는 오답**(hard negative, TODO
    §2a 1순위)을 rejected로 고른다. None이면(기본) 3종 중 무작위 선택 — SFT 어댑터
    없이도 파이프라인을 검증할 수 있게 하는 폴백이며 실제 DPO 학습은 scorer를 준다.

    rejected_ranks_cache가 주어지면 scorer 호출 없이 sample_id별로 미리 계산된
    **최종** rejected_ranks(정렬+슬라이스+include_random_rejected까지 전부 반영된
    결과)를 그대로 쓴다(DDP에서 rank0가 한 번만 scorer를 돌리고 다른 rank들이
    재사용하는 용도 — augment_sample은 seed 고정 rng만 쓰므로 rank 간
    aug.rank/caption/images는 결정적으로 동일하다). include_random_rejected는
    캐시 경로에서 rng를 추가 소비하지 않으므로(캐시가 이미 그 결과를 담고 있음)
    호출자가 rank0/재구성 양쪽에 동일 cfg를 써야 한다 — 호출자(train_dpo.py)가
    캐시+include_random_rejected 동시 사용을 막는다(rng 스트림 분기 위험).
    """
    rng = random.Random(cfg.seed)
    out: list[dict] = []
    iterable = samples
    if scorer is not None:
        # hard-negative 스코어링은 샘플당 forward 1회로 수십 분 걸린다 — 진행바 필수(전역 규약)
        from tqdm import tqdm
        iterable = tqdm(samples, desc="hard-negative 스코어링", unit="샘플", mininterval=5.0)
    for s in iterable:
        if s.rank is None:
            raise ValueError(f"라벨 없는 샘플: {s.id}")
        aug = augment_sample(s, cfg.augment, rng)
        msgs = build_score24_messages(aug.caption, aug.images, video_mode=cfg.video_mode)
        chosen = perm.letter_of_rank(aug.rank)

        if rejected_ranks_cache is not None:
            rejected_ranks = list(rejected_ranks_cache[s.id])
        else:
            candidates = perm.adjacent_swap_ranks(aug.rank)
            if scorer is not None:
                scores24 = scorer(aug.caption, aug.images)
                candidates = sorted(candidates, key=lambda r: scores24[perm.index_of(r)], reverse=True)
            else:
                rng.shuffle(candidates)
            rejected_ranks = candidates[: cfg.rejected_per_sample]
            if cfg.include_reversal_rejected:
                # 랭크 공간 완전역전: r -> (N-1)-r. 항상 d=6이라 chosen/스와프와 충돌 불가.
                # random rejected보다 먼저 넣어 아래 dedup 루프가 역전과의 중복도 걸러준다.
                rejected_ranks.append(tuple(perm.N - 1 - r for r in aug.rank))
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
