"""TTA (Test-Time Augmentation) — 입력 순서 셔플 후 점수 리맵·집계.

허용 기법(대회 규정 명시). 핵심 함정은 리맵: 셔플 공간의 24점수 벡터를
원본 공간으로 되돌릴 때 perm.unshuffle_rank 규약을 정확히 따라야 한다.
여기가 틀리면 TTA는 성능을 '올리는 척하며' 파괴한다 → property 테스트 필수.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .. import perm


@dataclass(frozen=True)
class TTAConfig:
    n_views: int = 1              # 1이면 TTA 없음(원본 1회)
    seed: int = 1234              # 뷰 셔플 추출 시드(전 샘플 공통 → 재현 가능)
    agg: str = "mean"             # "mean"(로그확률 평균) | "majority"(다수결)


def tta_shuffles(cfg: TTAConfig) -> list[perm.Perm]:
    """항등 + (n_views-1)개의 서로 다른 랜덤 셔플. 결정적."""
    views: list[perm.Perm] = [perm.IDENTITY]
    rng = random.Random(cfg.seed)
    while len(views) < min(cfg.n_views, 24):
        s = perm.random_shuffle(rng)
        if s not in views:
            views.append(s)
    return views


def remap_scores_from_shuffled(scores_shuffled: np.ndarray, s: Sequence[int]) -> np.ndarray:
    """셔플 공간 점수벡터(24,) → 원본 공간. out[idx(r)] = in[idx(r∘s)].

    유도: 원본 rank r 가설은 셔플 뷰에서 rank r∘s로 보인다(perm.shuffled_rank).
    """
    out = np.empty_like(scores_shuffled)
    for i_orig, r in enumerate(perm.PERMS24):
        out[i_orig] = scores_shuffled[perm.index_of(perm.shuffled_rank(r, s))]
    return out


def log_normalize(scores: np.ndarray) -> np.ndarray:
    """점수 → log-softmax (뷰 간 스케일 정합용)."""
    x = scores.astype(np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def aggregate(view_scores: list[np.ndarray], agg: str = "mean") -> np.ndarray:
    """뷰별 (원본 공간으로 리맵된) 점수 → 최종 24점수."""
    mat = np.stack([log_normalize(v) for v in view_scores])
    if agg == "mean":
        return mat.mean(axis=0)
    if agg == "majority":
        votes = np.zeros(24)
        for row in mat:
            votes[int(row.argmax())] += 1
        # 동률은 로그확률 평균으로 타이브레이크
        return votes * 1e6 + mat.mean(axis=0)
    raise ValueError(f"agg: {agg}")


def tta_scores(images: Sequence,
               scorer_fn: Callable[[list], np.ndarray],
               cfg: TTAConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    """scorer_fn(셔플된 이미지 4장) → 셔플 공간 점수(24,). 리맵·집계까지 수행.

    반환: (최종 점수(24,), 뷰별 원본공간 점수 목록)
    """
    per_view: list[np.ndarray] = []
    for s in tta_shuffles(cfg):
        shuffled = perm.apply_shuffle(list(images), s)
        raw = np.asarray(scorer_fn(shuffled), dtype=np.float64)
        if raw.shape != (24,):
            raise ValueError(f"scorer_fn은 (24,)를 반환해야 함: {raw.shape}")
        per_view.append(remap_scores_from_shuffled(raw, s))
    return aggregate(per_view, cfg.agg), per_view
