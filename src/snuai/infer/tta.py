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
    balanced8: bool = False       # n_views==8 을 균형 2-코셋 세트로 대체(아래 BALANCED8)


# Klein 4원군 (Ver11에서 이식, 2026-07-20) — sharply transitive 세트: 4뷰에 걸쳐
# 각 입력이 각 슬롯을 정확히 1회씩 방문(위치에 대한 라틴방진)해 슬롯-위치 편향이
# 기대값이 아니라 정확히 상쇄된다. 항등 외 원소는 모두 고정점 없는 대합.
BALANCED4: tuple[perm.Perm, ...] = ((0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0))

# 균형 8뷰 (2026-07-21) — Klein V 에 V의 코셋 하나를 더한 8뷰: 각 입력이 각 슬롯을
# 정확히 2회씩 방문해 TTA4의 정확 상쇄 성질을 유지한 채 뷰 분산만 절반으로 줄인다
# (TTA5~8의 "항등+랜덤" 세트는 이 성질이 없음 — tta_report_2026-07-20.md §1).
# 두 번째 블록은 역원에 닫혀 있어(4-cycle 쌍 + 대합 2개) 셔플 방향 규약과 무관하게
# 균형이 성립한다. 참고: 어떤 균형 8뷰 세트든 고정점 총합은 정확히 8로 동일 —
# 이 선택이 고정점 면에서 손해보지 않는다.
# 실측 확인(2026-07-22): BALANCED8은 임의의 8개 셔플 묶음이 아니라 S4의 실제
# **부분군**(폐쇄·역원 보유 — property test로 검증)이다. |S4|=24=8×3이므로 이건
# Sylow 2-부분군(이면군 D4) — 즉 지금까지의 계열은 "Klein 4원군(4차 정규부분군) →
# Sylow-2 부분군(8차)"이고, 이 계열이 대수적으로 완성되는 지점은 **전체군 S4(24차)
# 그 자체**뿐이다(다음 크기의 부분군은 즉시 24차로 건너뛴다 — 12차 A4는 부분군이지만
# 지금 계열의 자연스러운 연장선이 아니라 별도 구조). TTA4→TTA8 실LB +1.22pp(사전
# 추정보다 큼, `../CLAUDE.md` 공통 함정 참고)가 "뷰 개수"가 아니라 "대수적 완전성"
# 때문이라면, 다음 실측 대상은 BALANCED24(아래) — 24뷰라 819건 기준 ~6.5h 추정
# (TTA8 실측 ~9.6s/샘플의 24/8배), 24h 예산에 여유 있음.
BALANCED8: tuple[perm.Perm, ...] = BALANCED4 + (
    (1, 2, 3, 0), (0, 3, 2, 1), (3, 0, 1, 2), (2, 1, 0, 3))

# 완전균형 24뷰 — S4 전체(2026-07-22). 부분군 계열(V→D4)의 자연스러운 종점: 더 이상
# "균형 부분집합을 고르는" 설계 여지가 없다(전부 다 쓰므로). 각 입력이 각 슬롯을
# 정확히 6회 방문(24/4). perm.PERMS24 자체가 이미 정확히 이 24개 순열이다.
BALANCED24: tuple[perm.Perm, ...] = tuple(perm.PERMS24)


def tta_shuffles(cfg: TTAConfig) -> list[perm.Perm]:
    """항등 + (n_views-1)개의 서로 다른 랜덤 셔플. 결정적.

    예외: n_views==4 는 균형 Klein 세트(위 BALANCED4)를 반환 — Ver11과 동일 규약.
    예외: n_views==24 는 무조건 완전균형 BALANCED24(=S4 전체)를 반환 — n=24를 과거에
      쓴 적이 없어 재현성 우려가 없으므로 n=4와 같은 방식으로 무조건 하이재킹.
    예외: balanced8=True 는 n_views==8에서 BALANCED8 반환(예산 축소로 n_views==1이
      되면 항등만 — 기존 축소 경로와 동일 동작). 그 외 n_views와의 조합은 에러.
    다른 n은 기존 시드셔플 경로와 바이트 동일(TTA3 재현성 보존).
    """
    if cfg.n_views == 24:
        return list(BALANCED24)
    if cfg.balanced8:
        if cfg.n_views == 8:
            return list(BALANCED8)
        if cfg.n_views == 1:  # BudgetGuard 뷰 축소 경로
            return [perm.IDENTITY]
        raise ValueError(f"balanced8은 n_views 8(또는 축소된 1)에서만 유효: {cfg.n_views}")
    if cfg.n_views == 4:
        return list(BALANCED4)
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
