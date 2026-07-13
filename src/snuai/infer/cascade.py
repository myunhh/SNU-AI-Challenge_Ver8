"""캐스케이드 — 불확실도(margin) 기반 적응형 2단계 추론.

Stage 1: score24(또는 우도) 24점수 → margin = p(top1) − p(top2)
Stage 2: margin < τ 인 샘플만, top1·top2의 분쟁 pair를 pairwise 로짓으로 재검
융합:   final(π) = log p₁(π) + λ·Σ_pair log p(pair 방향이 π와 일치)

⚠️ τ·λ는 반드시 train 홀드아웃으로만 캘리브레이션(test 참조 = 누수 = 실격 사유).
   단일 모델 반복 호출 = 허용된 추론 전략(TTA/멀티턴 범주), 앙상블 아님.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .. import perm


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(x))


def margin_of(scores24: np.ndarray) -> float:
    """margin = p(top1) − p(top2). softmax 확률 공간 → 샘플 간 비교 가능한 단일 τ."""
    p = np.sort(softmax(scores24))[::-1]
    return float(p[0] - p[1])


def top2_ranks(scores24: np.ndarray) -> tuple[perm.Perm, perm.Perm]:
    i1, i2 = np.argsort(scores24)[::-1][:2]
    return perm.perm_at(int(i1)), perm.perm_at(int(i2))


@dataclass(frozen=True)
class CascadeConfig:
    enable: bool = True
    tau: float = 0.15          # margin 임계값 — 홀드아웃 캘리브레이션으로 교체할 것
    lam: float = 1.0           # pairwise 증거 가중 (0이면 Stage1 단독과 동일 = 성능 하한)
    max_pairs: int = 3         # top1·top2 분쟁 pair 상한 (보통 1개)


@dataclass
class CascadeResult:
    rank: perm.Perm
    escalated: bool
    margin: float
    queried_pairs: list[tuple[int, int]] = field(default_factory=list)
    fused_scores: np.ndarray | None = None


def fuse(scores24: np.ndarray,
         pair_probs: dict[tuple[int, int], float],
         lam: float) -> np.ndarray:
    """Stage1 로그점수 + λ·pairwise 로그증거.

    pair_probs[(i,j)] = P(입력 i가 j보다 시간상 앞) ∈ (0,1).
    각 순열 π에 대해 π가 (i 먼저)면 log p, (j 먼저)면 log(1-p)를 가산.
    """
    base = log_softmax(scores24).copy()
    eps = 1e-9
    for (i, j), p in pair_probs.items():
        p = min(max(float(p), eps), 1.0 - eps)
        for idx, r in enumerate(perm.PERMS24):
            base[idx] += lam * (np.log(p) if r[i] < r[j] else np.log(1.0 - p))
    return base


def run_cascade(scores24: np.ndarray,
                pairwise_fn: Callable[[int, int], float] | None,
                cfg: CascadeConfig) -> CascadeResult:
    """단일 샘플 캐스케이드. pairwise_fn(i, j) → P(i가 j보다 앞). None이면 Stage1 단독."""
    m = margin_of(scores24)
    top1, top2 = top2_ranks(scores24)
    if (not cfg.enable) or pairwise_fn is None or m >= cfg.tau:
        return CascadeResult(rank=top1, escalated=False, margin=m)

    pairs = perm.discordant_pairs(top1, top2)[: cfg.max_pairs]
    pair_probs = {(i, j): pairwise_fn(i, j) for (i, j) in pairs}
    fused = fuse(scores24, pair_probs, cfg.lam)
    final = perm.perm_at(int(fused.argmax()))
    return CascadeResult(rank=final, escalated=True, margin=m,
                         queried_pairs=pairs, fused_scores=fused)


# ---------------------------------------------------------------------------
# 캘리브레이션 (train 홀드아웃 전용)
# ---------------------------------------------------------------------------

def tau_by_escalation_budget(margins: Sequence[float], escalate_frac: float) -> float:
    """홀드아웃 margin 분포에서 '하위 escalate_frac 비율만 2단계로' 보내는 τ."""
    if not 0.0 <= escalate_frac <= 1.0:
        raise ValueError(f"escalate_frac: {escalate_frac}")
    if escalate_frac == 0.0:
        return 0.0
    return float(np.quantile(np.asarray(margins, dtype=np.float64), escalate_frac))


def margin_accuracy_table(margins: Sequence[float], correct: Sequence[bool],
                          n_bins: int = 10) -> list[dict]:
    """margin 구간별 정답률 — τ 선택 근거 + 보고서용 표."""
    m = np.asarray(margins, dtype=np.float64)
    c = np.asarray(correct, dtype=bool)
    qs = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    rows = []
    for k in range(n_bins):
        lo, hi = qs[k], qs[k + 1]
        sel = (m >= lo) & (m <= hi if k == n_bins - 1 else m < hi)
        if sel.sum() == 0:
            continue
        rows.append({"margin_lo": float(lo), "margin_hi": float(hi),
                     "n": int(sel.sum()), "accuracy": float(c[sel].mean())})
    return rows
