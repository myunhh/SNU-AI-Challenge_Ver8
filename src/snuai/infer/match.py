"""Decompose-and-Match — 프레임×이벤트 정렬 행렬에서 최적 순서 계산.

입력: S (4×k) — S[f, e] = "프레임 f가 이벤트 e를 보여준다" 로그점수
      (이벤트는 캡션 분해 결과로 시간순 정렬되어 있음, k는 4가 아닐 수 있음)
계산: 24개 순서 각각에 대해 '단조 정렬 점수'를 매겨 24점수 벡터 생성
      → score24와 같은 인터페이스(rank 인덱스 공간)라 캐스케이드·TTA에 그대로 연결.

단조 정렬: 시간순 p번째 프레임이 이벤트 e_p에 배정될 때 e_1 ≤ e_2 ≤ e_3 ≤ e_4
(연속 프레임이 같은 이벤트를 보여줄 수 있고, 이벤트를 건너뛸 수도 있음)
→ 순서당 O(4k) DP. 24개 순서 전부 돌아도 사실상 공짜.
"""

from __future__ import annotations

import numpy as np

from .. import perm


def monotone_alignment(S: np.ndarray, order: perm.Perm) -> tuple[float, list[int]]:
    """주어진 시간 순서(order)로 프레임을 이벤트에 단조 배정하는 최적 점수와 배정.

    반환: (최대 점수 합, 시간순 p번째 프레임의 이벤트 인덱스 목록)
    """
    n, k = S.shape
    assert n == perm.N
    # dp[p, e] = 시간순 p번째 프레임을 이벤트 e에 배정했을 때 최대 누적 점수
    dp = np.full((n, k), -np.inf)
    back = np.zeros((n, k), dtype=int)
    dp[0] = S[order[0]]
    for p in range(1, n):
        # prefix max: max_{e' <= e} dp[p-1, e']
        prefix = np.maximum.accumulate(dp[p - 1])
        argmax_prefix = np.zeros(k, dtype=int)
        best = 0
        for e in range(k):
            if dp[p - 1, e] > dp[p - 1, best]:
                best = e
            argmax_prefix[e] = best
        dp[p] = S[order[p]] + prefix
        back[p] = argmax_prefix
    e_last = int(dp[-1].argmax())
    assign = [e_last]
    for p in range(n - 1, 0, -1):
        e_last = int(back[p][e_last])
        assign.append(e_last)
    assign.reverse()
    return float(dp[-1].max()), assign


def scores24_from_matrix(S: np.ndarray) -> np.ndarray:
    """4×k 정렬 행렬 → 24점수 벡터 (인덱스 = PERMS24의 rank 튜플)."""
    S = np.asarray(S, dtype=np.float64)
    if S.shape[0] != perm.N or S.shape[1] < 1:
        raise ValueError(f"S는 (4, k>=1) 이어야 함: {S.shape}")
    out = np.empty(24)
    for idx, rank in enumerate(perm.PERMS24):
        order = perm.rank_to_order(rank)
        out[idx], _ = monotone_alignment(S, order)
    return out


def best_rank_from_matrix(S: np.ndarray) -> tuple[perm.Perm, np.ndarray, list[int]]:
    """최적 rank + 24점수 + (보고서용) 시간순 프레임→이벤트 배정."""
    scores = scores24_from_matrix(S)
    rank = perm.perm_at(int(scores.argmax()))
    _, assign = monotone_alignment(np.asarray(S, dtype=np.float64), perm.rank_to_order(rank))
    return rank, scores, assign
