"""순열 대수 — 인코딩 규약의 단일 진실 공급원 (Single Source of Truth).

이 모듈의 규약을 어기는 코드는 조용히 점수를 0으로 만든다. 반드시 이 모듈의
함수만 사용하고, 순열 연산을 직접 구현하지 말 것.

용어 (전부 0-based, 제출 경계에서만 1-based 변환):

  rank  (순위 인코딩) r : r[i] = 입력 슬롯 i 가 시간순으로 몇 번째인가.
                          ★ Kaggle Answer 컬럼의 의미와 동일 (1-based로 변환만 하면 됨).
                          ★ 이 코드베이스의 "정답 공간(canonical space)". score24의 24개
                            클래스 인덱스, 데이터셋 라벨, 캐스케이드 점수 벡터가 전부
                            rank 튜플의 PERMS24 인덱스를 쓴다.
  order (순서 인코딩) o : o[p] = 시간순 p번째에 오는 입력 슬롯. rank의 역순열.
                          (pairwise "누가 먼저?", decompose-and-match 내부 계산용)

  r = inverse(o), o = inverse(r).

예시: 입력 4장의 실제 시간 순서가 I2, I0, I3, I1 이면
  order = (2, 0, 3, 1)   # 첫 번째로 I2, 두 번째로 I0, ...
  rank  = (1, 3, 0, 2)   # I0은 2번째(0-based 1), I1은 4번째, I2는 1번째, I3은 3번째
  Kaggle Answer = "[2,4,1,3]"  # rank + 1

이미지 셔플 (증강과 TTA는 물리적으로 같은 연산):
  셔플 s 적용 = 새 슬롯 j에 원래 입력 s[j]를 배치  (new_images[j] = old_images[s[j]])
  → 새 rank  r' = r ∘ s          (shuffled_rank)
  → 원복      r  = r' ∘ s⁻¹      (unshuffle_rank)
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Iterable, Sequence

N = 4
Perm = tuple[int, int, int, int]

#: 24개 순열의 사전식 정렬 — 클래스 인덱스 공간. 절대 순서 변경 금지(라벨 호환성).
PERMS24: list[Perm] = [tuple(p) for p in itertools.permutations(range(N))]
PERM_INDEX: dict[Perm, int] = {p: i for i, p in enumerate(PERMS24)}

#: score24 단일 토큰 라벨. index i ↔ LETTERS24[i]. 24글자 = A..X.
LETTERS24 = "ABCDEFGHIJKLMNOPQRSTUVWX"
assert len(LETTERS24) == len(PERMS24) == 24

IDENTITY: Perm = tuple(range(N))  # (0,1,2,3)


def is_perm(t: Sequence[int]) -> bool:
    """0..3의 순열인지 검사."""
    return len(t) == N and sorted(t) == list(range(N))


def _check(t: Sequence[int], name: str = "perm") -> Perm:
    if not is_perm(t):
        raise ValueError(f"{name}={t!r} 는 0..{N-1}의 순열이 아님")
    return tuple(t)  # type: ignore[return-value]


def compose(a: Sequence[int], b: Sequence[int]) -> Perm:
    """(a ∘ b)[i] = a[b[i]]. 'b를 먼저, a를 나중에' 적용."""
    a = _check(a, "a")
    b = _check(b, "b")
    return tuple(a[b[i]] for i in range(N))  # type: ignore[return-value]


def inverse(p: Sequence[int]) -> Perm:
    """역순열. rank↔order 변환이 곧 역순열이다."""
    p = _check(p)
    inv = [0] * N
    for i, v in enumerate(p):
        inv[v] = i
    return tuple(inv)  # type: ignore[return-value]


# rank↔order 는 서로 역순열 — 가독성을 위한 별칭
rank_to_order = inverse
order_to_rank = inverse


def index_of(p: Sequence[int]) -> int:
    """순열 → PERMS24 인덱스 (0..23)."""
    return PERM_INDEX[_check(p)]


def perm_at(i: int) -> Perm:
    """PERMS24 인덱스 → 순열."""
    return PERMS24[i]


def letter_of_index(i: int) -> str:
    return LETTERS24[i]


def index_of_letter(ch: str) -> int:
    ch = ch.strip().upper()
    idx = LETTERS24.find(ch)
    if idx < 0:
        raise ValueError(f"유효한 라벨 문자가 아님: {ch!r}")
    return idx


def letter_of_rank(rank: Sequence[int]) -> str:
    """rank 튜플 → 학습 라벨 문자 (score24 단일 토큰)."""
    return LETTERS24[index_of(rank)]


def rank_of_letter(ch: str) -> Perm:
    return PERMS24[index_of_letter(ch)]


# ---------------------------------------------------------------------------
# 이미지 셔플 (순열 증강 · TTA 공용) — 물리 연산과 라벨 변환을 한 쌍으로 제공
# ---------------------------------------------------------------------------

def apply_shuffle(items: Sequence, s: Sequence[int]) -> list:
    """new[j] = old[s[j]]. 이미지 리스트에 셔플 s를 적용."""
    s = _check(s, "shuffle")
    return [items[s[j]] for j in range(N)]


def shuffled_rank(rank: Sequence[int], s: Sequence[int]) -> Perm:
    """셔플 s로 이미지를 재배치했을 때의 새 rank 라벨. r' = r ∘ s."""
    return compose(rank, s)


def unshuffle_rank(rank_shuffled: Sequence[int], s: Sequence[int]) -> Perm:
    """셔플 공간의 rank를 원본 공간으로 복원. r = r' ∘ s⁻¹. (TTA 리맵의 핵심)"""
    return compose(rank_shuffled, inverse(s))


def random_shuffle(rng: random.Random) -> Perm:
    """균일 랜덤 셔플."""
    s = list(range(N))
    rng.shuffle(s)
    return tuple(s)  # type: ignore[return-value]


def shuffle_for_target_rank(rank: Sequence[int], target_rank: Sequence[int]) -> Perm:
    """라벨이 target_rank가 되도록 하는 셔플 s를 계산. r∘s = target ⇒ s = r⁻¹∘target.

    (항등순열 비율 제어: target=IDENTITY 로 주면 '시간순으로 정렬된 입력' 샘플 생성)
    """
    return compose(inverse(rank), target_rank)


# ---------------------------------------------------------------------------
# 분석·2단계 재검용 유틸
# ---------------------------------------------------------------------------

def discordant_pairs(r1: Sequence[int], r2: Sequence[int]) -> list[tuple[int, int]]:
    """두 rank 가설에서 상대 순서가 어긋나는 입력 슬롯 쌍 (i, j) 목록.

    캐스케이드 2단계: top1·top2가 대개 인접 스와프 1개 차이 → 보통 1쌍만 반환됨.
    반환된 (i, j)는 'r1 기준으로 i가 j보다 먼저'인 방향으로 정렬.
    """
    r1 = _check(r1, "r1")
    r2 = _check(r2, "r2")
    out = []
    for i in range(N):
        for j in range(i + 1, N):
            a = r1[i] < r1[j]
            b = r2[i] < r2[j]
            if a != b:
                out.append((i, j) if a else (j, i))
    return out


def kendall_tau_distance(r1: Sequence[int], r2: Sequence[int]) -> int:
    """켄달 타우 거리 (어긋난 쌍의 수, 0..6). 오답 분석·보고서용."""
    return len(discordant_pairs(r1, r2))


def position_score(r1: Sequence[int], r2: Sequence[int]) -> float:
    """위치일치 부분점수 — 슬롯별 rank 일치 비율 (0..1). LB 프록시 후보 (TODO_VER8 P0)."""
    r1 = _check(r1, "r1")
    r2 = _check(r2, "r2")
    return sum(a == b for a, b in zip(r1, r2)) / N


def pairwise_score(r1: Sequence[int], r2: Sequence[int]) -> float:
    """쌍순서 부분점수 = 1 − KT거리/6 (0..1).

    LB 스케일과 가장 유사한 프록시(VER7.md LB 섹션: v3 4bit 홀드아웃 0.8351 ↔
    LB 0.82373). EM 게이트가 놓친 LB 이득을 잡기 위한 병행 지표.
    """
    return 1.0 - kendall_tau_distance(r1, r2) / (N * (N - 1) / 2)


def soft_target_distribution(true_rank: Sequence[int], temperature: float) -> list[float]:
    """metric-aligned soft label — pairwise_score 커널로 스무딩한 24클래스 타깃 분포.

    p(q) ∝ exp(pairwise_score(true_rank, q) / T). T→0이면 true_rank에 원핫으로
    수렴(EM SFT와 동일), T→∞면 균등분포. LB가 부분점수(쌍순서) 채점일 때만
    one-hot 대신 쓸 이유가 있다(TODO §2b) — EM 채점이면 이 함수는 불필요.
    """
    if temperature <= 0:
        raise ValueError(f"temperature는 양수여야 함: {temperature}")
    true_rank = _check(true_rank, "true_rank")
    weights = [pairwise_score(true_rank, q) / temperature for q in PERMS24]
    m = max(weights)
    exps = [math.exp(w - m) for w in weights]
    z = sum(exps)
    return [e / z for e in exps]


def adjacent_swap_ranks(rank: Sequence[int]) -> list[Perm]:
    """시간축 기준 인접 스와프 3종의 rank 튜플 (DPO rejected 후보).

    order 공간에서 위치 p, p+1을 교환한 뒤 rank로 되돌린다.
    """
    order = list(rank_to_order(rank))
    out = []
    for p in range(N - 1):
        o2 = order.copy()
        o2[p], o2[p + 1] = o2[p + 1], o2[p]
        out.append(order_to_rank(tuple(o2)))
    return out


# ---------------------------------------------------------------------------
# 1-based 변환 (제출/CoT 텍스트 경계 전용)
# ---------------------------------------------------------------------------

def to_1based(rank: Sequence[int]) -> tuple[int, ...]:
    _check(rank)
    return tuple(v + 1 for v in rank)


def from_1based(rank1: Iterable[int]) -> Perm:
    t = tuple(v - 1 for v in rank1)
    return _check(t, "rank(1-based)")
