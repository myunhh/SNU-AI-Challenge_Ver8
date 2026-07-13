"""GRPO용 검증가능 보상 (verifiable reward) — exact match가 완벽한 보상 함수.

스트레치 옵션(노션 우선순위: SFT → DPO → GRPO). TRL GRPOTrainer의
reward_funcs에 그대로 넣을 수 있는 시그니처로 제공.
"""

from __future__ import annotations

import re

from .. import perm

_LETTER_RE = re.compile(r"\b([A-X])\b")
_RANK_RE = re.compile(r"\[?\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*\]?")


def parse_rank_from_text(text: str) -> perm.Perm | None:
    """생성 텍스트에서 rank 추출 — 숫자 리스트 우선, 없으면 단일 글자 코드."""
    m = _RANK_RE.search(text)
    if m:
        try:
            return perm.from_1based(int(g) for g in m.groups())
        except ValueError:
            pass
    m = _LETTER_RE.search(text.strip())
    if m:
        return perm.rank_of_letter(m.group(1))
    return None


def exact_match_reward(completion: str, true_rank: perm.Perm) -> float:
    """정답이면 1.0, 오답·파싱불가 0.0 — GRPO 기본 보상."""
    got = parse_rank_from_text(completion)
    return 1.0 if got == true_rank else 0.0


def shaped_reward(completion: str, true_rank: perm.Perm, shaping: float = 0.3) -> float:
    """선택: 켄달 타우 기반 부분 보상 (exact 1.0 + 근접도 가중).

    reward = exact + shaping * (1 - kendall/6) * (1 - exact)
    희소 보상으로 GRPO가 안 뜰 때만 켤 것 — 채점은 exact match임을 잊지 말 것.
    """
    got = parse_rank_from_text(completion)
    if got is None:
        return 0.0
    if got == true_rank:
        return 1.0
    prox = 1.0 - perm.kendall_tau_distance(got, true_rank) / 6.0
    return shaping * prox
