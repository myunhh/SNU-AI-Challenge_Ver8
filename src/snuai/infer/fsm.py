"""제약 디코딩 (FSM) — 생성 모드에서 파싱 실패를 구조적으로 0으로.

listwise CoT의 답 구간 "FINAL ANSWER: [" 이후를 유한상태기계로 강제:
  d₁ , d₂ , d₃ , d₄ ]   (dᵢ ∈ 1..4, 이미 쓴 숫자 금지)

구성:
  PermStepper                 순수 FSM (토크나이저 무관, 완전 단위테스트 가능)
  PermutationLogitsProcessor  transformers generate()용 (프롬프트가 ANSWER_PREFIX로
                              끝나는 상태에서 시작; 단일 문자 토큰만 허용하도록 마스킹)
  constrained_perm_generate   수동 8-스텝 루프 (가장 단순·강건, 기본 경로)
"""

from __future__ import annotations

from typing import Callable, Sequence

from .. import perm

DIGITS = ("1", "2", "3", "4")
COMMA = ","
CLOSE = "]"
SYMBOLS = (*DIGITS, COMMA, CLOSE)


class PermStepper:
    """답 구간 심볼 FSM. 심볼: '1'..'4', ',', ']'. push()가 유효성 강제 + 기록."""

    def __init__(self) -> None:
        self.history: list[str] = []
        self.used: set[str] = set()
        self.n_digits = 0
        self.expect_digit = True  # digit ↔ 구분자 교대
        self.done = False

    def allowed(self) -> tuple[str, ...]:
        if self.done:
            return ()
        if self.expect_digit:
            return tuple(d for d in DIGITS if d not in self.used)
        return (CLOSE,) if self.n_digits == perm.N else (COMMA,)

    def push(self, sym: str) -> None:
        if sym not in self.allowed():
            raise ValueError(f"FSM 위반: {sym!r} (허용: {self.allowed()})")
        self.history.append(sym)
        if sym in DIGITS:
            self.used.add(sym)
            self.n_digits += 1
            self.expect_digit = False
        elif sym == COMMA:
            self.expect_digit = True
        elif sym == CLOSE:
            self.done = True

    def collected_rank(self) -> perm.Perm:
        """push된 숫자들 → 0-based rank 튜플 (Answer 인코딩과 동일)."""
        if self.n_digits != perm.N:
            raise ValueError("숫자 4개가 아직 다 나오지 않음")
        return perm.from_1based(int(d) for d in self.history if d in DIGITS)


def constrained_perm_generate(next_symbol_scores: Callable[[list[str]], dict[str, float]],
                              ) -> perm.Perm:
    """수동 제약 생성 — next_symbol_scores(지금까지 심볼)→{심볼:점수}만 있으면 동작.

    허용 심볼 중 최고 점수를 선택. 항상 유효한 순열 반환(파싱 실패 원천 차단).
    """
    st = PermStepper()
    while not st.done:
        allowed = st.allowed()
        scores = next_symbol_scores(list(st.history))
        st.push(max(allowed, key=lambda a: scores.get(a, float("-inf"))))
    return st.collected_rank()


class PermutationLogitsProcessor:
    """transformers LogitsProcessor — 프롬프트가 ANSWER_PREFIX('['까지)로 끝난 뒤 사용.

    각 시퀀스의 '프롬프트 이후 생성분'을 심볼로 해석해 FSM 허용 토큰만 남긴다.
    심볼은 단일 문자 토큰 id로 고정 — 멀티문자 병합 토큰은 마스킹되어 나올 수 없음.
    """

    def __init__(self, symbol_token_ids: dict[str, int], prompt_len: int):
        missing = [s for s in SYMBOLS if s not in symbol_token_ids]
        if missing:
            raise ValueError(f"토큰 id 누락 심볼: {missing}")
        self.sym2id = {s: symbol_token_ids[s] for s in SYMBOLS}
        self.id2sym = {v: k for k, v in self.sym2id.items()}
        if len(self.id2sym) != len(self.sym2id):
            raise ValueError("심볼 토큰 id 충돌")
        self.prompt_len = prompt_len

    def _stepper_for(self, generated_ids: Sequence[int]) -> PermStepper:
        st = PermStepper()
        for tid in generated_ids:
            sym = self.id2sym.get(int(tid))
            if sym is None:
                raise ValueError(f"FSM 구간에 비심볼 토큰 발생: id={tid}")
            st.push(sym)
        return st

    def __call__(self, input_ids, scores):
        import torch
        for b in range(scores.shape[0]):
            st = self._stepper_for(input_ids[b, self.prompt_len:].tolist())
            allowed_ids = [self.sym2id[s] for s in st.allowed()]
            mask = torch.full_like(scores[b], float("-inf"))
            if allowed_ids:
                mask[allowed_ids] = 0.0
            scores[b] = scores[b] + mask
        return scores


def parse_generated_rank(generated_ids: Sequence[int], id2sym: dict[int, str]) -> perm.Perm:
    """FSM 제약 하에 생성된 토큰열 → rank. (Processor와 세트로 사용)"""
    st = PermStepper()
    for tid in generated_ids:
        sym = id2sym.get(int(tid))
        if sym is None:
            break  # EOS 등
        st.push(sym)
        if st.done:
            break
    return st.collected_rank()
