"""24점수 스코어러 3종 + pairwise 판독기 + frame-event 정렬기.

모든 스코어러는 같은 인터페이스: scores(caption, images) → np.ndarray(24,)
(인덱스 = perm.PERMS24의 rank 튜플, 값 = 로그점수) → TTA·캐스케이드에 그대로 연결.

  Score24Scorer      단일 forward, 24글자 로짓 (파인튜닝 후 메인 경로, 샘플당 1 forward)
  LikelihoodScorer   후보 순서로 이미지를 배열하고 캡션 우도 측정 (zero-shot, 24 forwards)
  MatchScorer        캡션 분해 + 프레임×이벤트 정렬 행렬 (4×k forwards, 해석가능)
"""

from __future__ import annotations

import numpy as np

from .. import perm
from ..data.decompose import split_events_rule
from ..prompting import (_txt, build_frame_event_messages, build_pairwise_messages,
                         build_score24_messages, media_content)
from .match import scores24_from_matrix


class Score24Scorer:
    """단일 토큰 24-way — forward 1번으로 24개 순열 로그확률 동시 획득."""

    def __init__(self, engine, video_mode: bool = False, counterfactual: bool = False,
                 legend: bool = True, dup_factor: int = 1):
        self.engine = engine
        self.video_mode = video_mode
        self.counterfactual = counterfactual
        self.legend = legend
        self.dup_factor = dup_factor
        self.letter_ids = [engine.token_id_of(ch) for ch in perm.LETTERS24]
        if len(set(self.letter_ids)) != 24:
            raise ValueError("라벨 토큰 id 충돌 — 토크나이저 확인 필요")

    def scores(self, caption: str, images: list) -> np.ndarray:
        msgs = build_score24_messages(caption, images, video_mode=self.video_mode,
                                      counterfactual=self.counterfactual, legend=self.legend,
                                      dup_factor=self.dup_factor)
        return self.engine.restricted_logprobs(msgs, self.letter_ids)


class LikelihoodScorer:
    """우도 스코어링(zero-shot) — '이 배열이 맞다면 캡션이 나올 확률'을 24번 측정.

    파인튜닝 없이 동작 → B2 베이스라인·전처리 A/B의 주력.
    캡션 토큰열이 24후보에서 동일하므로 길이 정규화 불필요(상대 비교만 유효).
    """

    PREFIX = ("Here are four frames of a video, shown in chronological order.")
    ASK = "\nWrite one sentence describing the storyline of this video."

    def __init__(self, engine, video_mode: bool = False):
        self.engine = engine
        self.video_mode = video_mode

    def _messages(self, ordered_images: list) -> list[dict]:
        return [{"role": "user", "content": [
            _txt(self.PREFIX),
            *media_content(ordered_images, self.video_mode),
            _txt(self.ASK),
        ]}]

    def scores(self, caption: str, images: list) -> np.ndarray:
        out = np.empty(24)
        for idx, rank in enumerate(perm.PERMS24):
            order = perm.rank_to_order(rank)
            ordered = [images[order[p]] for p in range(perm.N)]
            out[idx] = self.engine.continuation_logprob(self._messages(ordered), caption)
        return out


class MatchScorer:
    """decompose-and-match — 캡션을 이벤트로 분해하고 프레임×이벤트 정렬 행렬 채점.

    이벤트 분해는 기본 규칙 기반(비용 0). events_fn 주입으로 모델 분해로 교체 가능.
    중간 산출물(S 행렬, 배정)이 meta로 나와 오답 분석·보고서에 사용.
    """

    def __init__(self, engine, events_fn=None):
        self.engine = engine
        self.events_fn = events_fn or split_events_rule
        self.yes_id = engine.token_id_of("Yes")
        self.no_id = engine.token_id_of("No")
        self.last_matrix: np.ndarray | None = None
        self.last_events: list[str] | None = None

    def alignment_matrix(self, caption: str, images: list) -> tuple[np.ndarray, list[str]]:
        events = self.events_fn(caption)
        S = np.empty((perm.N, len(events)))
        for f, img in enumerate(images):
            for e, ev in enumerate(events):
                lp = self.engine.restricted_logprobs(
                    build_frame_event_messages(img, ev), [self.yes_id, self.no_id])
                S[f, e] = lp[0]  # log P(Yes | frame, event)
        return S, events

    def scores(self, caption: str, images: list) -> np.ndarray:
        S, events = self.alignment_matrix(caption, images)
        self.last_matrix, self.last_events = S, events
        return scores24_from_matrix(S)


class CoTFSMScorer:
    """listwise CoT 생성 + FSM 제약 답 구간 (+ Self-Consistency 샘플링).

    2단계 생성:
      A) 자유 CoT — "FINAL ANSWER: [" 에서 정지
      B) FSM 제약 8토큰 (d,d,d,d + 콤마3 + ']') — 유효 순열만 생성 가능

    n_samples>1 이면 온도 샘플링으로 여러 번 돌려 다수결(Laplace 평활 로그 득표).
    ⚠️ 속도: 샘플당 CoT 길이 × n_samples — budget.allow로 게이팅할 것.
    """

    def __init__(self, engine, n_samples: int = 1, temperature: float = 0.7,
                 cot_max_tokens: int = 256, video_mode: bool = False,
                 counterfactual: bool = True, dup_factor: int = 1):
        from ..prompting import ANSWER_PREFIX
        from .fsm import SYMBOLS
        self.engine = engine
        self.n_samples = n_samples
        self.temperature = temperature
        self.cot_max_tokens = cot_max_tokens
        self.video_mode = video_mode
        self.counterfactual = counterfactual
        self.dup_factor = dup_factor
        self.prefix = ANSWER_PREFIX
        self.sym_ids = {s: engine.token_id_of(s) for s in SYMBOLS}
        self.id2sym = {v: k for k, v in self.sym_ids.items()}

    def _one_rank(self, caption: str, images: list, do_sample: bool) -> perm.Perm:
        from ..prompting import build_listwise_cot_messages, extract_media
        from .fsm import PermutationLogitsProcessor, parse_generated_rank
        msgs = build_listwise_cot_messages(caption, images, video_mode=self.video_mode,
                                           counterfactual=self.counterfactual,
                                           dup_factor=self.dup_factor)
        cot_text, _, _ = self.engine.generate_text(
            msgs, max_new_tokens=self.cot_max_tokens, do_sample=do_sample,
            temperature=self.temperature, stop_strings=[self.prefix])
        # 답 구간 직전까지 자르고 prefix를 강제 부착 (없으면 붙이고, 있으면 그 지점까지)
        cut = cot_text.find(self.prefix)
        cot_head = cot_text[:cut] if cut >= 0 else cot_text
        full_text = self.engine.raw_text_of(msgs) + cot_head + self.prefix

        imgs, vids = extract_media(msgs)
        # 제약 8토큰: prompt_len은 생성 시작 시점 길이 — 프로세서가 내부에서 재토크나이즈
        # (call_processor 경유 필수: 타임스탬프 토큰 유무가 generate 쪽과 갈리면 FSM 오프셋 붕괴)
        from ..prompting import call_processor
        inputs_len_probe = call_processor(
            self.engine.processor, [full_text], imgs, vids,
            return_tensors="pt")["input_ids"].shape[1]
        proc = PermutationLogitsProcessor(self.sym_ids, prompt_len=inputs_len_probe)
        _, gen_ids, _ = self.engine.generate_from_text(
            full_text, images=imgs, videos=vids, max_new_tokens=8,
            logits_processor=[proc])
        return parse_generated_rank(gen_ids[0].tolist(), self.id2sym)

    def scores(self, caption: str, images: list) -> np.ndarray:
        votes = np.zeros(24)
        for k in range(self.n_samples):
            r = self._one_rank(caption, images, do_sample=(self.n_samples > 1))
            votes[perm.index_of(r)] += 1
        alpha = 0.1  # Laplace 평활 — 다수결 유지 + 유한 로그점수
        p = (votes + alpha) / (votes.sum() + 24 * alpha)
        return np.log(p)


class PairwiseJudge:
    """A/B 단일 토큰 로짓으로 '어느 이미지가 먼저인가' 판독. 좌우 플립 2회 평균."""

    def __init__(self, engine, flip_average: bool = True):
        self.engine = engine
        self.flip_average = flip_average
        self.ab_ids = [engine.token_id_of("A"), engine.token_id_of("B")]

    def p_earlier(self, caption: str, img_i, img_j) -> float:
        """P(img_i 가 img_j 보다 시간상 앞)."""
        lp1 = self.engine.restricted_logprobs(
            build_pairwise_messages(caption, img_i, img_j), self.ab_ids)
        p = float(np.exp(lp1[0]))  # i가 A 자리 → P(A)
        if self.flip_average:
            lp2 = self.engine.restricted_logprobs(
                build_pairwise_messages(caption, img_j, img_i), self.ab_ids)
            p = 0.5 * (p + float(np.exp(lp2[1])))  # 플립: i가 B 자리 → P(B)
        return p

    def make_fn(self, caption: str, images: list):
        """cascade.run_cascade용 pairwise_fn(i, j) 클로저."""
        return lambda i, j: self.p_earlier(caption, images[i], images[j])
