"""SFT 데이터셋 — score24 단일 토큰 라벨 + 순열 증강 + 보조 검증 태스크 믹싱.

핵심 설계 (노션 §10 '단일 토큰 24-way'):
  - 라벨 = perm.LETTERS24 한 글자 → 추론은 forward 1번으로 24클래스 로짓
  - 증강은 __getitem__에서 온더플라이 → 에폭·인덱스마다 다른 셔플 (결정적 시드)
  - 보조 태스크(verify_ratio>0): "이 배열이 시간순인가?" Yes/No 샘플을 섞어
    Shuffle & Learn식 순서 감각을 같은 SFT loss로 학습 (별도 loss 코드 불필요)

torch 의존을 collator로 격리 — 데이터셋 로직 자체는 CPU 단위테스트 가능.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import perm
from ..data.augment import AugmentConfig, augment_sample
from ..data.sample import Sample
from ..prompting import build_score24_messages, build_verify_messages


@dataclass(frozen=True)
class SFTDatasetConfig:
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    video_mode: bool = False
    video_dup_factor: int = 1          # Ver7 video_dup(R1 재도전): 프레임 연속 복제 횟수(1/짝수만)
    counterfactual: bool = False       # 학습 프롬프트에도 동일 적용(학습=추론 프롬프트 일치 원칙)
    legend: bool = True                # Ver3: A~X↔순열 범례 명시 (학습=추론 프롬프트 일치)
    verify_ratio: float = 0.0          # 보조 검증 태스크 비율 (0.1~0.2 권장 실험 범위)
    epoch_multiplier: int = 1          # 가상 확장 배수(순열 증강 24배 활용 시 >1)
    seed: int = 20260709


class Score24SFTDataset:
    """torch.utils.data.Dataset 프로토콜(len/getitem) — 반환은 순수 dict.

    반환 형식: {"messages": [...], "target_text": "C", "task": "score24"|"verify",
               "sample_id": ..., "rank": (…)}
    """

    def __init__(self, samples: list[Sample], cfg: SFTDatasetConfig):
        if any(s.rank is None for s in samples):
            raise ValueError("SFT 데이터셋에는 라벨(rank) 있는 샘플만 넣을 것")
        self.samples = samples
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples) * self.cfg.epoch_multiplier

    def _rng_for(self, index: int) -> random.Random:
        # 인덱스(가상 반복 포함)별 결정적 rng → 재현 가능 + 반복마다 다른 증강
        # (프로세스·머신 무관 안정 시드: 튜플/hash() 사용 금지)
        return random.Random(self.cfg.seed * 1_000_003 + index)

    def __getitem__(self, index: int) -> dict:
        base = self.samples[index % len(self.samples)]
        rng = self._rng_for(index)
        aug = augment_sample(base, self.cfg.augment, rng)

        if self.cfg.verify_ratio > 0 and rng.random() < self.cfg.verify_ratio:
            return self._verify_item(aug, rng)

        messages = build_score24_messages(
            aug.caption, aug.images, video_mode=self.cfg.video_mode,
            counterfactual=self.cfg.counterfactual, legend=self.cfg.legend,
            dup_factor=self.cfg.video_dup_factor)
        return {
            "messages": messages,
            "target_text": perm.letter_of_rank(aug.rank),
            "task": "score24",
            "sample_id": base.id,
            "rank": aug.rank,
        }

    def _verify_item(self, aug: Sample, rng: random.Random) -> dict:
        """보조 태스크: 50% 확률로 '시간순 정렬본'(Yes) / '오배열본'(No) 제시."""
        order = perm.rank_to_order(aug.rank)
        if rng.random() < 0.5:
            shown = [aug.images[order[p]] for p in range(perm.N)]  # 시간순 정렬
            target = "Yes"
        else:
            wrong_rank = rng.choice(perm.adjacent_swap_ranks(aug.rank))
            wrong_order = perm.rank_to_order(wrong_rank)
            shown = [aug.images[wrong_order[p]] for p in range(perm.N)]
            target = "No"
        messages = build_verify_messages(aug.caption, shown, video_mode=self.cfg.video_mode,
                                         dup_factor=self.cfg.video_dup_factor)
        return {"messages": messages, "target_text": target, "task": "verify",
                "sample_id": aug.id, "rank": aug.rank}


class SFTCollator:
    """(messages, target_text) 배치 → 모델 입력 + labels.

    라벨 마스킹 규약: 각 행의 '마지막 k 토큰'(= target + EOS)만 학습, 나머지 -100.
    이미지 placeholder가 processor 단계에서 확장되므로 프롬프트 길이를 미리 알 수
    없기 때문에, target을 끝에 붙이고 끝에서부터 마스킹하는 방식이 유일하게 안전.
    검증: 마스킹된 위치의 input_ids가 target 토큰과 일치하는지 assert.
    """

    def __init__(self, processor, train_on_eos: bool = True):
        import torch  # collator만 torch 의존
        self.torch = torch
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.train_on_eos = train_on_eos

    def __call__(self, batch: list[dict]):
        torch = self.torch
        from ..prompting import call_processor, extract_media

        texts, images, videos, target_ids_list = [], [], [], []
        for item in batch:
            prompt = self.processor.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=True)
            t_ids = self.tokenizer.encode(item["target_text"], add_special_tokens=False)
            eos = self.tokenizer.eos_token or ""
            if self.train_on_eos and eos:
                t_ids = t_ids + [self.tokenizer.eos_token_id]
            texts.append(prompt + item["target_text"] + (eos if self.train_on_eos else ""))
            target_ids_list.append(t_ids)
            im, vi = extract_media(item["messages"])
            images.extend(im)
            videos.extend(vi)

        enc = call_processor(self.processor, texts, images, videos,
                             padding=True, return_tensors="pt")
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = torch.full_like(input_ids, -100)
        for b, t_ids in enumerate(target_ids_list):
            seq_len = int(attn[b].sum().item())
            k = len(t_ids)
            if self.tokenizer.padding_side == "left":
                start, end = input_ids.shape[1] - k, input_ids.shape[1]
            else:
                start, end = seq_len - k, seq_len
            got = input_ids[b, start:end].tolist()
            if got != t_ids:
                raise ValueError(f"라벨 정렬 실패: 끝 토큰 {got} ≠ target {t_ids}")
            labels[b, start:end] = input_ids[b, start:end]
        enc["labels"] = labels
        return enc
