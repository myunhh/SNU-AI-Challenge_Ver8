"""train_dpo.py 순수 로직 — 전부 CPU. 실제 HF Trainer/모델 없이 검증한다.

DPOCollator._compute_letter_token_ids는 토크나이저 스텁만 있으면 되고,
_make_dpo_trainer_cls(base_trainer_cls, beta)는 팩토리라 object를 베이스로 넣으면
transformers.Trainer 없이 compute_loss의 DPO 손실 수식만 단위테스트할 수 있다.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from snuai import perm
from snuai.train.train_dpo import DPOCollator, _make_dpo_trainer_cls


class _LetterTokenizer:
    """A..X ↔ 0..23 단일 토큰, 그 외 문자열은 항상 2토큰(모호함 시뮬레이션)."""

    def __init__(self, known=None):
        self.known = known if known is not None else {ch: i for i, ch in enumerate(perm.LETTERS24)}

    def encode(self, text, add_special_tokens=False):
        if text in self.known:
            return [self.known[text]]
        return [998, 999]


def test_compute_letter_token_ids_maps_az_to_ids():
    processor = SimpleNamespace(tokenizer=_LetterTokenizer())
    coll = DPOCollator(processor)
    ids = coll._compute_letter_token_ids()
    assert ids == list(range(24))
    assert coll._compute_letter_token_ids() == ids   # 캐시 경로도 동일 결과


def test_compute_letter_token_ids_raises_when_ambiguous():
    processor = SimpleNamespace(tokenizer=_LetterTokenizer(known={}))   # 전부 모호
    coll = DPOCollator(processor)
    with pytest.raises(ValueError):
        coll._compute_letter_token_ids()


class _FakeOutputs:
    def __init__(self, logits):
        self.logits = logits


class _FakeDPOModel:
    """model(**inputs) + model.disable_adapter() 컨텍스트 — policy/ref 로짓을 토글."""

    def __init__(self, policy_logits, ref_logits):
        self.policy_logits = policy_logits
        self.ref_logits = ref_logits
        self._ref_mode = False
        self.calls = []
        self.disable_adapter_calls = 0

    def __call__(self, **inputs):
        self.calls.append(set(inputs.keys()))
        return _FakeOutputs(self.ref_logits if self._ref_mode else self.policy_logits)

    @contextmanager
    def disable_adapter(self):
        self.disable_adapter_calls += 1
        self._ref_mode = True
        try:
            yield
        finally:
            self._ref_mode = False


def test_dpo_loss_matches_manual_calculation_and_pops_extra_keys():
    torch.manual_seed(0)
    B, L, V = 2, 4, 30
    policy_logits = torch.randn(B, L, V)
    ref_logits = torch.randn(B, L, V)
    last_pos = torch.tensor([3, 3])
    chosen_ids = torch.tensor([5, 7])
    rejected_ids = torch.tensor([6, 8])
    beta = 0.1

    model = _FakeDPOModel(policy_logits, ref_logits)
    inputs = {
        "input_ids": torch.zeros(B, L, dtype=torch.long),
        "last_pos": last_pos, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids,
    }
    trainer_cls = _make_dpo_trainer_cls(object, beta=beta)
    trainer = trainer_cls()
    loss = trainer.compute_loss(model, dict(inputs))

    # model(**inputs)에는 last_pos/chosen_ids/rejected_ids가 전달되면 안 됨(pop 확인)
    assert model.calls[0] == {"input_ids"}
    assert model.disable_adapter_calls == 1

    b = torch.arange(B)
    pi_logp = torch.log_softmax(policy_logits[b, last_pos].float(), dim=-1)
    ref_logp = torch.log_softmax(ref_logits[b, last_pos].float(), dim=-1)
    pi_ratio = pi_logp[b, chosen_ids] - pi_logp[b, rejected_ids]
    ref_ratio = ref_logp[b, chosen_ids] - ref_logp[b, rejected_ids]
    expected = -torch.nn.functional.logsigmoid(beta * (pi_ratio - ref_ratio)).mean()

    assert torch.isclose(loss, expected, atol=1e-6)


def test_dpo_loss_prefers_larger_chosen_minus_rejected_margin():
    """정책이 참조보다 chosen을 더 선호할수록(마진 개선) loss가 작아져야 함(단조성 검증)."""
    V = 10
    last_pos = torch.tensor([0])
    chosen_ids, rejected_ids = torch.tensor([0]), torch.tensor([1])
    ref_logits = torch.zeros(1, 1, V)   # 참조: chosen=rejected 무차별

    def loss_for(chosen_boost):
        logits = torch.zeros(1, 1, V)
        logits[0, 0, 0] = chosen_boost    # policy가 chosen을 얼마나 밀어주는지
        model = _FakeDPOModel(logits, ref_logits)
        trainer = _make_dpo_trainer_cls(object, beta=1.0)()
        return trainer.compute_loss(model, {"input_ids": torch.zeros(1, 1, dtype=torch.long),
                                            "last_pos": last_pos, "chosen_ids": chosen_ids,
                                            "rejected_ids": rejected_ids}).item()

    assert loss_for(chosen_boost=5.0) < loss_for(chosen_boost=0.0) < loss_for(chosen_boost=-5.0)
