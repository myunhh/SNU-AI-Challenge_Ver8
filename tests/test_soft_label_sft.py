"""§2b metric-aligned soft SFT — dataset 배선 + SoftLabelTrainer 손실 계산 (전부 CPU).

실제 HF Trainer/모델 없이 검증한다: Score24SFTDataset은 순수 파이썬 로직이고,
SoftLabelTrainer는 base_trainer_cls를 주입받는 팩토리(_make_soft_label_trainer_cls)라
object를 베이스로 넣으면 무거운 transformers.Trainer 없이 compute_loss만 단위테스트 가능.
"""

import torch

from snuai import perm
from snuai.data.augment import AugmentConfig
from snuai.data.sample import Sample
from snuai.train.dataset import Score24SFTDataset, SFTDatasetConfig
from snuai.train.train_sft import _make_soft_label_trainer_cls


def _marker_image(k: int):
    from PIL import Image
    return Image.new("RGB", (8, 8), color=(k, 0, 0))


def test_dataset_soft_target_matches_perm_function_when_enabled():
    rank = (2, 0, 3, 1)
    sample = Sample(id="t", caption="c", images=[_marker_image(k) for k in range(4)], rank=rank)
    cfg = SFTDatasetConfig(augment=AugmentConfig(perm_mode="off"),  # 셔플 없이 라벨 고정
                           soft_label_temperature=0.2)
    ds = Score24SFTDataset([sample], cfg)
    item = ds[0]
    assert "soft_target" in item
    expected = perm.soft_target_distribution(item["rank"], 0.2)
    assert item["soft_target"] == expected


def test_dataset_omits_soft_target_by_default():
    rank = (0, 1, 2, 3)
    sample = Sample(id="t", caption="c", images=[_marker_image(k) for k in range(4)], rank=rank)
    ds = Score24SFTDataset([sample], SFTDatasetConfig())
    assert "soft_target" not in ds[0]


class _FakeOutputs:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class _FakeModel:
    """model(**inputs) 스텁 — 고정된 loss/logits를 돌려주고 어떤 kwargs로 불렸는지 기록."""

    def __init__(self, loss, logits):
        self._loss = loss
        self._logits = logits
        self.seen_keys = None

    def __call__(self, **inputs):
        self.seen_keys = set(inputs.keys())
        return _FakeOutputs(self._loss, self._logits)


def _make_trainer():
    # base_trainer_cls=object → transformers.Trainer 없이 compute_loss만 테스트
    cls = _make_soft_label_trainer_cls(object)
    return cls()


def test_soft_label_loss_matches_manual_calculation():
    torch.manual_seed(0)
    V, L, B = 40, 5, 2
    logits = torch.randn(B, L, V)
    answer_pos = torch.tensor([3, 4])          # letter 토큰의 시퀀스 내 위치
    letter_token_ids = torch.tensor(list(range(24)))
    soft_targets = torch.tensor(perm.soft_target_distribution((2, 0, 3, 1), 0.2)), \
                   torch.tensor(perm.soft_target_distribution((0, 1, 2, 3), 0.2))
    soft_targets = torch.stack(soft_targets)
    hard_loss = torch.tensor(0.7)

    model = _FakeModel(hard_loss, logits)
    inputs = {
        "input_ids": torch.zeros(B, L, dtype=torch.long),   # 더미 — pop 대상 아님, 그대로 전달돼야 함
        "soft_targets": soft_targets,
        "answer_pos": answer_pos,
        "letter_token_ids": letter_token_ids,
    }
    trainer = _make_trainer()
    loss = trainer.compute_loss(model, dict(inputs))

    # model(**inputs)에는 soft_targets/answer_pos/letter_token_ids가 전달되면 안 됨(pop 확인)
    assert model.seen_keys == {"input_ids"}

    # 수동 계산: letter 직전 위치(answer_pos-1) 로짓을 24클래스로 제한 후 soft CE
    pred = logits[torch.arange(B), answer_pos - 1]      # (B, V)
    restricted = pred[:, letter_token_ids]               # (B, 24)
    logp = torch.log_softmax(restricted, dim=-1)
    expected_soft = -(soft_targets * logp).sum(dim=-1).mean()
    expected_total = hard_loss + expected_soft

    assert torch.isclose(loss, expected_total, atol=1e-6)


def test_soft_label_loss_falls_back_to_hard_loss_when_disabled():
    """soft_targets가 없는 배치(soft-label 미사용)는 표준 loss 그대로 반환 — byte-identical."""
    hard_loss = torch.tensor(1.23)
    model = _FakeModel(hard_loss, torch.zeros(2, 3, 10))
    trainer = _make_trainer()
    loss = trainer.compute_loss(model, {"input_ids": torch.zeros(2, 3, dtype=torch.long)})
    assert torch.equal(loss, hard_loss)
