"""QLoRA SFT 학습 스크립트 (Ver8: 32B-bnb-4bit 기본, VESSL A100 트랙 — Ver4와 동일 base).

Ver3 레시피(이미지 모드 + uniform 증강 + legend 프롬프트, `video-mode/counterfactual`
OFF·`perm_mode=uniform`·`max-pixels 602112`·`max-steps 2000`)를 그대로 따르되
base 모델만 32B-bnb-4bit(Ver4 E1/A안: 4bit 학습=4bit 서빙)로 교체한 것이 Ver8
기본값. 사전양자화 체크포인트는 config.json의 quantization_config로 자동 감지되어
vision-skip 보정(patch_prequant_vision_skip)까지 알아서 적용된다 — 플래그 불필요.

예 (32B, VESSL A100 80GB — 세션 재개):
  python -m snuai.train.train_sft --csv train.csv --image-dir images/train \
      --model-id unsloth/Qwen3-VL-32B-Instruct-bnb-4bit \
      --out /vessl-volume/sft32b_v8 --resume --grad-accum 32

예 (8B 회귀 실험 — Ver3/Ver7 계열과 직접 비교):
  python -m snuai.train.train_sft --csv data/train.csv --image-dir data/train \
      --model-id /home/yhmin/model/hub/Qwen3-VL-8B-Instruct --out runs/sft8b --grad-accum 16

예 (§2a DPO 인접 스와프 — SFT 어댑터 이어서 학습): train_dpo.py 참고.
예 (§2b metric-aligned soft SFT — §1이 쌍순서로 판명될 때만):
  ... --soft-label-temperature 0.15   (기본 None = one-hot, byte-identical)

VESSL 재개 설계(노션 결정): --out을 영속 볼륨에 두고 --resume만 붙이면
Trainer가 마지막 checkpoint(어댑터+옵티마이저 상태)에서 이어서 학습한다.

⚠️ 학습·추론 환경의 transformers/bitsandbytes 버전 일치 필수 (노션 유의사항).
   버전은 학습 시작 시 out/env.json에 기록되어 추론 쪽과 대조할 수 있다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import perm
from ..data.augment import AugmentConfig, identity_ratio_for_target
from ..data.sample import load_csv
from ..data.split import split_samples
from ..infer.engine import apply_pixel_budget
from .dataset import Score24SFTDataset, SFTCollator, SFTDatasetConfig
from .qlora import (lora_targets_of_model, make_bnb_kwargs, make_lora_config,
                    patch_prequant_vision_skip, verify_lora_only_on_language,
                    verify_vision_not_quantized)


def _make_soft_label_trainer_cls(base_trainer_cls):
    """§2b metric-aligned soft SFT — Trainer를 상속하되 compute_loss만 교체.

    letter 위치는 collator가 hard label에서 이미 빼놨으므로(labels[b,start]=-100)
    outputs.loss는 EOS 등 나머지 위치의 표준 CE만 남는다. 거기에 letter 위치를
    24클래스로 제한한 soft CE(-Σ target·log_softmax)를 더한다 — 채점이 쌍순서
    부분점수일 때 one-hot보다 채점 함수와 정렬되는 타깃(TODO §2b). EM으로 판명되면
    이 경로는 쓰지 않는다(soft_label_temperature=None이 기본, byte-identical 유지).
    transformers.Trainer는 무거운 의존이라 CPU 테스트 임포트를 막지 않게 지연 생성.
    """

    class SoftLabelTrainer(base_trainer_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch
            soft_targets = inputs.pop("soft_targets", None)
            answer_pos = inputs.pop("answer_pos", None)
            letter_token_ids = inputs.pop("letter_token_ids", None)
            outputs = model(**inputs)
            loss = outputs.loss
            if soft_targets is not None:
                logits = outputs.logits
                b_idx = torch.arange(logits.shape[0], device=logits.device)
                pred_logits = logits[b_idx, answer_pos - 1]       # letter 직전 위치가 letter를 예측
                letter_logits = pred_logits[:, letter_token_ids]   # (B, 24)로 제한
                logp = torch.log_softmax(letter_logits.float(), dim=-1)
                soft_loss = -(soft_targets * logp).sum(dim=-1).mean()
                loss = loss + soft_loss
            return (loss, outputs) if return_outputs else loss

    return SoftLabelTrainer


def load_model_and_processor(args):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(args.model_id)

    # 사전양자화 체크포인트 자동 감지 (engine._is_prequantized와 동일 규약) — 32B-bnb-4bit
    # 등은 config.json에 quantization_config가 이미 내장돼 있어 별도 BitsAndBytesConfig 불필요.
    auto_cfg = AutoConfig.from_pretrained(args.model_id)
    prequantized = getattr(auto_cfg, "quantization_config", None) is not None

    kw: dict = {"dtype": torch.bfloat16}
    if args.precision == "bf16":
        if prequantized:
            raise SystemExit("--precision bf16에 사전양자화(bnb-4bit) 체크포인트는 불가 — "
                             "비양자화 체크포인트를 --model-id로 지정할 것")
        pass  # 풀정밀 LoRA (양자화 비용 곡선 E2 실험·VESSL 트랙) — 양자화 없음
    elif prequantized:
        # 체크포인트 자체 quantization_config의 vision 스킵이 bare name이라 이 버전의
        # transformers에서 vision까지 4bit로 재양자화된다 — model.visual 접두형으로 보정
        # (Ver4 실측 이식: unsloth Qwen3-VL-32B-Instruct-bnb-4bit에서 확인된 버그).
        if patch_prequant_vision_skip(auto_cfg):
            print("[quant] 사전양자화 skip_modules에 model.visual 보정(vision 비양자화 강제)")
        kw["config"] = auto_cfg
        print("[quant] 사전양자화 체크포인트 감지 — 보정된 quantization_config로 로드")
    else:
        bnb = make_bnb_kwargs()
        bnb["bnb_4bit_compute_dtype"] = torch.bfloat16
        kw["quantization_config"] = BitsAndBytesConfig(**bnb)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, device_map={"": 0}, attn_implementation=args.attn, **kw)

    print("[verify]", verify_vision_not_quantized(model))

    from peft import get_peft_model, prepare_model_for_kbit_training
    model.config.use_cache = False
    if args.precision == "bf16":
        # 비양자화 경로: kbit 준비 대신 체크포인팅+입력 grad만 활성화
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = lora_targets_of_model(model)
    print(f"[lora] target {len(targets)}개 (vision 제외 확인됨) 예: {targets[:2]}")
    model = get_peft_model(model, make_lora_config(
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        target_modules=targets))
    print("[verify]", verify_lora_only_on_language(model))
    model.print_trainable_parameters()
    return model, processor


def build_dataset(args):
    samples = load_csv(args.csv, args.image_dir, caption_col=args.caption_col)
    train_s, val_s = split_samples(samples, val_frac=args.val_frac)
    print(f"[data] train {len(train_s)} / holdout {len(val_s)} (holdout은 predict.py로 평가)")
    aug = AugmentConfig(
        perm_mode=args.perm_mode,
        identity_ratio=(identity_ratio_for_target(args.identity_target)
                        if args.perm_mode == "identity_ratio" else 0.0),
        grayscale_p=args.grayscale_p, jitter_p=args.jitter_p,
        res_aug_p=args.res_aug_p, blur_p=args.blur_p)
    ds_cfg = SFTDatasetConfig(augment=aug, video_mode=args.video_mode,
                              video_dup_factor=args.video_dup_factor,
                              counterfactual=args.counterfactual,
                              legend=args.legend,
                              verify_ratio=args.verify_ratio,
                              epoch_multiplier=args.epoch_multiplier,
                              soft_label_temperature=args.soft_label_temperature)
    return Score24SFTDataset(train_s, ds_cfg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    d = ap.add_argument
    d("--csv", required=True); d("--image-dir", required=True)
    d("--caption-col", default="Caption"); d("--val-frac", type=float, default=0.1)
    # 32B-bnb-4bit(A100/VESSL 트랙) 기본 — 사전양자화는 config.json으로 자동 감지되므로
    # 플래그 불필요. 8B 회귀 실험 시 --model-id /home/yhmin/model/hub/Qwen3-VL-8B-Instruct
    d("--model-id", default="/home/yhmin/model/hub/Qwen3-VL-32B-Instruct-bnb-4bit")
    # 4bit=QLoRA(기본, 24GB 검증됨) / bf16=풀정밀 LoRA(E2 실험 — 24GB 경계, 스모크 필수)
    d("--precision", default="4bit", choices=["4bit", "bf16"])
    # Ver2: 추론(predict.py)과 동일한 602112으로 정렬 — 학습/추론 해상도 일치 원칙
    d("--attn", default="sdpa"); d("--max-pixels", type=int, default=602112)
    d("--lora-r", type=int, default=16); d("--lora-alpha", type=int, default=32)
    d("--lora-dropout", type=float, default=0.05)
    # 증강 — Ver3 기본: uniform 복원. identity_ratio는 Ver2 홀드아웃에서 항등 오탐을
    # 9→74건으로 늘리고 recall 개선은 0이었음 (VER3.md) — A/B 옵션으로만 유지.
    d("--perm-mode", default="uniform", choices=["uniform", "identity_ratio", "off"])
    d("--identity-target", type=float, default=0.155)
    d("--grayscale-p", type=float, default=0.1); d("--jitter-p", type=float, default=0.3)
    d("--res-aug-p", type=float, default=0.3); d("--blur-p", type=float, default=0.1)
    d("--verify-ratio", type=float, default=0.1)
    d("--epoch-multiplier", type=int, default=1)
    d("--soft-label-temperature", type=float, default=None,
      help="§2b metric-aligned soft SFT (기본 None=one-hot, byte-identical). "
           "쌍순서 채점 확정 후에만 켤 것 — EM이면 one-hot이 이미 최적(TODO §2b)")
    # Ver3 기본 OFF: video-mode는 Ver2 하락 주범(프레임 쌍 병합·토큰 -67%·타임스탬프
    # 2개 — VER3.md), counterfactual은 CoT 경로 전용으로 회귀. legend만 신규 ON.
    d("--video-mode", action=argparse.BooleanOptionalAction, default=False)
    # Ver7 video_dup: R1 재도전 — 각 프레임 연속 복제로 temporal 병합쌍을 자기 자신과의
    # 쌍으로 만든다(무관 프레임 병합 오염 차단). 1 또는 짝수만 유효.
    d("--video-dup-factor", type=int, default=1,
      help="video-mode에서 각 프레임을 이만큼 연속 복제 (1 또는 짝수만 유효)")
    d("--video-max-pixels", type=int, default=None,
      help="video_processor 전용 예산(전체 프레임 합산제) — 생략 시 "
           "4*video-dup-factor*max-pixels로 자동 계산")
    d("--counterfactual", action=argparse.BooleanOptionalAction, default=False)
    d("--legend", action=argparse.BooleanOptionalAction, default=True,
      help="A~X↔순열 범례를 score24 프롬프트에 명시")
    # 트레이닝 — Ver2: 2000스텝(≈3.7에폭; Ver1 1000스텝 종료 시점에도 loss 하강 중이었음)
    d("--out", required=True); d("--epochs", type=float, default=1.0)
    d("--max-steps", type=int, default=2000)
    d("--lr", type=float, default=1e-4); d("--grad-accum", type=int, default=16)
    d("--save-steps", type=int, default=200); d("--logging-steps", type=int, default=10)
    d("--resume", action="store_true", help="out의 마지막 checkpoint에서 재개(VESSL)")
    d("--unsloth", action="store_true", help="Unsloth 가속 경로(설치 시)")
    args = ap.parse_args(argv)

    # video_dup 가드 — out.mkdir/_record_env(부수효과) 이전에 즉시 하드 실패시켜
    # 잘못된 플래그 조합으로 빈 run 디렉터리가 생기지 않게 함(predict.py와 동일 원칙).
    if args.video_dup_factor > 1 and not args.video_mode:
        raise SystemExit("--video-dup-factor>1은 --video-mode에서만 유효")
    if args.video_dup_factor != 1 and args.video_dup_factor % 2 != 0:
        raise SystemExit("--video-dup-factor는 1 또는 짝수만 유효 "
                         "(홀수는 temporal 병합쌍이 프레임 블록과 어긋나 일부 교차 오염됨)")

    # transformers 5.x는 기본 verbosity(warning)에서 Trainer의 loss 로그(INFO)를 삼킨다
    import transformers
    transformers.logging.set_verbosity_info()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _record_env(out)

    if args.unsloth:
        model, processor = _load_unsloth(args)
    else:
        model, processor = load_model_and_processor(args)
    # 학습 해상도 예산 — 추론 engine과 같은 함수로 image/video processor 모두,
    # transformers 4.x(max_pixels)/5.x(size.longest_edge) 규약 모두 적용.
    # video_max_pixels 자동계산 근거는 infer/engine.py의 apply_pixel_budget 주석 참고.
    video_max_pixels = args.video_max_pixels
    if video_max_pixels is None and args.video_mode and args.video_dup_factor > 1:
        video_max_pixels = perm.N * args.video_dup_factor * args.max_pixels
    apply_pixel_budget(processor, max_pixels=args.max_pixels, video_max_pixels=video_max_pixels)
    train_ds = build_dataset(args)
    collator = SFTCollator(processor)

    # report_to="tensorboard"를 무조건 주면 tensorboard 미설치 환경(예: VESSL 기본
    # 이미지)에서 TensorBoardCallback 생성자가 즉시 crash한다(명시 지정은 존재 여부
    # 필터를 우회함). 설치돼 있을 때만 켠다 — 없어도 watch_train.py가 trainer_state.json
    # 으로 곡선을 그리므로 학습엔 지장 없음.
    import importlib.util
    _tb = any(importlib.util.find_spec(m) is not None
              for m in ("tensorboard", "tensorboardX"))

    from transformers import Trainer, TrainingArguments
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,           # 24GB 규약 (노션: batch=1)
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Ver3: 전 체크포인트 보존(2000스텝/200 = 10개) → checkpoint-1000 vs 2000을
        # 홀드아웃으로 사후 비교해 스텝 수 효과를 분리 (학습 중 eval hook 대체)
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=12,
        remove_unused_columns=False,             # 커스텀 collator 필수 설정
        dataloader_num_workers=2,
        # loss·lr·grad_norm 실시간 곡선: tensorboard --logdir <out>/tb (설치 시에만)
        report_to=("tensorboard" if _tb else "none"), logging_dir=str(out / "tb"),
    )
    trainer_cls = (_make_soft_label_trainer_cls(Trainer)
                   if args.soft_label_temperature is not None else Trainer)
    trainer = trainer_cls(model=model, args=targs, train_dataset=train_ds,
                          data_collator=collator)
    # VESSL 재개 설계: --resume는 항상 붙여도 안전해야 한다(첫 런엔 체크포인트가 없음).
    # resume_from_checkpoint=True를 그대로 넘기면 첫 런에서 "No valid checkpoint" 에러가
    # 나므로, out에 실제 체크포인트가 있을 때만 그 경로를 재개 대상으로 넘긴다.
    resume_ckpt = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        resume_ckpt = get_last_checkpoint(str(out))
        if resume_ckpt is None:
            print(f"[resume] {out}에 체크포인트 없음 → 처음부터 학습")
        else:
            print(f"[resume] {resume_ckpt}에서 재개")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    import torch
    if torch.cuda.is_available():
        # bf16/4bit VRAM 비교 실측용 (TODO §5 스모크)
        print(f"[vram] peak allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB")

    trainer.save_model(str(out / "adapter_final"))
    processor.save_pretrained(str(out / "adapter_final"))
    print(f"[done] adapter → {out/'adapter_final'} — 추론: predict.py --adapter 로 사용")


def _load_unsloth(args):
    try:
        from unsloth import FastVisionModel
    except ImportError as e:
        raise SystemExit(f"unsloth 미설치: pip install unsloth ({e})")
    model, processor = FastVisionModel.from_pretrained(
        args.model_id, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        finetune_vision_layers=False,           # vision 제외 규약
        finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True)
    return model, processor


def _record_env(out: Path):
    """학습·추론 환경 버전 일치 검증용 기록 (노션 유의사항 대응)."""
    import sys
    env = {"python": sys.version}
    for pkg in ("torch", "transformers", "bitsandbytes", "peft", "accelerate"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception:
            env[pkg] = None
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("[env]", env)


if __name__ == "__main__":
    main()
