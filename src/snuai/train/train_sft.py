"""QLoRA SFT 학습 스크립트 (8B: 로컬 3090 / 32B: VESSL A100 — 같은 스크립트).

Ver3 기본값 = Ver1 검증 레시피(이미지 모드 + uniform 증강) + legend 프롬프트:
  video-mode/counterfactual OFF(Ver2 하락 원인 — VER3.md) · perm_mode=uniform ·
  legend ON(A~X↔순열 범례 명시, Ver1 대비 유일한 프롬프트 변경) ·
  max-pixels 602112 · max-steps 2000 · save-total-limit 12(체크포인트 사후 선택용).

예 (8B, Ver3 기본값 그대로):
  python -m snuai.train.train_sft --csv data/train.csv --image-dir data/train \
      --out runs/sft8b_v3 --grad-accum 16

예 (32B, VESSL A100 80GB — bnb-4bit 체크포인트 + 세션 재개):
  python -m snuai.train.train_sft --csv train.csv --image-dir images/train \
      --model-id unsloth/Qwen3-VL-32B-Instruct-bnb-4bit --prequantized \
      --out /vessl-volume/sft32b --resume --grad-accum 32

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
                    verify_lora_only_on_language, verify_vision_not_quantized)


def load_model_and_processor(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(args.model_id)

    kw: dict = {"dtype": torch.bfloat16}
    if args.precision == "bf16":
        pass  # 풀정밀 LoRA (양자화 비용 곡선 E2 실험·VESSL 트랙) — 양자화 없음
    elif args.prequantized:
        pass  # unsloth bnb-4bit 체크포인트: 자체 quantization_config 포함
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
                              epoch_multiplier=args.epoch_multiplier)
    return Score24SFTDataset(train_s, ds_cfg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    d = ap.add_argument
    d("--csv", required=True); d("--image-dir", required=True)
    d("--caption-col", default="Caption"); d("--val-frac", type=float, default=0.1)
    d("--model-id", default="/home/yhmin/model/hub/Qwen3-VL-8B-Instruct")
    d("--prequantized", action="store_true", help="bnb-4bit 사전양자화 체크포인트 사용")
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
        # loss·lr·grad_norm 실시간 곡선: tensorboard --logdir <out>/tb
        report_to="tensorboard", logging_dir=str(out / "tb"),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=collator)
    trainer.train(resume_from_checkpoint=args.resume or None)
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
