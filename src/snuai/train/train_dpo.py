"""DPO 인접 스와프 hard negative (TODO §2a, 1순위) — SFT 어댑터 이어서 학습.

배경: 홀드아웃 오답의 최빈 유형이 KT거리 1(인접 스와프 한 개 차이, Ver7 실측
189/945건). chosen=정답 letter, rejected=인접 스와프 3종 중 **현재 SFT 어댑터가
가장 그럴듯하다고 보는 오답**(hard negative, dpo_pairs.build_dpo_records의
scorer 인자)으로 선호쌍을 구성해 그 오답 방향을 직접 깎는다.

score24는 단일 토큰 구조라 TRL DPOTrainer(멀티턴 생성 전제)를 쓸 이유가 없다 —
같은 프롬프트에서 다음 토큰의 chosen/rejected 두 로그확률만 비교하면 되므로,
직접 구현한 단일토큰 DPO(compute_loss)가 더 단순하고 확실하다. 참조 로그확률은
별도 참조모델 없이 PEFT의 model.disable_adapter()로 얻는다(LoRA만 끄면 SFT 이전
base가 곧 참조 정책이라는 표준 트릭 — VRAM 두 배 필요 없음).

사용 (32B-4bit, VESSL A100 — Ver4/Ver8 공통 base):
  python -m snuai.train.train_dpo --csv data/train.csv --image-dir data/train \
      --model-id unsloth/Qwen3-VL-32B-Instruct-bnb-4bit \
      --adapter runs/sft32b_v8/adapter_final --out runs/dpo32b_v8 --max-steps 1500

--hard-negative(기본 on)는 학습 시작 전 --adapter로 전체 train 홀드아웃-제외분을
한 번 스코어링해 rejected를 고른다(추가 forward 1회분, 학습 자체보다 훨씬 쌈).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import perm
from ..data.sample import load_csv
from ..data.split import split_samples
from ..infer.engine import apply_pixel_budget
from .dpo_pairs import DPOPairConfig, build_dpo_records
from .qlora import patch_prequant_vision_skip, verify_lora_only_on_language, verify_vision_not_quantized


def _make_dpo_trainer_cls(base_trainer_cls, beta: float):
    """단일토큰 DPO — Trainer를 상속하되 compute_loss만 교체 (transformers 지연 임포트 위해 팩토리).

    다음 토큰 로짓(prompt 끝 = letter 직전 위치)에서 chosen/rejected 두 토큰의
    log-softmax만 비교하면 되므로 시퀀스 전체를 이어붙일 필요가 없다 — collator가
    prompt만 인코딩하고 chosen_ids/rejected_ids(단일 토큰 id)를 함께 넘긴다.
    """

    class DPOSingleTokenTrainer(base_trainer_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch
            chosen_ids = inputs.pop("chosen_ids")
            rejected_ids = inputs.pop("rejected_ids")
            last_pos = inputs.pop("last_pos")

            outputs = model(**inputs)
            b_idx = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
            logp = torch.log_softmax(outputs.logits[b_idx, last_pos].float(), dim=-1)
            pi_chosen, pi_rejected = logp[b_idx, chosen_ids], logp[b_idx, rejected_ids]

            with torch.no_grad(), model.disable_adapter():
                ref_out = model(**inputs)
                ref_logp = torch.log_softmax(ref_out.logits[b_idx, last_pos].float(), dim=-1)
                ref_chosen, ref_rejected = ref_logp[b_idx, chosen_ids], ref_logp[b_idx, rejected_ids]

            pi_ratio = pi_chosen - pi_rejected
            ref_ratio = (ref_chosen - ref_rejected).detach()
            dpo_logits = beta * (pi_ratio - ref_ratio)
            loss = -torch.nn.functional.logsigmoid(dpo_logits).mean()
            if return_outputs:
                return loss, outputs
            return loss

    return DPOSingleTokenTrainer


class DPOCollator:
    """DPO 레코드({"prompt_messages","chosen","rejected",...}) 배치 → 모델 입력.

    SFT collator와 달리 target 텍스트를 이어붙이지 않는다 — chosen/rejected 둘 다
    같은 prompt의 '다음 토큰'이라 prompt 하나만 인코딩하고 두 토큰 id를 따로 넘기면
    충분하다(단일 토큰 구조의 이점).
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self._letter_token_ids: list[int] | None = None

    def _compute_letter_token_ids(self) -> list[int]:
        if self._letter_token_ids is not None:
            return self._letter_token_ids
        ids = []
        for ch in perm.LETTERS24:
            for cand in (ch, " " + ch):
                cids = self.tokenizer.encode(cand, add_special_tokens=False)
                if len(cids) == 1:
                    ids.append(cids[0])
                    break
            else:
                raise ValueError(f"단일 토큰이 아님: {ch!r}")
        if len(set(ids)) != 24:
            raise ValueError("라벨 토큰 id 충돌 — 토크나이저 확인 필요")
        self._letter_token_ids = ids
        return ids

    def __call__(self, batch: list[dict]):
        import torch
        from ..prompting import call_processor, extract_media

        texts, images, videos = [], [], []
        for rec in batch:
            prompt = self.processor.apply_chat_template(
                rec["prompt_messages"], tokenize=False, add_generation_prompt=True)
            texts.append(prompt)
            im, vi = extract_media(rec["prompt_messages"])
            images.extend(im)
            videos.extend(vi)

        enc = call_processor(self.processor, texts, images, videos,
                             padding=True, return_tensors="pt")
        attn = enc["attention_mask"]
        if self.tokenizer.padding_side == "left":
            last_pos = torch.full((attn.shape[0],), attn.shape[1] - 1, dtype=torch.long)
        else:
            last_pos = attn.sum(dim=1) - 1

        letter_ids = self._compute_letter_token_ids()
        chosen_ids = torch.tensor([letter_ids[perm.index_of_letter(r["chosen"])] for r in batch])
        rejected_ids = torch.tensor([letter_ids[perm.index_of_letter(r["rejected"])] for r in batch])

        enc["last_pos"] = last_pos
        enc["chosen_ids"] = chosen_ids
        enc["rejected_ids"] = rejected_ids
        return enc


def _build_hard_negative_scorer(args):
    """--adapter로 VLMEngine을 잠깐 띄워 Score24Scorer 스코어러 함수를 만든다.

    학습용 모델 로딩(load_model_and_adapter)과 별도 인스턴스 — 스코어링이 끝나면
    호출자가 명시적으로 해제(del + empty_cache)해 학습 시작 전 VRAM을 비운다.
    """
    from ..infer.engine import EngineConfig, VLMEngine
    from ..infer.scorers import Score24Scorer

    eng = VLMEngine(EngineConfig(model_id=args.model_id, adapter_path=args.adapter,
                                attn=args.attn, max_pixels=args.max_pixels))
    scorer = Score24Scorer(eng, legend=args.legend)

    def scorer_fn(caption, images):
        return scorer.scores(caption, images)

    return scorer_fn, eng


def load_model_and_adapter(args):
    """base(사전양자화 자동 감지) + 기존 SFT 어댑터를 is_trainable=True로 로드.

    device_map은 PartialState().process_index로 고정 — torchrun/accelerate
    멀티프로세스(DDP) 하에서 각 rank가 자기 GPU에만 올라가게 한다(단일 프로세스
    실행 시엔 process_index==0이라 기존과 동일하게 동작, 하위호환).
    """
    import torch
    from accelerate import PartialState
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id)
    auto_cfg = AutoConfig.from_pretrained(args.model_id)
    prequantized = getattr(auto_cfg, "quantization_config", None) is not None
    if not prequantized:
        raise SystemExit("train_dpo.py는 사전양자화(bnb-4bit) 체크포인트 전용 — "
                         "비양자화 8B 회귀 실험은 --precision bf16 지원이 필요하면 추가할 것")
    if patch_prequant_vision_skip(auto_cfg):
        print("[quant] 사전양자화 skip_modules에 model.visual 보정(vision 비양자화 강제)")

    device_map = {"": PartialState().process_index}
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, config=auto_cfg, device_map=device_map,
        attn_implementation=args.attn, dtype=torch.bfloat16)
    print("[verify]", verify_vision_not_quantized(model))

    model.config.use_cache = False
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    print("[verify]", verify_lora_only_on_language(model))
    model.print_trainable_parameters()
    return model, processor


def main(argv=None):
    ap = argparse.ArgumentParser()
    d = ap.add_argument
    d("--csv", required=True); d("--image-dir", required=True)
    d("--caption-col", default="Caption"); d("--val-frac", type=float, default=0.1)
    d("--model-id", default="/home/yhmin/model/hub/Qwen3-VL-32B-Instruct-bnb-4bit")
    d("--adapter", required=True, help="이어서 학습할 SFT 어댑터 (runs/.../adapter_final)")
    d("--attn", default="sdpa"); d("--max-pixels", type=int, default=602112)
    d("--legend", action=argparse.BooleanOptionalAction, default=True)
    d("--rejected-per-sample", type=int, default=1)
    d("--include-random-rejected", action="store_true")
    d("--hard-negative", action=argparse.BooleanOptionalAction, default=True,
      help="on(기본): --adapter로 3종 스코어링 후 최고점 오답 채택. off: 무작위 선택")
    d("--beta", type=float, default=0.1, help="DPO 온도(참조 대비 로그오즈 스케일)")
    d("--out", required=True)
    d("--max-steps", type=int, default=1500)
    d("--lr", type=float, default=5e-6); d("--grad-accum", type=int, default=16)
    d("--save-steps", type=int, default=200); d("--logging-steps", type=int, default=10)
    d("--resume", action="store_true")
    d("--seed", type=int, default=777)
    d("--ddp-find-unused-parameters", action="store_true", default=False,
      help="DDP(멀티GPU)에서 'mark variable ready only once' 크래시 시 켜볼 것 "
           "(gradient checkpointing+PEFT 조합에서 간헐적으로 필요)")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from accelerate import PartialState
    state = PartialState()   # torchrun 환경변수가 있으면 여기서 프로세스그룹 초기화

    samples = load_csv(args.csv, args.image_dir, caption_col=args.caption_col)
    train_s, val_s = split_samples(samples, val_frac=args.val_frac)
    print(f"[data] train {len(train_s)} / holdout {len(val_s)} (SFT와 동일 split — split.py 결정적)")

    pair_cfg = DPOPairConfig(rejected_per_sample=args.rejected_per_sample,
                             include_random_rejected=args.include_random_rejected,
                             seed=args.seed)

    cache_path = out / "hard_negative_cache.json"
    if not args.hard_negative:
        records = build_dpo_records(train_s, pair_cfg, scorer=None)
    elif state.num_processes == 1:
        print("[dpo] hard-negative 스코어링 시작 (--adapter 기준 3종 중 최고점 오답 선택)")
        scorer_fn, scorer_eng = _build_hard_negative_scorer(args)
        records = build_dpo_records(train_s, pair_cfg, scorer=scorer_fn)
        del scorer_eng
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[dpo] 스코어링용 엔진 해제 — 학습용 모델 로딩 시작")
    else:
        # DDP: rank0만 스코어링(비싼 forward pass, 80분+)해서 sample_id별 정렬된
        # 인접스와프 순위를 캐시로 남기고, 다른 rank는 그 캐시로 동일 records를
        # 재구성한다(augment_sample이 seed 고정 rng뿐이라 rank 간 결정적으로 동일).
        if args.include_random_rejected:
            raise SystemExit("--include-random-rejected는 DDP(멀티프로세스) hard-negative "
                             "캐시 경로에서 rng 스트림이 rank 간 어긋날 수 있어 미지원 — "
                             "단일 프로세스(torchrun 없이)로 돌리거나 이 옵션을 빼고 실행할 것")
        if state.is_main_process:
            print(f"[dpo] hard-negative 스코어링 시작 (rank0 전용, world_size={state.num_processes})")
            scorer_fn, scorer_eng = _build_hard_negative_scorer(args)
            records = build_dpo_records(train_s, pair_cfg, scorer=scorer_fn)
            cache: dict[str, list[int]] = {}
            for r in records:
                cache.setdefault(r["sample_id"], []).append(r["rejected_rank"])
            cache_path.write_text(json.dumps(cache))
            del scorer_eng
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[dpo] 스코어링용 엔진 해제, 캐시 {len(cache)}건 저장 → {cache_path}")
        state.wait_for_everyone()   # rank0의 GPU 해제까지 다른 rank가 대기
        if not state.is_main_process:
            cache = {sid: ranks for sid, ranks in json.loads(cache_path.read_text()).items()}
            records = build_dpo_records(train_s, pair_cfg, rejected_ranks_cache=cache)

    print(f"[dpo] 선호쌍 {len(records)}개 (샘플 {len(train_s)} × rejected_per_sample "
          f"{args.rejected_per_sample}{'+random' if args.include_random_rejected else ''})")

    model, processor = load_model_and_adapter(args)
    apply_pixel_budget(processor, max_pixels=args.max_pixels)
    collator = DPOCollator(processor)

    import transformers
    transformers.logging.set_verbosity_info()
    import importlib.util
    _tb = any(importlib.util.find_spec(m) is not None for m in ("tensorboard", "tensorboardX"))

    from transformers import Trainer, TrainingArguments
    targs = TrainingArguments(
        output_dir=str(out),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,          # 24GB/A100 규약 (score24 SFT와 동일)
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=6,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to=("tensorboard" if _tb else "none"), logging_dir=str(out / "tb"),
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )
    trainer_cls = _make_dpo_trainer_cls(Trainer, beta=args.beta)
    trainer = trainer_cls(model=model, args=targs, train_dataset=records, data_collator=collator)
    trainer.train(resume_from_checkpoint=args.resume or None)

    import torch
    if torch.cuda.is_available():
        print(f"[vram] peak allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB")

    trainer.save_model(str(out / "adapter_final"))
    processor.save_pretrained(str(out / "adapter_final"))
    print(f"[done] adapter → {out/'adapter_final'} — 추론: predict.py --adapter 로 사용")


if __name__ == "__main__":
    main()
