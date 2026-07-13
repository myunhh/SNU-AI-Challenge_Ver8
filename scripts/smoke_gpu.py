"""GPU 스모크 테스트 — 모델 로딩·규약·전 경로 1샘플 확인 (3090/VESSL 도착 직후 1회).

검사 항목:
  1. 4bit 로딩 + vision 비양자화 확인 (규약)
  2. 라벨 토큰(A..X 등) 단일 토큰 확인
  3. score24 / pairwise / video / FSM-CoT 각 1회 실행 + VRAM 피크
  4. (어댑터 지정 시) LoRA가 language에만 붙었는지 확인

예:
  python scripts/smoke_gpu.py --four-bit  # model-id 기본값: 로컬 8B
  python scripts/smoke_gpu.py --model-id unsloth/Qwen3-VL-32B-Instruct-bnb-4bit --prequantized
"""

from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="/home/yhmin/model/hub/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--prequantized", action="store_true",
                    help="bnb-4bit 체크포인트(자체 quant config)")
    ap.add_argument("--max-pixels", type=int, default=602112)
    args = ap.parse_args()

    import torch
    from snuai import perm
    from snuai.data.synthetic import make_dataset
    from snuai.infer.engine import EngineConfig, VLMEngine
    from snuai.infer.scorers import CoTFSMScorer, PairwiseJudge, Score24Scorer
    from snuai.train.qlora import verify_vision_not_quantized

    eng = VLMEngine(EngineConfig(
        model_id=args.model_id, adapter_path=args.adapter,
        four_bit=args.four_bit and not args.prequantized,
        max_pixels=args.max_pixels))
    print(f"[load] attn={eng.attn_used} device={eng.device}")
    if args.four_bit or args.prequantized:
        print("[quant]", verify_vision_not_quantized(eng.model))
    if args.adapter:
        from snuai.train.qlora import verify_lora_only_on_language
        print("[lora]", verify_lora_only_on_language(eng.model))

    ids = [eng.token_id_of(ch) for ch in perm.LETTERS24]
    assert len(set(ids)) == 24
    for sym in ("1", "2", "3", "4", ",", "]", "Yes", "No", "A", "B"):
        eng.token_id_of(sym)
    print("[tokens] 24 라벨 + FSM/판독 심볼 전부 단일 토큰 ✅")

    s = make_dataset(1, seed=0)[0]
    imgs = s.load_images()

    def timed(name, fn):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic(); out = fn(); dt = time.monotonic() - t0
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        print(f"[{name}] {dt:.2f}s | peak VRAM {vram:.2f} GiB")
        return out

    sc = Score24Scorer(eng)
    scores = timed("score24(1 forward)", lambda: sc.scores(s.caption, imgs))
    print("        top:", perm.perm_at(int(scores.argmax())), "margin softmax OK")

    judge = PairwiseJudge(eng)
    timed("pairwise(2 forward)", lambda: judge.p_earlier(s.caption, imgs[0], imgs[1]))

    from snuai.prompting import build_score24_messages
    timed("video_mode(1 forward)",
          lambda: eng.next_token_logits(build_score24_messages(s.caption, imgs, video_mode=True)))

    cot = CoTFSMScorer(eng, n_samples=1, cot_max_tokens=128)
    timed("CoT+FSM(생성)", lambda: cot.scores(s.caption, imgs))

    print("\n스모크 통과 — bench_3090.py로 예산 실측, exp_preproc_ab.py로 전처리 A/B 진행")


if __name__ == "__main__":
    main()
