"""3090 속도·VRAM 벤치마크 — 샘플당 예산(하드 105초 / 목표 60초) 판정.

max_pixels 그리드별로 실측해 '819개 × 24h' 완주 가능성을 표로 출력한다.
어댑터가 있으면 score24(1 forward), 없으면 likelihood(24 forwards)로 측정.

예:
  python scripts/bench_3090.py --four-bit \
      --csv train.csv --image-dir images/train --n 12 \
      --pixel-grid 200704,401408,602112,1003520  # model-id 기본값: 로컬 8B

  # 32B-INT4 스모크 겸용 (노션: 가중치 19GB + KV 1.1GB + 활성화 ≈ 21.5-22.5GB)
  python scripts/bench_3090.py --model-id unsloth/Qwen3-VL-32B-Instruct-bnb-4bit --n 6
"""

from __future__ import annotations

import argparse
import statistics
import time

N_TEST = 819
HARD_LIMIT_SEC = 24 * 3600 / N_TEST          # ≈ 105.5s
TARGET_SEC = 60.0                             # 안전마진 목표(노션)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="/home/yhmin/model/hub/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--attn", default=None)
    ap.add_argument("--csv"); ap.add_argument("--image-dir")
    ap.add_argument("--caption-col", default="Caption")
    ap.add_argument("--n", type=int, default=12, help="측정 샘플 수")
    ap.add_argument("--pixel-grid", default="200704,401408,602112",
                    help="max_pixels 후보 (콤마 구분)")
    ap.add_argument("--video-mode", action="store_true")
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()

    import torch
    from snuai.data.sample import load_csv
    from snuai.infer.engine import EngineConfig, VLMEngine
    from snuai.infer.scorers import LikelihoodScorer, Score24Scorer
    from snuai.infer.tta import TTAConfig, tta_scores

    if args.csv:
        samples = load_csv(args.csv, args.image_dir, caption_col=args.caption_col)[: args.n]
    else:
        from snuai.data.synthetic import make_dataset
        samples = make_dataset(args.n, seed=0)
        print("⚠️ 합성 이미지 벤치 — 실데이터 해상도와 다르므로 참고치로만 사용")

    rows = []
    for mp in [int(x) for x in args.pixel_grid.split(",")]:
        eng = VLMEngine(EngineConfig(model_id=args.model_id, adapter_path=args.adapter,
                                     four_bit=args.four_bit, attn=args.attn, max_pixels=mp))
        scorer = (Score24Scorer(eng, video_mode=args.video_mode) if args.adapter
                  else LikelihoodScorer(eng, video_mode=args.video_mode))
        mode = type(scorer).__name__
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        times = []
        for s in samples:
            imgs = s.load_images()
            t0 = time.monotonic()
            tta_scores(imgs, lambda im: scorer.scores(s.caption, im),
                       TTAConfig(n_views=args.tta))
            times.append(time.monotonic() - t0)
        vram = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
        med = statistics.median(times)
        rows.append({"max_pixels": mp, "scorer": mode, "median_s": med,
                     "p95_s": sorted(times)[max(0, int(len(times) * 0.95) - 1)],
                     "vram_gib": vram,
                     "total_h_819": med * N_TEST / 3600,
                     "verdict": ("✅ 목표내" if med <= TARGET_SEC else
                                 "⚠️ 하드리밋내" if med <= HARD_LIMIT_SEC else "❌ 초과")})
        del eng
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n== {args.model_id} | attn 자동 | TTA={args.tta} | n={len(samples)} ==")
    print(f"{'max_pixels':>10} {'scorer':>18} {'median_s':>9} {'p95_s':>8} "
          f"{'VRAM_GiB':>9} {'819개_h':>8}  판정")
    for r in rows:
        print(f"{r['max_pixels']:>10} {r['scorer']:>18} {r['median_s']:>9.2f} "
              f"{r['p95_s']:>8.2f} {r['vram_gib']:>9.2f} {r['total_h_819']:>8.2f}  {r['verdict']}")
    print(f"\n기준: 샘플당 하드리밋 {HARD_LIMIT_SEC:.1f}s (24h/819), 목표 {TARGET_SEC:.0f}s (노션)")


if __name__ == "__main__":
    main()
