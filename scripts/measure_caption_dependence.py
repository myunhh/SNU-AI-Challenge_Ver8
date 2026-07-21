#!/usr/bin/env python
"""캡션 의존도 절단 진단 (Agent B 추천 #1) — 챔피언(Ver8 DPO ckpt600)이 실제로 캡션에서
얼마나 근거를 얻는지 정량화한다. train N건을 3조건으로 채점:
  normal  : 실제 캡션
  empty   : 빈 캡션("")
  swapped : 다른 무작위 샘플의 캡션(고정 시드로 결정적 매칭, 자기 자신과는 매칭 안 함)

image-only(=암기/이미지단서만으로 얼마나 맞히는지) 하한과 caption-swap이 오답을 유발하는
비율(=캡션을 실제로 읽고 쓰는지)을 동시에 본다. 학습 없음, 모델 가중치 무변경 — 순수 측정.

주의(Agent B가 이미 명시): train 전량 학습이라 암기가 "캡션 없이도 정답"을 만들어
의존도를 과소측정할 수 있음 — 보고 시 이 캐비엇을 반드시 포함할 것.

Usage (A100 또는 4090, --strategy score24는 어댑터 필수):
  python scripts/measure_caption_dependence.py \
      --csv data/train.csv --image-dir data/train \
      --adapter "runs/DPO-checkpoint-600" --n 200 --out runs/caption_dependence
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train.csv")
    ap.add_argument("--image-dir", default="data/train")
    ap.add_argument("--caption-col", default="Sentence")
    ap.add_argument("--model-id", default="/home/yhmin/model/hub/Qwen3-VL-32B-Instruct-bnb-4bit")
    ap.add_argument("--adapter", default="runs/DPO-checkpoint-600")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="runs/caption_dependence")
    args = ap.parse_args()

    from snuai import perm
    from snuai.data.sample import load_csv
    from snuai.infer.engine import EngineConfig, VLMEngine
    from snuai.infer.scorers import Score24Scorer
    from tqdm import tqdm

    all_samples = load_csv(args.csv, args.image_dir, caption_col=args.caption_col)
    all_samples = [s for s in all_samples if s.rank is not None]
    rng = random.Random(args.seed)
    samples = rng.sample(all_samples, min(args.n, len(all_samples)))
    print(f"[capdep] train {len(all_samples)}건 중 {len(samples)}건 표본(seed={args.seed})")

    # 결정적 스왑 매칭: 각 샘플에 자기 자신이 아닌 다른 표본의 캡션을 1:1로 배정
    swap_order = list(range(len(samples)))
    rng.shuffle(swap_order)
    for i in range(len(swap_order)):
        if swap_order[i] == i:  # 자기 자신 매칭이면 다음과 교환(홀수 길이도 안전)
            j = (i + 1) % len(swap_order)
            swap_order[i], swap_order[j] = swap_order[j], swap_order[i]
    swapped_captions = [samples[j].caption for j in swap_order]

    print(f"[capdep] loading {args.model_id} (adapter={args.adapter})")
    eng = VLMEngine(EngineConfig(
        model_id=args.model_id, four_bit=args.four_bit, adapter_path=args.adapter,
        max_pixels=args.max_pixels))
    print(f"[engine] attn={eng.attn_used} device={eng.device}")
    scorer = Score24Scorer(eng)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prog_path = out_dir / "progress.jsonl"
    done: dict[str, dict] = {}
    if prog_path.exists():
        for line in open(prog_path, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
        print(f"[resume] 기존 진행 {len(done)}건 재사용")

    conditions = ["normal", "empty", "swapped"]
    with open(prog_path, "a", encoding="utf-8") as f:
        for i, s in enumerate(tqdm(samples, desc="capdep", unit="샘플", mininterval=5.0)):
            if s.id in done:
                continue
            rec: dict = {"id": s.id, "true_rank": list(s.rank)}
            images = s.load_images()
            for cond, cap in [("normal", s.caption), ("empty", ""), ("swapped", swapped_captions[i])]:
                scores = scorer.scores(cap, images)
                pred_idx = int(scores.argmax())
                pred_rank = perm.perm_at(pred_idx)
                rec[cond] = {"pred": list(pred_rank), "correct": pred_rank == s.rank}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[s.id] = rec

    n = len(done)
    print(f"\n=== 결과 (n={n}) ===")
    for cond in conditions:
        em = sum(1 for r in done.values() if r[cond]["correct"])
        print(f"  {cond:8s}  EM {em}/{n} = {em/n:.4f}")

    normal_correct_ids = {rid for rid, r in done.items() if r["normal"]["correct"]}
    for cond in ["empty", "swapped"]:
        flipped = sum(1 for rid in normal_correct_ids if not done[rid][cond]["correct"])
        print(f"  normal에서 맞았다가 {cond}로 바꾸면 틀리는 비율: "
              f"{flipped}/{len(normal_correct_ids)} = {flipped/max(len(normal_correct_ids),1):.4f} "
              f"(캡션 실제 사용의 하한 추정치)")

    summary = {
        "n": n,
        "em": {c: sum(1 for r in done.values() if r[c]["correct"]) / n for c in conditions},
        "caveat": "train 전량학습이라 암기로 caption 없이도 맞힐 수 있어 의존도 과소측정 가능성 있음",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[capdep] 저장 -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
