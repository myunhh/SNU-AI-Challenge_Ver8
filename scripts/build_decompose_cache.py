#!/usr/bin/env python
"""Phase 0: 모델 기반 캡션 분해 일괄 실행 → JSONL 캐시 (TODO §3).

⚠️ 추론 전용 — 생성 모델의 분해 텍스트를 학습 데이터에 넣는 것은
   '생성형 모델 증강 금지' 조항 위반 소지 (학습엔 split_events_rule만).

같은 VLM의 텍스트 전용 호출 = 단일 모델 반복 호출(CoT 범주, 규정 허용).
재개형: 기존 캐시에 있는 id는 건너뜀. base 모델 사용(분해는 일반 능력이라
score24 특화 어댑터 불필요).

사용처: MatchScorer(events_fn 주입)·CoT→재스코어 노트(§4). 로더는
snuai.data.decompose.load_events_cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train.csv")
    ap.add_argument("--image-dir", default="data/train")
    ap.add_argument("--holdout-val", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="preproc/events_holdout.jsonl")
    ap.add_argument("--model-id", default="/home/yhmin/model/hub/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max-events", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from snuai.data.decompose import build_decompose_messages, parse_numbered_events
    from snuai.data.sample import load_csv
    from snuai.data.split import split_samples
    from snuai.infer.engine import EngineConfig, VLMEngine

    samples = load_csv(args.csv, args.image_dir)
    if args.holdout_val:
        _, samples = split_samples(samples, val_frac=args.val_frac)
    if args.limit:
        samples = samples[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
        print(f"[decompose] 재개: 기존 {len(done)}건 스킵")

    todo = [s for s in samples if s.id not in done]
    print(f"[decompose] 대상 {len(todo)}건 (총 {len(samples)})")
    if not todo:
        return

    eng = VLMEngine(EngineConfig(model_id=args.model_id))
    with open(out, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo):
            msgs = build_decompose_messages(s.caption, max_events=args.max_events)
            txt, _, _ = eng.generate_text(msgs, max_new_tokens=args.max_new_tokens)
            events = parse_numbered_events(txt, s.caption, max_events=args.max_events)
            f.write(json.dumps({"id": s.id, "caption": s.caption, "events": events,
                                "raw": txt[:400]}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"[decompose] {i + 1}/{len(todo)}")
    print(f"[decompose] 완료 → {out}")


if __name__ == "__main__":
    main()
