#!/usr/bin/env python3
"""홀드아웃 추론 체인 현황 상태바 — stdlib만 사용 (bare python3으로 실행 가능).

사용:  watch -n 30 python3 scripts/watch_infer.py
"""
import json
import time
from pathlib import Path

RUNS = ["test_dpo_ckpt400", "test_dpo_ckpt600", "test_dpo_ckpt800",
        "test_dpo_ckpt1000"]
TOTAL = 819  # test 건수
BAR = 40

root = Path(__file__).resolve().parent.parent / "runs"
now = time.time()
for name in RUNS:
    out = root / name
    prog = out / "progress.jsonl"
    n = 0
    mtime = None
    if prog.exists():
        with open(prog, "rb") as f:
            n = sum(1 for _ in f)
        mtime = prog.stat().st_mtime
    filled = int(BAR * n / TOTAL)
    bar = "█" * filled + "░" * (BAR - filled)
    if (out / "report.json").exists():
        rep = json.load(open(out / "report.json"))
        acc = rep.get("accuracy")
        extra = f"완료 ✓ EM={acc:.4f}" if acc is not None else "완료 ✓"
    elif mtime is None:
        extra = "대기"
    elif now - mtime > 300:
        extra = f"정지? (마지막 기록 {int((now - mtime) / 60)}분 전)"
    else:
        extra = "진행 중"
    print(f"{name:<17} [{bar}] {n:>4}/{TOTAL} ({n / TOTAL * 100:5.1f}%)  {extra}")
