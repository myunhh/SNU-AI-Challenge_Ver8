#!/usr/bin/env python
"""터미널 실시간 학습 대시보드 — TensorBoard 이벤트 파일을 plotext로 렌더.

SSH 터널 없이 서버 터미널에서 바로 학습 곡선을 본다 (TensorBoard 웹 UI 대체).

사용:
  python scripts/watch_train.py [tb_dir] [total_steps]
  python scripts/watch_train.py runs/sft8b_v2/tb 2000     # 기본값과 동일

종료: Ctrl+C (학습에는 영향 없음 — 읽기 전용)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def resolve_logdir(root: str) -> str | None:
    """root 아래에서 가장 최근 이벤트 파일이 있는 디렉터리를 찾는다.

    transformers 5.x는 logging_dir를 무시하고 <out>/runs/<타임스탬프>_<호스트>/에
    쓰므로(실측), 사용자는 out 디렉터리만 주면 되게 한다.
    """
    p = Path(root)
    if not p.exists():
        return None
    events = sorted(p.rglob("events.out.tfevents*"), key=lambda f: f.stat().st_mtime)
    return str(events[-1].parent) if events else None


def load_scalars(logdir: str) -> dict[str, list]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})  # 0 = 전부 로드
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value, e.wall_time) for e in ea.Scalars(tag)]
    return out


def eta_of(points: list, total_steps: int) -> str:
    """최근 구간의 스텝 속도로 남은 시간 추정."""
    if len(points) < 2:
        return "측정 중..."
    tail = points[-min(20, len(points)):]
    (s0, _, t0), (s1, _, t1) = tail[0], tail[-1]
    if s1 <= s0 or t1 <= t0:
        return "측정 중..."
    sec_per_step = (t1 - t0) / (s1 - s0)
    remain = (total_steps - s1) * sec_per_step
    done_at = datetime.now() + timedelta(seconds=remain)
    return (f"{sec_per_step:.1f}s/step · 남은시간 {timedelta(seconds=int(remain))} "
            f"· 완료예상 {done_at:%m/%d %H:%M}")


def render(scalars: dict, total_steps: int) -> None:
    import plotext as plt
    plt.clf(); plt.clt(); plt.cld()

    loss = scalars.get("train/loss", [])
    lr = scalars.get("train/learning_rate", [])
    gn = scalars.get("train/grad_norm", [])

    plt.subplots(2, 1)

    p1 = plt.subplot(1, 1)
    p1.theme("dark")
    if loss:
        steps, vals = [p[0] for p in loss], [p[1] for p in loss]
        p1.plot(steps, vals, marker="braille", color="orange", label="train/loss")
        # 이동평균(10포인트) — 노이즈 속 추세 확인용
        if len(vals) >= 10:
            ma = [sum(vals[max(0, i - 9):i + 1]) / len(vals[max(0, i - 9):i + 1])
                  for i in range(len(vals))]
            p1.plot(steps, ma, marker="braille", color="cyan", label="MA(10)")
    p1.title(f"loss  (최근 {loss[-1][1]:.4f} @ step {loss[-1][0]})" if loss else "loss 대기 중")
    p1.xlabel("step")

    p2 = plt.subplot(2, 1)
    p2.theme("dark")
    if gn:
        p2.plot([p[0] for p in gn], [p[1] for p in gn],
                marker="braille", color="red", label="grad_norm")
    if lr:
        # lr은 스케일이 달라 정규화해서 겹쳐 그림 (모양 확인용)
        mx = max(p[1] for p in lr) or 1.0
        gmx = max((p[1] for p in gn), default=1.0)
        p2.plot([p[0] for p in lr], [p[1] / mx * gmx for p in lr],
                marker="braille", color="green", label="lr(정규화)")
    p2.title(f"grad_norm (최근 {gn[-1][1]:.2f}) · lr {lr[-1][1]:.2e}" if gn and lr
             else "grad_norm/lr 대기 중")
    p2.xlabel("step")

    plt.show()

    if loss:
        cur = loss[-1][0]
        bar_w = 40
        filled = int(bar_w * cur / total_steps)
        print(f"\n  [{'█' * filled}{'░' * (bar_w - filled)}] "
              f"{cur}/{total_steps} steps ({cur / total_steps:.1%})")
        print(f"  {eta_of(loss, total_steps)}")
    print("  (15초마다 자동 갱신 · Ctrl+C로 종료 — 학습에 영향 없음)")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs/sft8b_v2"
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    while (logdir := resolve_logdir(root)) is None:
        print(f"{root} 아래 이벤트 파일 대기 중... (학습이 첫 로그를 쓰면 시작)", flush=True)
        time.sleep(10)
    while True:
        try:
            logdir = resolve_logdir(root) or logdir  # 재시작으로 새 run 디렉터리가 생기면 따라감
            render(load_scalars(logdir), total)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 — 이벤트 파일 쓰는 중 read race 등은 다음 턴에 회복
            print(f"[watch] 일시 오류(다음 갱신에 회복): {type(e).__name__}: {e}")
        time.sleep(15)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료 (학습은 계속 돌고 있음)")
