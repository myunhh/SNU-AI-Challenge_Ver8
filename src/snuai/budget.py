"""시간 예산 가드 — 24h 하드리밋 내 완주 보장 (샘플당 ~105초, 목표 60초).

정책:
  - per-sample 동적 예산 = 남은시간 × safety ÷ 남은 샘플 수
  - allow(cost)로 TTA 추가 뷰·캐스케이드 2단계 같은 선택 연산을 게이팅
  - 어떤 경우에도 Stage1 결과(또는 항등순열)는 있으므로 완주가 보장됨
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BudgetGuard:
    total_seconds: float
    n_samples: int
    safety: float = 0.85
    clock: callable = time.monotonic
    log_path: str | Path | None = None

    _t0: float = field(init=False)
    _sample_t0: float | None = field(default=None, init=False)
    done: int = field(default=0, init=False)
    timings: list[float] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._t0 = self.clock()

    # ---- 전체/샘플 시계 ----
    def remaining_total(self) -> float:
        return self.total_seconds - (self.clock() - self._t0)

    def per_sample_budget(self) -> float:
        left = max(self.n_samples - self.done, 1)
        return max(self.remaining_total() * self.safety / left, 0.0)

    def start_sample(self) -> None:
        self._sample_t0 = self.clock()

    def elapsed_sample(self) -> float:
        return 0.0 if self._sample_t0 is None else self.clock() - self._sample_t0

    def allow(self, est_cost_seconds: float) -> bool:
        """지금 샘플에서 est_cost짜리 선택 연산을 추가로 실행해도 되는가."""
        return self.elapsed_sample() + est_cost_seconds <= self.per_sample_budget()

    def end_sample(self, sample_id: str = "", extra: dict | None = None) -> float:
        dt = self.elapsed_sample()
        self.timings.append(dt)
        self.done += 1
        self._sample_t0 = None
        if self.log_path:
            rec = {"id": sample_id, "sec": round(dt, 3),
                   "remaining_total_sec": round(self.remaining_total(), 1), **(extra or {})}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return dt

    # ---- 리포트 ----
    def stats(self) -> dict:
        if not self.timings:
            return {"n": 0}
        arr = sorted(self.timings)
        n = len(arr)
        return {
            "n": n,
            "mean_sec": sum(arr) / n,
            "p50_sec": arr[n // 2],
            "p95_sec": arr[min(n - 1, int(n * 0.95))],
            "max_sec": arr[-1],
            "projected_total_h": (sum(arr) / n) * self.n_samples / 3600,
        }
