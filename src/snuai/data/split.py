"""로컬 검증셋 분할 — τ·λ 캘리브레이션과 모든 A/B의 기준.

⚠️ 규정(데이터 누수): 캐스케이드 τ, TTA 채택, 전처리 A/B 등 모든 하이퍼파라미터는
   반드시 train에서 쪼갠 이 홀드아웃으로만 결정한다. test 분포 참조 금지.

id의 md5 해시 기반 → 코드 실행 순서·머신과 무관하게 항상 같은 분할(재현성).
"""

from __future__ import annotations

import hashlib
from typing import Sequence, TypeVar

T = TypeVar("T")


def stable_bucket(sample_id: str, salt: str = "snuai-v1") -> float:
    """id → [0,1) 결정적 버킷."""
    h = hashlib.md5(f"{salt}:{sample_id}".encode()).hexdigest()
    return int(h[:12], 16) / 16**12


def split_ids(ids: Sequence[str], val_frac: float = 0.1, salt: str = "snuai-v1"
              ) -> tuple[list[str], list[str]]:
    """(train_ids, val_ids). 같은 인자면 언제 어디서든 같은 결과."""
    tr, va = [], []
    for sid in ids:
        (va if stable_bucket(sid, salt) < val_frac else tr).append(sid)
    return tr, va


def split_samples(samples: Sequence[T], val_frac: float = 0.1, salt: str = "snuai-v1",
                  id_attr: str = "id") -> tuple[list[T], list[T]]:
    tr, va = [], []
    for s in samples:
        (va if stable_bucket(getattr(s, id_attr), salt) < val_frac else tr).append(s)
    return tr, va
