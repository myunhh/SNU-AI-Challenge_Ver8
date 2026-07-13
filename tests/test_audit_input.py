"""scripts/audit_input.py의 순수 함수 — 실제 프로세서/모델 불필요 (CPU 전용)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from audit_input import collapse_vision_runs, count_image_labels, count_timestamps, diff_facts  # noqa: E402


def test_count_timestamps():
    # 실측(Qwen3-VL-8B-Instruct, transformers 5.12.1): 타임스탬프 토큰은
    # "<0.5 seconds>" 형태로 디코드됨(공백+"seconds", 축약형 "s>"가 아님) —
    # 이 형식을 실제 프로세서로 직접 확인하지 않고 짐작해서 규약을 하드코딩하면
    # audit_input.py 전체가 타임스탬프 개수를 항상 0으로 오판하는 조용한 버그가 됨.
    text = "frames at <0.25 seconds> <1.25 second> <2.25 seconds> <3.25 seconds> shown"
    assert count_timestamps(text) == 4
    assert count_timestamps("no timestamps here") == 0


def test_count_image_labels():
    text = "Image 1: <img> Image 2: <img> Image 3: <img> Image 4: <img>"
    assert count_image_labels(text) == 4
    assert count_image_labels("Storyline: nothing here") == 0


def test_collapse_vision_runs_collapses_contiguous_runs():
    ids = [1, 2, 99, 99, 99, 3, 99, 99, 4]
    out = collapse_vision_runs(ids, vision_ids={99})
    assert out == "1 2 [VISION×3] 3 [VISION×2] 4"


def test_collapse_vision_runs_no_vision_ids_passthrough():
    ids = [1, 2, 3]
    assert collapse_vision_runs(ids, vision_ids=set()) == "1 2 3"


def test_diff_facts_ignores_recipe_and_label_check():
    prev = {"frame_count": 4, "recipe": {"model_id": "a"}, "label_check": {"ok": True}}
    curr = {"frame_count": 4, "recipe": {"model_id": "b"}, "label_check": {"ok": False}}
    assert diff_facts(prev, curr, allow=set()) == []


def test_diff_facts_flags_unallowed_structural_diff():
    prev = {"image_label_count": 4, "timestamp_count": 2}
    curr = {"image_label_count": 0, "timestamp_count": 4}
    diffs = diff_facts(prev, curr, allow={"timestamp_count"})
    assert len(diffs) == 1
    assert "image_label_count" in diffs[0]


def test_diff_facts_allows_listed_fields():
    prev = {"frame_count": 4, "vision_tokens_total": 880}
    curr = {"frame_count": 8, "vision_tokens_total": 880}
    assert diff_facts(prev, curr, allow={"frame_count"}) == []
