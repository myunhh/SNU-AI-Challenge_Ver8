"""video_dup (R1 재도전) — 프레임 복제 순서·라벨 안내문·픽셀 예산 분리·
do_sample_frames 방지·CLI 가드. 전부 CPU/모델 불필요.
"""

import pytest
from PIL import Image

from snuai.infer.engine import apply_pixel_budget
from snuai.prompting import (VIDEO_FRAME_LABEL_NOTE, build_score24_messages,
                             call_processor, media_content, video_frame_label_note,
                             video_metadata_of)


def _marker_image(k: int) -> Image.Image:
    return Image.new("RGB", (8, 8), color=(k, 0, 0))


def _marker_of(im: Image.Image) -> int:
    return im.getpixel((0, 0))[0]


# ---------------------------------------------------------------------------
# media_content — dup 순서
# ---------------------------------------------------------------------------

def test_media_content_dup_factor_2_repeats_each_frame_consecutively():
    imgs = [_marker_image(k) for k in range(4)]
    content = media_content(imgs, video_mode=True, dup_factor=2)
    video_item = next(c for c in content if c["type"] == "video")
    frames = video_item["video"]
    assert [_marker_of(f) for f in frames] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_media_content_dup_factor_1_unchanged():
    imgs = [_marker_image(k) for k in range(4)]
    content = media_content(imgs, video_mode=True, dup_factor=1)
    video_item = next(c for c in content if c["type"] == "video")
    assert [_marker_of(f) for f in video_item["video"]] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 프레임 라벨 안내문
# ---------------------------------------------------------------------------

def test_video_frame_label_note_dup1_byte_identical_to_existing_constant():
    assert video_frame_label_note(1) == VIDEO_FRAME_LABEL_NOTE
    assert video_frame_label_note() == VIDEO_FRAME_LABEL_NOTE


def test_video_frame_label_note_dup2_spans_correct():
    note = video_frame_label_note(2)
    assert "frames 1-2 are Image 1" in note
    assert "frames 3-4 are Image 2" in note
    assert "frames 5-6 are Image 3" in note
    assert "frames 7-8 are Image 4" in note
    assert "8 frames" in note


# ---------------------------------------------------------------------------
# build_score24_messages — dup_factor=1 하위호환
# ---------------------------------------------------------------------------

def test_build_score24_messages_dup_factor_1_byte_identical():
    caption, imgs = "A dog runs.", [_marker_image(k) for k in range(4)]
    default_call = build_score24_messages(caption, imgs, video_mode=True)
    explicit_dup1 = build_score24_messages(caption, imgs, video_mode=True, dup_factor=1)
    assert default_call == explicit_dup1


def test_build_score24_messages_dup_factor_2_doubles_video_frames():
    caption, imgs = "A dog runs.", [_marker_image(k) for k in range(4)]
    msgs = build_score24_messages(caption, imgs, video_mode=True, dup_factor=2)
    video_item = next(c for c in msgs[1]["content"] if c["type"] == "video")
    assert len(video_item["video"]) == 8


# ---------------------------------------------------------------------------
# video_metadata_of — frames_indices (do_sample_frames 재샘플링 무효화 방지)
# ---------------------------------------------------------------------------

def test_video_metadata_of_includes_frames_indices():
    videos = [list(range(8))]  # dup_factor=2 상당의 8프레임짜리 더미
    meta = video_metadata_of(videos)
    assert meta[0]["frames_indices"] == list(range(8))
    assert meta[0]["total_num_frames"] == 8


def test_call_processor_passes_do_sample_frames_false_when_video_present():
    captured = {}

    def fake_processor(**kw):
        captured.update(kw)
        return kw

    frames = [_marker_image(k) for k in range(8)]
    call_processor(fake_processor, ["dummy text"], None, [frames])
    assert captured.get("do_sample_frames") is False
    assert "video_metadata" in captured
    assert captured["video_metadata"][0]["frames_indices"] == list(range(8))


def test_call_processor_falls_back_when_do_sample_frames_unsupported():
    """구버전 폴백: do_sample_frames 미지원이면 video_metadata만으로 재시도."""
    calls = []

    def fake_processor(**kw):
        calls.append(kw)
        if "do_sample_frames" in kw:
            raise TypeError("unexpected keyword argument 'do_sample_frames'")
        return kw

    frames = [_marker_image(k) for k in range(4)]
    with pytest.warns(UserWarning, match="do_sample_frames"):
        result = call_processor(fake_processor, ["dummy text"], None, [frames])
    assert len(calls) == 2
    assert "do_sample_frames" not in result
    assert "video_metadata" in result


# ---------------------------------------------------------------------------
# apply_pixel_budget — video 전용 예산 분리
# ---------------------------------------------------------------------------

class _FakeSubProcessor:
    def __init__(self):
        # 실제 Qwen-VL processor의 size dict는 기본값이 이미 채워져 있음(placeholder
        # int) — apply_pixel_budget은 "이미 값이 있는 키만" 덮어쓰므로 None으로
        # 초기화하면 안 됨(실측: None이면 조용히 스킵되고 경고만 남음)
        self.size = {"longest_edge": 1000000, "shortest_edge": 100}


class _FakeProcessor:
    def __init__(self):
        self.image_processor = _FakeSubProcessor()
        self.video_processor = _FakeSubProcessor()


def test_apply_pixel_budget_splits_video_max_pixels():
    proc = _FakeProcessor()
    apply_pixel_budget(proc, max_pixels=602112, video_max_pixels=4816896)
    assert proc.image_processor.size["longest_edge"] == 602112
    assert proc.video_processor.size["longest_edge"] == 4816896


def test_apply_pixel_budget_video_falls_back_to_max_pixels_when_none():
    proc = _FakeProcessor()
    apply_pixel_budget(proc, max_pixels=602112, video_max_pixels=None)
    assert proc.image_processor.size["longest_edge"] == 602112
    assert proc.video_processor.size["longest_edge"] == 602112  # 회귀 방지: 기존 동작


# ---------------------------------------------------------------------------
# CLI 가드 — predict.py / train_sft.py
# ---------------------------------------------------------------------------

def test_predict_guard_rejects_dup_without_video_mode():
    from snuai.infer.predict import main
    with pytest.raises(SystemExit, match="video-mode"):
        main(["--video-dup-factor", "2", "--synthetic", "4", "--strategy", "dummy"])


def test_predict_guard_rejects_odd_dup_factor():
    from snuai.infer.predict import main
    with pytest.raises(SystemExit, match="짝수"):
        main(["--video-mode", "--video-dup-factor", "3",
             "--synthetic", "4", "--strategy", "dummy"])


def test_train_sft_guard_rejects_dup_without_video_mode(tmp_path):
    from snuai.train.train_sft import main
    out_dir = tmp_path / "run"
    with pytest.raises(SystemExit, match="video-mode"):
        main(["--csv", "x.csv", "--image-dir", "x", "--out", str(out_dir),
             "--video-dup-factor", "2"])
    assert not out_dir.exists()  # 가드가 out.mkdir보다 먼저 걸려야 함


def test_train_sft_guard_rejects_odd_dup_factor(tmp_path):
    from snuai.train.train_sft import main
    out_dir = tmp_path / "run"
    with pytest.raises(SystemExit, match="짝수"):
        main(["--csv", "x.csv", "--image-dir", "x", "--out", str(out_dir),
             "--video-mode", "--video-dup-factor", "3"])
    assert not out_dir.exists()
