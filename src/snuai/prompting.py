"""프롬프트·메시지 빌더 — 학습(dataset)과 추론(engine)이 공유하는 유일한 출처.

학습과 추론의 프롬프트가 한 글자라도 다르면 SFT 효과가 깎이므로,
여기 함수만 사용하고 문자열을 직접 만들지 말 것.

이미지 표기: 프롬프트의 "Image 1..4" = 내부 슬롯 0..3 (1-based는 텍스트 경계뿐).
video_mode: 4프레임을 낱장 이미지 대신 비디오 채널로 입력하는 가설(노션 §10)용.
"""

from __future__ import annotations

from typing import Sequence

# ---------------------------------------------------------------------------
# 공통 조각
# ---------------------------------------------------------------------------

SYSTEM_TEXT = (
    "You are an expert at temporal reasoning over video frames. "
    "You are given a storyline and four frames sampled from one video, "
    "shown in shuffled order."
)

COUNTERFACTUAL_SUFFIX = (
    "\nBefore answering, silently verify: (1) Is your ordering grounded in the "
    "events described in the storyline? (2) Are you relying on superficial style "
    "cues such as zoom level, image quality, color tone, or motion blur instead "
    "of the actual content? If so, reconsider using content only."
)

ANSWER_PREFIX = "FINAL ANSWER: ["  # FSM 제약 디코딩 트리거 (이 뒤부터 숫자 강제)

SCORE24_INSTRUCTION = (
    "Rank the four images by the moment they occur in the storyline. "
    "Respond with exactly one letter (A-X): the code of the correct ranking."
)


def score24_legend() -> str:
    """A~X ↔ rank 수열 범례. 반드시 perm에서 파생 (손으로 쓰면 규약 사고).

    각 항목 "N=3142"의 자릿수 k = Image k의 시간순 순위(1=earliest). Kaggle
    Answer(rank+1)와 같은 인코딩이므로 legend 자체가 규약의 재검증이 된다.
    """
    from . import perm
    return ", ".join(f"{perm.LETTERS24[i]}={''.join(str(v + 1) for v in p)}"
                     for i, p in enumerate(perm.PERMS24))


# Ver3: A~X가 어떤 순열인지 모델에게 정의해 주는 명시 버전 (기본값).
# 기존 SCORE24_INSTRUCTION은 letter↔순열 대응을 SFT 그래디언트로만 암묵 학습시켰다.
SCORE24_INSTRUCTION_LEGEND = (
    "Assign each image its rank in time according to the storyline "
    "(1 = earliest, 4 = latest), forming a 4-digit rank sequence: "
    "digit k is the rank of Image k.\n"
    "Letter code: {legend}.\n"
    "Respond with exactly one letter (A-X): the code of the correct rank sequence."
)

LISTWISE_INSTRUCTION = (
    "Determine, for each image, its position in time (1 = earliest, 4 = latest). "
    "Think step by step about which event of the storyline each image shows. "
    f"Then give your answer as: {ANSWER_PREFIX}rank of Image 1, rank of Image 2, "
    "rank of Image 3, rank of Image 4]."
)


def _img(image) -> dict:
    return {"type": "image", "image": image}


def _txt(text: str) -> dict:
    return {"type": "text", "text": text}


VIDEO_FRAME_LABEL_NOTE = (
    "The video below contains exactly four frames in this fixed order: "
    "the 1st frame is Image 1, the 2nd is Image 2, the 3rd is Image 3, the 4th is Image 4. "
    "This is only a labeling convention for the frames as shown — it says nothing about "
    "their chronological order in the storyline, which you must determine separately."
)


def video_frame_label_note(dup_factor: int = 1) -> str:
    """dup_factor 인지형 프레임 라벨 안내문. dup_factor==1이면 VIDEO_FRAME_LABEL_NOTE와 byte-identical.

    Ver7 video_dup(R1 재도전): 각 입력 프레임을 dup_factor번 연속 복제해 비디오로
    넣을 때(temporal_patch_size=2의 인접쌍 병합이 항상 자기 자신과의 쌍이 되게 함),
    "몇 번째 원본 프레임 구간이 Image 몇인가"를 dup_factor에 맞춰 다시 설명해야 한다.
    """
    if dup_factor == 1:
        return VIDEO_FRAME_LABEL_NOTE
    total = 4 * dup_factor
    spans = ", ".join(
        f"frames {k * dup_factor + 1}-{(k + 1) * dup_factor} are Image {k + 1}"
        for k in range(4))
    return (
        f"The video below contains {total} frames: each of the 4 input frames is "
        f"repeated {dup_factor} times consecutively, in this fixed order — {spans}. "
        "This is only a labeling convention for the frames as shown — it says nothing "
        "about their chronological order in the storyline, which you must determine "
        "separately."
    )


def media_content(images: Sequence, video_mode: bool = False, fps: float = 1.0,
                  dup_factor: int = 1) -> list[dict]:
    """4프레임 → 메시지 콘텐츠. video_mode=True면 비디오 1개(프레임 시퀀스)로 입력.

    ⚠️ video_mode에서 프레임 순서 = 슬롯 순서 그대로 (셔플된 채 비디오로 묶음).
       rank 의미는 이미지 모드와 완전히 동일하다.

    ⚠️ image 모드와 달리 video 모드는 프레임마다 "Image k:" 텍스트 라벨을 붙일 수
       없다(단일 비디오 블록이라 프레임 사이에 텍스트를 끼워 넣지 못함) — 대신
       video_frame_label_note로 "몇 번째 프레임 = Image 몇"을 텍스트로 명시한다.
       이게 없으면 score24(단일 토큰, CoT 없음)가 프레임-Image 대응을 추론할 단서가
       전혀 없어 확신도(margin)가 전반적으로 낮아지는 것으로 실측됨(Ver2 1차 학습).

    dup_factor>1 (Ver7 video_dup, R1 재도전): 각 프레임을 dup_factor번 연속 복제해
    temporal_patch_size=2의 병합쌍이 항상 자기 자신과의 쌍이 되게 한다(무관한
    인접 프레임끼리 병합돼 정보가 오염되는 Ver2 −16.9pp 주범을 원천 차단).
    dup_factor는 1 또는 짝수만 유효(predict.py/train_sft.py CLI 가드로 강제) —
    홀수면 병합쌍 경계가 프레임 블록과 어긋나 일부 쌍이 여전히 교차 오염된다.
    """
    if video_mode:
        frames = [im for im in images for _ in range(dup_factor)] if dup_factor > 1 else list(images)
        return [_txt(video_frame_label_note(dup_factor)),
                {"type": "video", "video": frames, "fps": fps}]
    out: list[dict] = []
    for k, im in enumerate(images):
        out.append(_txt(f"Image {k + 1}:"))
        out.append(_img(im))
    return out


# ---------------------------------------------------------------------------
# 1) score24 — 단일 토큰 24-way (메인 경로)
# ---------------------------------------------------------------------------

def build_score24_messages(caption: str, images: Sequence, video_mode: bool = False,
                           counterfactual: bool = False, legend: bool = True,
                           dup_factor: int = 1) -> list[dict]:
    base = (SCORE24_INSTRUCTION_LEGEND.format(legend=score24_legend())
            if legend else SCORE24_INSTRUCTION)
    instr = base + (COUNTERFACTUAL_SUFFIX if counterfactual else "")
    return [
        {"role": "system", "content": [_txt(SYSTEM_TEXT)]},
        {"role": "user", "content": [
            _txt(f"Storyline: {caption}\n"),
            *media_content(images, video_mode, dup_factor=dup_factor),
            _txt("\n" + instr),
        ]},
    ]


# ---------------------------------------------------------------------------
# 2) listwise CoT — 생성 모드 (FSM 제약 디코딩과 세트)
# ---------------------------------------------------------------------------

def build_listwise_cot_messages(caption: str, images: Sequence, video_mode: bool = False,
                                counterfactual: bool = True, dup_factor: int = 1) -> list[dict]:
    instr = LISTWISE_INSTRUCTION + (COUNTERFACTUAL_SUFFIX if counterfactual else "")
    return [
        {"role": "system", "content": [_txt(SYSTEM_TEXT)]},
        {"role": "user", "content": [
            _txt(f"Storyline: {caption}\n"),
            *media_content(images, video_mode, dup_factor=dup_factor),
            _txt("\n" + instr),
        ]},
    ]


# ---------------------------------------------------------------------------
# 3) pairwise — 캐스케이드 2단계 재검 (A/B 단일 토큰 로짓)
# ---------------------------------------------------------------------------

PAIRWISE_INSTRUCTION = (
    "According to the storyline, which image shows the EARLIER moment? "
    "Respond with exactly one letter: A or B."
)


def build_pairwise_messages(caption: str, image_a, image_b) -> list[dict]:
    """A/B 두 이미지 중 시간상 앞선 쪽을 묻는다. 플립 평균은 pairwise.py가 담당."""
    return [
        {"role": "system", "content": [_txt(SYSTEM_TEXT)]},
        {"role": "user", "content": [
            _txt(f"Storyline: {caption}\n"),
            _txt("Image A:"), _img(image_a),
            _txt("Image B:"), _img(image_b),
            _txt("\n" + PAIRWISE_INSTRUCTION),
        ]},
    ]


# ---------------------------------------------------------------------------
# 4) frame-event 정렬 — decompose-and-match (Yes/No 단일 토큰 로짓)
# ---------------------------------------------------------------------------

ALIGN_INSTRUCTION = 'Does this frame show the event: "{event}"? Respond with exactly Yes or No.'


def build_frame_event_messages(image, event: str) -> list[dict]:
    return [
        {"role": "user", "content": [
            _img(image),
            _txt("\n" + ALIGN_INSTRUCTION.format(event=event)),
        ]},
    ]


# ---------------------------------------------------------------------------
# 5) describe-then-reason — 멀티턴 CoT (보고서 독창성 소재)
# ---------------------------------------------------------------------------

def build_describe_messages(image) -> list[dict]:
    return [{"role": "user", "content": [
        _img(image),
        _txt("Describe what is happening in this frame in 1-2 short sentences. "
             "Focus on actions and object positions, not style."),
    ]}]


def build_reason_from_descriptions_messages(caption: str, descriptions: Sequence[str],
                                            counterfactual: bool = True) -> list[dict]:
    lines = "\n".join(f"Image {k + 1}: {d}" for k, d in enumerate(descriptions))
    instr = LISTWISE_INSTRUCTION + (COUNTERFACTUAL_SUFFIX if counterfactual else "")
    return [{"role": "user", "content": [_txt(
        f"Storyline: {caption}\n\nFrame descriptions:\n{lines}\n\n{instr}"
    )]}]


# ---------------------------------------------------------------------------
# 6) 순서 검증 보조 태스크 (Shuffle & Learn 스타일, 학습 믹싱용)
# ---------------------------------------------------------------------------

VERIFY_INSTRUCTION = (
    "The four images are claimed to be in correct chronological order according "
    "to the storyline. Is this claim correct? Respond with exactly Yes or No."
)


def build_verify_messages(caption: str, images: Sequence, video_mode: bool = False,
                          dup_factor: int = 1) -> list[dict]:
    return [
        {"role": "system", "content": [_txt(SYSTEM_TEXT)]},
        {"role": "user", "content": [
            _txt(f"Storyline: {caption}\n"),
            *media_content(images, video_mode, dup_factor=dup_factor),
            _txt("\n" + VERIFY_INSTRUCTION),
        ]},
    ]


# ---------------------------------------------------------------------------
# 유틸 — 메시지에서 미디어 추출 (processor 호출용)
# ---------------------------------------------------------------------------

def extract_media(messages: list[dict]) -> tuple[list, list]:
    """messages → (images, videos). engine이 processor에 넘길 때 사용."""
    images, videos = [], []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image":
                images.append(item["image"])
            elif item.get("type") == "video":
                videos.append(item["video"])
    return images, videos


def normalize_videos(videos: list) -> list | None:
    """PIL 프레임 리스트 → (T,H,W,C) ndarray 목록 (video processor 최대 호환).

    실데이터는 한 샘플 안에서도 프레임 해상도가 제각각이라(크기 불일치로 stack 불가)
    첫 프레임 크기로 통일한다 — 비디오 텐서는 T×H×W 균일이 필수. 최종 해상도는
    어차피 video processor가 픽셀 예산(max_pixels)으로 다시 리사이즈한다.
    """
    if not videos:
        return None
    import numpy as np
    from PIL import Image
    out = []
    for v in videos:
        if isinstance(v, list):
            frames = [f.convert("RGB") if hasattr(f, "convert")
                      else Image.fromarray(np.asarray(f)).convert("RGB") for f in v]
            w, h = frames[0].size
            frames = [f if f.size == (w, h) else f.resize((w, h), Image.BICUBIC)
                      for f in frames]
            out.append(np.stack([np.asarray(f) for f in frames]))
        else:
            out.append(v)
    return out


VIDEO_FPS = 1.0  # R1: 프레임 간 1초 간격 가정 → <1.0 seconds>류 타임스탬프 토큰의 기준


def video_metadata_of(videos: list | None, fps: float = VIDEO_FPS) -> list | None:
    """normalize_videos 결과에 대응하는 video_metadata (R1 타임스탬프 필수 요소).

    metadata 없이 비디오를 넣으면 Qwen3-VL processor가 fps=24로 가정해 4프레임에
    0.0/0.1초짜리 퇴화 타임스탬프가 붙는다(실측) — R1 시간축 인코딩이 무력화되므로
    비디오 입력 시 반드시 함께 전달한다. 학습·추론 모두 call_processor를 경유할 것.

    ⚠️ `frames_indices`가 없으면 Qwen3-VL processor가 `do_sample_frames=True`
    기본값으로 `metadata.fps` 기준 프레임을 **재샘플링**한다(Ver7 video_dup 검증 중
    실측: dup_factor=2로 8프레임을 넣어도 metadata.fps=8.0이면 내부적으로 4프레임
    으로 재계산돼 dup 복제가 조용히 무효화되고 Ver2와 같은 크기의 붕괴가 재현됨).
    `frames_indices`를 명시하면 이 재샘플링의 인덱스 소스가 되어 `call_processor`의
    `do_sample_frames=False`와 세트로 "넣은 프레임 그대로 쓴다"를 보장한다.
    """
    if not videos:
        return None
    return [{"fps": fps, "total_num_frames": len(v), "duration": len(v) / fps,
            "frames_indices": list(range(len(v)))} for v in videos]


_META_UNSUPPORTED_WARNED = False
_SAMPLE_FRAMES_UNSUPPORTED_WARNED = False


def call_processor(processor, text: list[str], images: list | None, videos: list | None, **kw):
    """processor 호출의 유일한 창구 — 비디오 정규화 + video_metadata 주입 + 구버전 폴백.

    학습(SFTCollator)·추론(VLMEngine)·CoT 길이 프로브가 전부 이 함수를 써야
    타임스탬프 토큰 유무가 갈리지 않는다(프롬프트 길이·내용 일치 원칙).

    `do_sample_frames=False`: video_metadata_of의 docstring 참고 — 이게 없으면
    넣은 프레임 수와 무관하게 processor가 내부 fps 기준으로 재샘플링해버려
    video_dup(프레임 복제)이 조용히 무효화된다. 3단계 폴백(신버전→video_metadata만
    지원하는 버전→둘 다 미지원)으로 구버전 transformers도 안전하게 처리.
    """
    vids = normalize_videos(videos or [])
    meta = video_metadata_of(vids)
    if meta is not None:
        try:
            return processor(text=text, images=images or None, videos=vids,
                             video_metadata=meta, do_sample_frames=False, **kw)
        except TypeError:
            global _SAMPLE_FRAMES_UNSUPPORTED_WARNED
            if not _SAMPLE_FRAMES_UNSUPPORTED_WARNED:
                import warnings
                warnings.warn("이 transformers 버전은 do_sample_frames를 지원하지 않음 — "
                              "video_dup(프레임 복제)이 내부 재샘플링으로 무효화될 수 있음")
                _SAMPLE_FRAMES_UNSUPPORTED_WARNED = True
            try:
                return processor(text=text, images=images or None, videos=vids,
                                 video_metadata=meta, **kw)
            except TypeError:
                global _META_UNSUPPORTED_WARNED
                if not _META_UNSUPPORTED_WARNED:
                    import warnings
                    warnings.warn("이 transformers 버전은 video_metadata를 지원하지 않음 — "
                                  "타임스탬프가 fps=24 기준으로 붙어 R1 효과가 약화됨")
                    _META_UNSUPPORTED_WARNED = True
    return processor(text=text, images=images or None, videos=vids, **kw)
