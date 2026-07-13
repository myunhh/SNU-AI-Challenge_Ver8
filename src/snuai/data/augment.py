"""학습 전용 증강 (TRAIN-ONLY).

⚠️ 추론에서 이 모듈을 쓰면 안 된다 — 추론의 입력 변형은 tta.py(의도적 TTA)와
   importance.py(결정적 전처리)만 사용한다. 학습/추론 전처리 분리 원칙:

   - 확률적 증강(순열 셔플·색상·해상도 지터) = 학습 전용. 표준 관행
     (ImageNet 이후 모든 학습이 train=랜덤증강 / eval=결정적 전처리).
   - 결정적 변환(중요도 다운스케일, max_pixels)은 학습·추론 일치가 기본값.
     불일치를 쓰려면 ①학습 증강이 추론 변형을 커버(res_aug/blur_aug)하거나
     ②홀드아웃 A/B로 열화 없음을 실측한 뒤 채택 (scripts/exp_preproc_ab.py).

순열 증강 라벨 재계산은 perm.shuffled_rank()가 유일한 규약이다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .. import perm
from .sample import Sample


@dataclass(frozen=True)
class AugmentConfig:
    # --- 순열 증강 (샘플당 최대 24배 효과, 노션 §10) ---
    perm_mode: str = "uniform"          # "uniform" | "identity_ratio" | "off"
    identity_ratio: float = 0.0         # perm_mode="identity_ratio": 라벨을 항등으로 강제할 확률
    # --- 색상 증강 (shortcut bias 완화, 노션 §10 색상 증강) ---
    grayscale_p: float = 0.0
    jitter_p: float = 0.0
    jitter_strength: float = 0.2        # brightness/contrast/saturation ±strength
    color_scope: str = "sample"         # "sample"=4장 동일 변형(기본) | "frame"=프레임별 독립
    # --- 해상도·블러 지터 (추론측 다운스케일 전처리를 학습 분포로 커버) ---
    res_aug_p: float = 0.0
    res_aug_range: tuple[float, float] = (0.5, 1.0)
    blur_p: float = 0.0
    blur_radius_range: tuple[float, float] = (0.5, 2.0)


def identity_ratio_for_target(target_frac: float) -> float:
    """최종 항등순열 라벨 비율이 target_frac이 되도록 identity_ratio를 역산.

    p_final = p + (1-p)/24  ⇒  p = (target - 1/24) / (1 - 1/24)
    train 원본 항등 비율 15.5%를 유지하려면 identity_ratio_for_target(0.155)≈0.118.
    """
    base = 1.0 / 24.0
    if not (base <= target_frac <= 1.0):
        raise ValueError(f"target_frac은 [{base:.4f}, 1] 범위여야 함: {target_frac}")
    return (target_frac - base) / (1.0 - base)


def draw_shuffle(rank: perm.Perm, cfg: AugmentConfig, rng: random.Random) -> perm.Perm:
    """설정에 따라 이번 샘플에 적용할 셔플 s를 추출."""
    if cfg.perm_mode == "off":
        return perm.IDENTITY
    if cfg.perm_mode == "identity_ratio" and rng.random() < cfg.identity_ratio:
        return perm.shuffle_for_target_rank(rank, perm.IDENTITY)
    if cfg.perm_mode in ("uniform", "identity_ratio"):
        return perm.random_shuffle(rng)
    raise ValueError(f"알 수 없는 perm_mode: {cfg.perm_mode}")


def _jitter_params(cfg: AugmentConfig, rng: random.Random) -> dict:
    s = cfg.jitter_strength
    return {
        "brightness": 1.0 + rng.uniform(-s, s),
        "contrast": 1.0 + rng.uniform(-s, s),
        "saturation": 1.0 + rng.uniform(-s, s),
    }


def _apply_color(img: Image.Image, params: dict) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(params["brightness"])
    img = ImageEnhance.Contrast(img).enhance(params["contrast"])
    img = ImageEnhance.Color(img).enhance(params["saturation"])
    return img


def _apply_res_blur(img: Image.Image, scale: float | None, radius: float | None) -> Image.Image:
    if scale is not None and scale < 1.0:
        w, h = img.size
        small = img.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
        img = small.resize((w, h), Image.BILINEAR)
    if radius is not None and radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return img


def augment_sample(sample: Sample, cfg: AugmentConfig, rng: random.Random) -> Sample:
    """학습 샘플 1개에 증강 적용 — 새 Sample 반환(원본 불변), 라벨 자동 재계산.

    데이터셋 __getitem__에서 매 호출마다 실행(온더플라이) → 에폭마다 다른 셔플.
    """
    if sample.rank is None:
        raise ValueError("rank 없는(test) 샘플에는 학습 증강을 적용할 수 없음")
    images = sample.load_images()

    # 1) 순열 증강 + 라벨 재계산 (규약: perm.py)
    s = draw_shuffle(sample.rank, cfg, rng)
    images = perm.apply_shuffle(images, s)
    new_rank = perm.shuffled_rank(sample.rank, s)

    # 2) 색상 증강 — 기본은 4장 동일 변형(프레임 간 상대 단서 보존, 스타일 프라이어만 약화)
    ops: list[str] = [f"shuffle={s}"]
    if cfg.grayscale_p > 0 and rng.random() < cfg.grayscale_p:
        images = [ImageOps.grayscale(im).convert("RGB") for im in images]
        ops.append("grayscale")
    if cfg.jitter_p > 0:
        if cfg.color_scope == "sample":
            if rng.random() < cfg.jitter_p:
                params = _jitter_params(cfg, rng)
                images = [_apply_color(im, params) for im in images]
                ops.append(f"jitter(sample,{params})")
        elif cfg.color_scope == "frame":
            images = [
                _apply_color(im, _jitter_params(cfg, rng)) if rng.random() < cfg.jitter_p else im
                for im in images
            ]
            ops.append("jitter(frame)")
        else:
            raise ValueError(f"color_scope: {cfg.color_scope}")

    # 3) 해상도·블러 지터 — 추론 전처리(다운스케일/블러) 분포 커버용, 4장 동일 적용
    scale = rng.uniform(*cfg.res_aug_range) if (cfg.res_aug_p > 0 and rng.random() < cfg.res_aug_p) else None
    radius = rng.uniform(*cfg.blur_radius_range) if (cfg.blur_p > 0 and rng.random() < cfg.blur_p) else None
    if scale is not None or radius is not None:
        images = [_apply_res_blur(im, scale, radius) for im in images]
        ops.append(f"res_blur(scale={scale},r={radius})")

    meta = dict(sample.meta)
    meta["augment_ops"] = ops
    return replace(sample, images=images, rank=new_rank, meta=meta)
