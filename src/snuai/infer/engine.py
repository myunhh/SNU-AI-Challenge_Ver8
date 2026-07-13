"""VLM 엔진 — 모델 로딩·입력 구성·로짓 판독의 유일한 창구.

설계 원칙:
  - batch=1 (3090 24GB, 노션 결정)
  - 단일 토큰 로짓 판독은 logits_to_keep=1 (미지원 버전은 자동 폴백)
  - attention: flash_attention_2 → sdpa 자동 폴백
  - CPU(tiny 모델 통합 테스트)와 CUDA(3090/VESSL)를 같은 코드로

transformers 4.57+(팀 GPU 환경) / 5.x(로컬 검증) 모두 지원하도록 기능 감지 사용.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np

from ..prompting import extract_media


def apply_pixel_budget(processor, max_pixels: int | None = None,
                       min_pixels: int | None = None,
                       video_max_pixels: int | None = None) -> None:
    """Qwen-VL 계열 image/video processor의 해상도 예산 설정 (속도·VRAM 1순위 레버).

    버전별 규약을 모두 처리 (통합 테스트로 실효성 검증):
      - transformers 4.x: min_pixels / max_pixels 어트리뷰트
      - transformers 5.x: size(SizeDict/dict)의 shortest_edge / longest_edge (픽셀 수)
    학습(train_sft.py)과 추론(VLMEngine)이 같은 함수를 써서 '전처리 일치 원칙'을 보장.

    video_max_pixels: video_processor에만 다른 예산을 적용(Ver7 video_dup 전용).
    video의 max_pixels는 이미지처럼 "장당"이 아니라 **전체 프레임 합산 예산제**라,
    dup_factor로 프레임 수를 늘리면 총 예산도 비례 상향해야 프레임당 해상도가
    유지된다(안 하면 230k→147k처럼 조용히 깎임 — Ver2 실측). None이면 기존처럼
    video도 max_pixels를 그대로 공유(하위호환).
    """
    targets = ((getattr(processor, "image_processor", None), max_pixels, min_pixels),
              (getattr(processor, "video_processor", None),
               video_max_pixels if video_max_pixels is not None else max_pixels, min_pixels))
    for proc, mp, mnp in targets:
        if proc is None:
            continue
        pairs = (("max_pixels", "longest_edge", mp), ("min_pixels", "shortest_edge", mnp))
        for old_name, edge_name, val in pairs:
            if val is None:
                continue
            val = int(val)
            applied = False
            if getattr(proc, old_name, None) is not None:
                setattr(proc, old_name, val)
                applied = True
            size = getattr(proc, "size", None)
            if size is not None:
                for key in (edge_name, old_name):
                    if isinstance(size, dict):
                        if key in size and size[key] is not None:
                            size[key] = val
                            applied = True
                    elif getattr(size, key, None) is not None:
                        try:
                            setattr(size, key, val)
                            applied = True
                        except Exception:  # noqa: BLE001 (frozen dataclass 등)
                            pass
            if not applied:
                import warnings
                warnings.warn(f"{type(proc).__name__}에 {old_name} 적용 실패 — "
                              "해상도 예산이 무시되고 있을 수 있음")


@dataclass
class EngineConfig:
    model_id: str
    processor_id: str | None = None   # None이면 model_id에서 로드 (tiny 테스트·어댑터 대응)
    device: str = "auto"              # "auto"|"cuda"|"cpu"
    four_bit: bool = False            # bnb 4bit (CUDA 전용; 32B-INT4 체크포인트면 False)
    dtype: str = "auto"               # "auto"|"bfloat16"|"float16"|"float32"
    attn: str | None = None           # None=자동(fa2→sdpa), 또는 명시
    max_pixels: int | None = None     # 이미지당 비주얼 토큰 예산(해상도 제어) — 추론 속도의 1순위 레버
    min_pixels: int | None = None
    video_max_pixels: int | None = None  # video 전용 예산(전체 프레임 합산제) — Ver7 video_dup
    video_dup_factor: int = 1         # 각 프레임 연속 복제 횟수(1 또는 짝수만 유효) — Ver7 video_dup
    adapter_path: str | None = None   # QLoRA 어댑터 디렉터리
    kv_quant: bool = False            # quantized KV cache(quanto) — CoT/TTA 배치 시 트리거
    trust_remote_code: bool = False


class VLMEngine:
    def __init__(self, cfg: EngineConfig):
        import torch
        from transformers import AutoProcessor

        self.cfg = cfg
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device

        self.processor = AutoProcessor.from_pretrained(
            cfg.processor_id or cfg.model_id, trust_remote_code=cfg.trust_remote_code)
        self._apply_pixel_budget()
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        self.model = self._load_model()
        self.model.eval()
        self._logits_to_keep_kw = self._detect_logits_kw()
        self._token_id_cache: dict[str, int] = {}

    # ------------------------------------------------------------------ 로딩
    def _apply_pixel_budget(self) -> None:
        apply_pixel_budget(self.processor, self.cfg.max_pixels, self.cfg.min_pixels,
                          video_max_pixels=self.cfg.video_max_pixels)

    def _is_prequantized(self) -> bool:
        """체크포인트 config.json에 quantization_config가 내장돼 있는지 (train_sft와 동일 규약)."""
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(self.cfg.model_id,
                                         trust_remote_code=self.cfg.trust_remote_code)
        return getattr(cfg, "quantization_config", None) is not None

    def _load_model(self):
        import torch
        from transformers import AutoModelForImageTextToText

        kw: dict = {"trust_remote_code": self.cfg.trust_remote_code}
        if self.cfg.dtype != "auto":
            kw["dtype"] = getattr(torch, self.cfg.dtype)
        elif self.device == "cuda":
            kw["dtype"] = torch.bfloat16

        # 사전양자화 체크포인트(config.json에 quantization_config 내장 — unsloth 32B-bnb-4bit 등)
        # 자동 감지: from_pretrained가 알아서 4bit로 로드하므로 four_bit 플래그는 불필요하고,
        # 4bit 모델은 .to(device) 금지라 device_map으로 배치해야 한다 (아래 .to() 분기가 스킵됨).
        if self._is_prequantized():
            if self.cfg.four_bit:
                raise RuntimeError("사전양자화 체크포인트에 four_bit=True 중복 지정 — "
                                   "이중 양자화 사고 방지를 위해 플래그를 제거할 것")
            if self.device != "cuda":
                raise RuntimeError("사전양자화(bnb 4bit) 체크포인트는 CUDA 전용")
            # 체크포인트 자체 quantization_config의 vision 스킵이 bare name이면 이 버전의
            # transformers에서 vision까지 4bit로 재양자화된다(_check_vision_not_quantized가
            # 잡는 지점) — train.qlora와 동일하게 model.visual 접두형으로 보정.
            from transformers import AutoConfig
            from ..train.qlora import patch_prequant_vision_skip
            auto_cfg = AutoConfig.from_pretrained(
                self.cfg.model_id, trust_remote_code=self.cfg.trust_remote_code)
            patch_prequant_vision_skip(auto_cfg)
            kw["config"] = auto_cfg
            kw["device_map"] = "auto"

        if self.cfg.four_bit:
            if self.device != "cuda":
                raise RuntimeError("four_bit=True는 CUDA 전용 (bitsandbytes)")
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                # "model.visual" 없이 4bit를 걸면 vision tower까지 양자화된다 — transformers는
                # skip_modules를 명시하면 자체 기본 스킵 목록을 안 쓴다(실측: 5.12.1, Qwen3-VL-8B
                # 에서 116개 vision 모듈 양자화됨). train/qlora.py VISION_SKIP_MODULES와 동일해야 함.
                llm_int8_skip_modules=["model.visual"],
            )
            kw["device_map"] = "auto"

        attn_chain = [self.cfg.attn] if self.cfg.attn else ["flash_attention_2", "sdpa"]
        model, last_err = None, None
        for attn in attn_chain:
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.cfg.model_id, attn_implementation=attn, **kw)
                self.attn_used = attn
                break
            except (ImportError, ValueError, OSError) as e:
                last_err = e
        if model is None:
            raise RuntimeError(f"모델 로딩 실패({attn_chain}): {last_err}")

        # four_bit 여부와 무관하게 항상 검사 — 사전양자화 체크포인트(32B-INT4,
        # four_bit=False로 로딩)에 vision이 양자화돼 들어있는 경우도 잡아야 한다.
        self._check_vision_not_quantized(model)

        if "device_map" not in kw:
            model = model.to(self.device)

        if self.cfg.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.cfg.adapter_path)
            self._check_lora_only_on_language(model)

        return model

    @staticmethod
    def _check_vision_not_quantized(model) -> None:
        """로딩 후 vision tower에 Linear4bit/Params4bit이 없는지 확인 (규약: vision 비양자화).

        train/qlora.py의 verify_vision_not_quantized와 동일 검사 — infer가 train에 의존하지
        않도록 별도 유지(리팩터링 시 두 쪽 다 고칠 것).
        """
        n_vis = sum(1 for name, mod in model.named_modules()
                   if type(mod).__name__ in ("Linear4bit", "Params4bit")
                   and any(x in name.lower() for x in ("visual", "vision")))
        if n_vis > 0:
            raise RuntimeError(f"vision tower에 4bit 모듈 {n_vis}개 — 규약 위반(vision 비양자화). "
                              "BitsAndBytesConfig(llm_int8_skip_modules=...)를 확인할 것")

    @staticmethod
    def _check_lora_only_on_language(model) -> None:
        """어댑터 로딩 후 LoRA가 language 쪽에만 붙었는지 확인 (규약: vision엔 LoRA 금지).

        train/qlora.py의 verify_lora_only_on_language와 동일 검사 — infer가 train에
        의존하지 않도록 별도 유지(리팩터링 시 두 쪽 다 고칠 것).
        """
        n_lang = n_vis = 0
        for name, _ in model.named_modules():
            if "lora_" not in name:
                continue
            if any(x in name.lower() for x in ("visual", "vision")):
                n_vis += 1
            else:
                n_lang += 1
        if n_vis > 0:
            raise RuntimeError(f"vision tower에 LoRA {n_vis}개 — 잘못 학습된 어댑터(규약 위반)")
        if n_lang == 0:
            raise RuntimeError("어댑터에 LoRA 모듈이 하나도 없음 — adapter_path 확인 필요")

    def _detect_logits_kw(self) -> str | None:
        params = inspect.signature(self.model.forward).parameters
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in params:
                return name
        return None

    # ------------------------------------------------------------- 토큰 유틸
    def token_id_of(self, text: str) -> int:
        """단일 토큰 보장 id. score24 라벨(A..X), FSM 심볼, Yes/No, A/B에 사용.

        후보(원문, 앞공백)를 순서대로 시도해 '정확히 1토큰'인 것을 채택.
        어느 것도 1토큰이 아니면 즉시 실패 — 조용한 다중토큰 라벨은 재앙이므로.
        """
        if text in self._token_id_cache:
            return self._token_id_cache[text]
        for cand in (text, " " + text):
            ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                self._token_id_cache[text] = ids[0]
                return ids[0]
        raise ValueError(f"단일 토큰이 아님: {text!r} → {self.tokenizer.encode(text, add_special_tokens=False)}")

    # ------------------------------------------------------------- 입력 구성
    def build_inputs(self, messages: list[dict], add_generation_prompt: bool = True):
        from ..prompting import call_processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        images, videos = extract_media(messages)
        inputs = call_processor(self.processor, [text], images, videos,
                                return_tensors="pt")
        return inputs.to(self.model.device)

    # ------------------------------------------------------------- 로짓 판독
    def next_token_logits(self, messages: list[dict]) -> np.ndarray:
        """프롬프트 다음 1토큰의 로짓 (vocab,). score24/pairwise/Yes-No의 공통 기반.

        logits_to_keep=1로 전체 시퀀스 로짓 materialize 방지(3090에서 ~GB 절약).
        """
        torch = self.torch
        inputs = self.build_inputs(messages)
        kw = {self._logits_to_keep_kw: 1} if self._logits_to_keep_kw else {}
        with torch.no_grad():
            out = self.model(**inputs, **kw)
        return out.logits[0, -1].float().cpu().numpy()

    def restricted_logprobs(self, messages: list[dict], token_ids: list[int]) -> np.ndarray:
        """다음 토큰 로짓을 token_ids로 제한한 log-softmax (제한 집합 내 정규화)."""
        logits = self.next_token_logits(messages)[token_ids]
        x = logits - logits.max()
        return x - np.log(np.exp(x).sum())

    def continuation_logprob(self, messages: list[dict], continuation: str) -> float:
        """log P(continuation | messages). 우도 스코어링(zero-shot)용.

        같은 프롬프트 구조에서 candidate 간 상대 비교용이므로 절대값 의미는 없음.
        """
        torch = self.torch
        inputs = self.build_inputs(messages, add_generation_prompt=True)
        cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
        if not cont_ids:
            raise ValueError("빈 continuation")
        ids = inputs["input_ids"]
        old_len = ids.shape[1]
        cont = torch.tensor([cont_ids], device=ids.device)
        full = torch.cat([ids, cont], dim=1)
        inputs = dict(inputs)
        inputs["input_ids"] = full
        # 토큰 축(길이 old_len)을 공유하는 모든 텐서를 함께 연장
        # (attention_mask→1, mm_token_type_ids 등 타입 표기→0=텍스트)
        for key, val in list(inputs.items()):
            if key == "input_ids" or not torch.is_tensor(val):
                continue
            if val.ndim >= 2 and val.shape[0] == 1 and val.shape[1] == old_len:
                fill = 1 if key == "attention_mask" else 0
                pad_shape = (1, len(cont_ids)) + tuple(val.shape[2:])
                pad = torch.full(pad_shape, fill, dtype=val.dtype, device=val.device)
                inputs[key] = torch.cat([val, pad], dim=1)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, L, V)
        lp = torch.log_softmax(logits[0, ids.shape[1] - 1: full.shape[1] - 1].float(), dim=-1)
        tgt = torch.tensor(cont_ids, device=lp.device)
        return float(lp.gather(1, tgt.unsqueeze(1)).sum().item())

    # --------------------------------------------------------------- 생성
    def raw_text_of(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """chat template 적용 텍스트 (CoT 2단계 이어붙이기용)."""
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def generate_from_text(self, text: str, images: list | None = None,
                           videos: list | None = None, max_new_tokens: int = 8,
                           logits_processor=None, do_sample: bool = False,
                           temperature: float = 1.0) -> tuple[str, "object", int]:
        """이미 template이 적용된 텍스트에서 이어서 생성 (FSM 제약 답 구간용)."""
        from ..prompting import call_processor
        inputs = call_processor(self.processor, [text], images, videos,
                                return_tensors="pt").to(self.model.device)
        kw: dict = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            kw["temperature"] = temperature
        if logits_processor is not None:
            kw["logits_processor"] = logits_processor
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, prompt_len:]
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True), gen_ids, prompt_len

    def generate_text(self, messages: list[dict], max_new_tokens: int = 512,
                      do_sample: bool = False, temperature: float = 1.0,
                      stop_strings: list[str] | None = None,
                      logits_processor=None) -> tuple[str, "object", int]:
        """생성 → (생성 텍스트, 생성 토큰 id 텐서(1,g), 프롬프트 길이)."""
        inputs = self.build_inputs(messages)
        kw: dict = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            kw["temperature"] = temperature
        if stop_strings:
            kw["stop_strings"] = stop_strings
            kw["tokenizer"] = self.tokenizer
        if logits_processor is not None:
            kw["logits_processor"] = logits_processor
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, prompt_len:]
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return text, gen_ids, prompt_len
