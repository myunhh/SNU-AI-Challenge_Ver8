"""QLoRA 구성 — 4bit 양자화 + LoRA 타겟 선정 (vision encoder 제외 규약).

⚠️ 가장 흔한 실수: peft target_modules에 "q_proj" 같은 접미사만 주면
   vision tower의 q_proj까지 LoRA가 붙는다 (Qwen-VL 비전 블록에도 동명 모듈 존재).
   → 반드시 full-path로 필터링해 language 쪽만 타겟팅한다 (노션: vision은 양자화도
   LoRA도 하지 않음). select_lora_targets가 그 필터의 유일한 구현이다.
"""

from __future__ import annotations

from typing import Iterable

# language 블록에서 LoRA를 붙일 leaf 모듈명
DEFAULT_LEAF_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")
# 경로에 이 문자열이 들어가면 무조건 제외 (vision tower·프로젝터·임베딩류)
DEFAULT_EXCLUDE_SUBSTR = ("visual", "vision", "merger", "patch_embed",
                          "image", "video", "lm_head", "embed")


def select_lora_targets(module_names: Iterable[str],
                        leaf_targets: tuple[str, ...] = DEFAULT_LEAF_TARGETS,
                        exclude_substr: tuple[str, ...] = DEFAULT_EXCLUDE_SUBSTR) -> list[str]:
    """모델의 전체 모듈 이름 목록 → LoRA 타겟 full-path 목록 (vision 제외 보장)."""
    out = []
    for name in module_names:
        low = name.lower()
        if any(x in low for x in exclude_substr):
            continue
        if any(name.endswith("." + leaf) or name == leaf for leaf in leaf_targets):
            out.append(name)
    if not out:
        raise ValueError("LoRA 타겟이 비었음 — 모델 구조/이름 규칙 확인 필요")
    return sorted(out)


def lora_targets_of_model(model) -> list[str]:
    return select_lora_targets(name for name, _ in model.named_modules())


#: Qwen3-VL의 vision tower 경로 — llm_int8_skip_modules에 그대로 전달.
#: transformers는 skip_modules를 지정하면(quantizers/base.py get_modules_to_not_convert)
#: 자체 기본 스킵 목록을 쓰지 않으므로, "model.visual" 없이 4bit를 걸면 vision tower까지
#: 양자화된다(실측 확인: transformers 5.12.1, Qwen3-VL-8B-Instruct에서 116개 모듈 양자화됨).
#: should_convert_module()이 prefix/suffix 매칭이라 "visual"만으로는 "model.visual...."
#: 경로에 안 걸린다 — 반드시 "model." 포함한 전체 접두사를 줄 것.
VISION_SKIP_MODULES = ["model.visual"]


def patch_prequant_vision_skip(config) -> bool:
    """사전양자화 체크포인트의 llm_int8_skip_modules를 vision 비양자화가 되도록 보정.

    unsloth `Qwen3-VL-32B-Instruct-bnb-4bit` 등은 quantization_config에
    vision 스킵을 bare name(`visual`/`vision_tower`)으로 넣어두는데, transformers
    5.x `should_convert_module()`은 re.match 앵커링이라 `model.visual.*` 실경로에
    안 걸린다 → vision tower가 로드 시 4bit로 재양자화됨(체크포인트엔 bf16으로
    저장돼 있는데도). 규약(vision 비양자화)에 맞게 VISION_SKIP_MODULES(`model.`
    접두형)를 skip 목록에 in-place로 추가한다. `should_convert_module`은 스킵
    목록의 OR라 기존 항목을 지우지 않고 더하기만 하면 안전하다 (Ver4 실측 이식).

    반환: 실제로 보정했으면 True (양자화 config 없으면 False).
    """
    q = getattr(config, "quantization_config", None)
    if q is None:
        return False
    is_dict = isinstance(q, dict)
    skips = list((q.get if is_dict else lambda k, d=None: getattr(q, k, d))(
        "llm_int8_skip_modules") or [])
    added = [m for m in VISION_SKIP_MODULES if m not in skips]
    if not added:
        return False
    skips.extend(added)
    if is_dict:
        q["llm_int8_skip_modules"] = skips
    else:
        q.llm_int8_skip_modules = skips
    return True


def make_bnb_kwargs() -> dict:
    """BitsAndBytesConfig 인자 (노션 QLoRA 항목: NF4 + double quant + bf16 compute + vision 스킵)."""
    return dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="bfloat16",
                llm_int8_skip_modules=list(VISION_SKIP_MODULES))


def make_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05,
                     target_modules: list[str] | None = None):
    """peft LoraConfig 생성 (lazy import — GPU 머신 전용 경로)."""
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=target_modules)


def verify_vision_not_quantized(model) -> dict:
    """로딩된 모델에서 vision tower에 4bit 모듈이 없는지 검사. 위반 시 raise.

    반환: {"n_quant_lang": int, "n_quant_vision": int} (보고서·로그용)
    """
    n_lang = n_vis = 0
    for name, mod in model.named_modules():
        is_4bit = type(mod).__name__ in ("Linear4bit", "Params4bit")
        if not is_4bit:
            continue
        if any(x in name.lower() for x in ("visual", "vision")):
            n_vis += 1
        else:
            n_lang += 1
    if n_vis > 0:
        raise RuntimeError(f"vision tower에 4bit 모듈 {n_vis}개 — 규약 위반(노션: vision 비양자화)")
    return {"n_quant_lang": n_lang, "n_quant_vision": n_vis}


def verify_lora_only_on_language(model) -> dict:
    """PEFT 적용 후 LoRA 모듈이 vision 쪽에 없는지 검사. 위반 시 raise."""
    n_lang = n_vis = 0
    for name, _ in model.named_modules():
        if "lora_" not in name:
            continue
        if any(x in name.lower() for x in ("visual", "vision")):
            n_vis += 1
        else:
            n_lang += 1
    if n_vis > 0:
        raise RuntimeError(f"vision tower에 LoRA {n_vis}개 — select_lora_targets를 우회했는지 확인")
    if n_lang == 0:
        raise RuntimeError("LoRA가 하나도 안 붙음")
    return {"n_lora_lang": n_lang, "n_lora_vision": n_vis}
