# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# SNU AI Challenge (2026) — **Ver8**

캡션(스토리라인)에 맞춰 셔플된 비디오 프레임 4장을 시간순으로 재배열하는 멀티모달
태스크 (4! = 24 순열 중 정답 찾기). 대회: https://snuaichallenge.github.io/

> **이 저장소는 Ver8이다** — 아직 코드 없음(TODO.md §0 부트스트랩부터).
> 코드 베이스는 **Ver7을 복사**해서 시작할 것: Ver7의 `dup_factor=1` 경로는 Ver3와
> byte-identical이면서 Ver3에 없는 도구(audit_input·eval_report·scores24 덤프·
> pairwise 게이트 지표·decode_expected)를 전부 포함한 superset이다.
> **현 제출 구성 = Ver3+TTA3 bf16 서빙, LB 0.865** (`Ver7/LB_LOG.md` #3).
> Ver8의 목표: 합법적인 **학습 단 개선**(DPO·metric-aligned 라벨·증강)으로 이걸 넘기.

## 일정 · 제출 예산

| 일정 | 내용 |
|---|---|
| **07.24 23:59** | 예선 제출 마감 (Kaggle, **1일 2회** — 07-12 저녁 기준 잔여 ~23슬롯) |
| 07.25~28 | 상위 16팀 검증자료(코드·보고서) 제출 — 게이트 기록·audit 덤프 전부 보존 |
| 08.03 / 08.07 | 본선 발표 / 본선 오프라인 발표(서울대) |

## 필수 준수 규정 (위반 시 실격 — Ver3/CLAUDE.md에서 승계, 전문은 Notion)

- 완전 오프라인 추론, **RTX 3090(24GB) 1대·24h 내 test 819건**, 모델 총량 80GB
  (개발 머신 4090 — 3090 환산 병기)
- 생성형 모델 데이터 증강 금지(모델 생성 텍스트를 학습 데이터에 넣지 않는다),
  외부 데이터 학습 금지, 일반 augmentation 허용, 2026-05-31 이전 공개 모델만
- **앙상블 전면 금지** — ⚠️ SigLIP2 등 별도 모델은 "입력 전처리 한정"으로만 방어
  가능, **출력/점수 융합은 실격 사유**. (이 규정 때문에 TODO_VER8의 P4-a
  "SigLIP2 힌트"는 폐기됨 — 전처리형 변형은 이미 R3에서 −3.07pp 기각)
- 허용: Quantization, LoRA, CoT, TTA. 단일 모델 반복 호출(TTA/캐스케이드)은 허용 범주
- **데이터 누수 금지**: 모든 튜닝 결정은 train 홀드아웃 945건으로만. test 분포 참조 = 실격

## 검증 체계 (2026-07-12 확립 — Ver8에서 그대로 사용)

- **게이트**: `scripts/ab_gate.py` paired bootstrap, **EM과 쌍순서(1−KT/6) 병행**
  (`--metric pairwise`), Δ ≥ +2%p AND 95% CI 하한 > 0
- **모든 제출은 bf16 서빙** — 4bit 서빙은 test에서 ~4pp 페널티 실측
  (`Ver7/LB_LOG.md` 관찰). 홀드아웃에선 이 페널티가 안 보이니 주의
- **LB 예측자**: bf16 구성 한정, **홀드아웃 쌍순서 + 2.6~2.7pp ≈ LB** (n=3에서
  일정). 제출 전 예측치를 `Ver7/LB_LOG.md`에 기록하고 실측으로 검증할 것
- ⚠️ **채점 방식 미확정**: 규정 문서는 "예선 EM만(부분점수 없음)"인데 LB 실측
  정합은 쌍순서 채점을 강하게 시사(`Ver7/runs/drift_forensics_v7.md` §2 —
  단, test가 홀드아웃보다 훨씬 쉬워서 EM 가설도 LB 스케일 설명 가능). TODO §1이
  최우선인 이유. 어느 쪽이든 위 게이트(양 지표 병행)는 유효
- 홀드아웃↔test 분포 차이 큼(test margin 중앙값 0.99 vs 홀드아웃 0.6) —
  **계열 내부 A/B(같은 어댑터 위 기법)에만 홀드아웃 신뢰**, 계열 간 최종 판정은 LB 슬롯
- fresh split 재평가 불가(기존 어댑터들이 홀드아웃 외 전체로 학습됨) — Ver8에서
  **새로 학습하는 어댑터는 동일 split을 유지**해야 과거 런과 비교 가능

## 결과 이력 (동일 홀드아웃 945건 / LB)

| 버전 | 요점 | 홀드아웃 EM | 쌍순서 | LB |
|---|---|---|---|---|
| Ver1 | QLoRA 1000스텝 이미지 모드 | 0.4931 | 0.8187 | 0.77486 |
| Ver2 | video-mode 사고 | 0.3238 | 0.7596 | 0.45724 |
| **Ver3** | 이미지+legend+2000스텝, **+TTA3** | 0.5185 / **0.5566** | 0.8205 / **0.8386** | 4bit 0.82373 / **bf16 0.865 ← 현 최고** |
| Ver5 | 노트-재스코어 캐스케이드 | 기각(±0) | 기각 | — |
| Ver7 | video_dup(R1 재도전)+TTA3 | 0.5577 | 0.8312 | bf16 0.85863 |

**07-12 기각 누적** (전부 `Ver7/runs/gate_*.json`에 기록): metric-optimal
decoding(분포 꼬리=노이즈, 온도 보정 무효) · TTA5 v3/v7(3뷰 포화) · 4bit 서빙 v7.
과거 기각: 캐스케이드, SigLIP2 소프닝(−3.07pp), identity_ratio, merge→PTQ(B안),
Ver5 재스코어. **패턴: "같은 모델·같은 정보의 재해석"은 전부 무효 — 남은 레버는
학습 단(가중치를 바꾸는 것)뿐** (Ver8 TODO §2가 그 목록).

## 아키텍처 요점 (상세는 Ver3/CLAUDE.md·Ver7/VER7.md — 여기선 함정만)

- **QLoRA(4bit 학습, vision 비양자화) + one-pass score24(24순열→단일토큰 A~X) +
  legend 프롬프트**. 학습=추론 프롬프트 byte-identical 원칙 — 프롬프트 변경 = 재학습
- **순열 규약**: `perm.py`가 SSOT. adapter는 order, snuai는 rank 인코딩 —
  `perm.order_to_rank()`로만 변환, 직접 구현 금지(50.5% 함정). `tests/test_perm.py`가 방어선
- **부분점수 지표도 perm.py가 SSOT**: `pairwise_score`/`position_score` (Ver7에서 추가)
- 증강은 학습 전용, 추론 변형은 TTA(셔플+리맵)만. 결정적 전처리도 학습=추론 일치 기본
- 새 인코딩/전처리 채택 전 `scripts/audit_input.py`로 실제 input_ids 감사 (§17 게이트,
  Ver2 사고 재발 방지). transformers 버전업 시 `do_sample_frames` 조용한 재샘플링 주의
- 속도 여유 큼: TTA3 bf16 test 풀런 ~0.27h (예산 24h의 ~1%) — 토큰 최적화 불필요

## 개발 환경 (⚠️ 함정 2개)

conda env `py3_11`. **`snuai` editable install이 Ver1을 가리킴** — 반드시:

```bash
export PYTHONPATH=$PWD/src                                # Ver8 코드 강제
export LD_LIBRARY_PATH=~/anaconda3/envs/py3_11/lib        # pandas GLIBCXX 에러 방지
```

```bash
pytest tests/ -q                                          # CPU 안전망 (Ver7 기준 55개)
python -m snuai.infer.predict --synthetic 24 --strategy dummy --tta 3 --eval --out runs/dryrun  # GPU 없이 E2E
# 홀드아웃 평가 → 게이트 (모든 채택 결정의 유일한 경로)
python -m snuai.infer.predict --csv data/train.csv --image-dir data/train --holdout-val --eval --adapter <path> --tta 3 --out runs/cal_X
python scripts/ab_gate.py runs/cal_A runs/cal_B --metric pairwise --csv data/train.csv --image-dir data/train --holdout-val --name X --out runs/gate_X.json
```

데이터는 `~/SNU-AI-Challenge/data/` 공유본 심볼릭 링크 (부트스트랩 시 생성, TODO §0).

## 현행 문서 지도

- `Ver8/TODO.md` — **현행 작업 목록** (Ver7/TODO_VER8.md를 승계·대체)
- `Ver7/LB_LOG.md` — 제출 대장 (**계속 여기에 기록**, 이사하지 않음)
- `Ver7/runs/drift_forensics_v7.md` · `reeval_pairwise_all.md` — 검증 체계의 근거
- `Ver3/CLAUDE.md` — 규약·함정 상세 / `Ver3/REPORT_E_CURVE.md` — 양자화 결론
- `Ver3/VER3.md` — Ver2 부검 / `Ver7/VER7.md` — video_dup·LB 반전 경위
