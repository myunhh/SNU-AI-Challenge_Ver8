# 캡션 의존도 절단 진단 — 2026-07-21

**작성**: Claude (elicer 세션, A100) · **대상**: 다음 세션/다른 AI 인수인계용
**스크립트**: `scripts/measure_caption_dependence.py` (origin `3ee770b`, 이 세션에서 버그 수정)
**모델**: 챔피언 어댑터 `runs/DPO-checkpoint-600` (base `unsloth/Qwen3-VL-32B-Instruct-bnb-4bit`)
**데이터**: `data/train.csv` 전체 9537건 중 200건 표본(seed=17), score24 단일 뷰(TTA 없음)

## 스크립트 버그 (수정 완료, origin 미반영 상태였음)

`s.images`(lazy 경로 리스트)를 `load_images()` 없이 그대로 `scorer.scores()`에 넘겨
첫 샘플부터 `TypeError: only a single or a list of entries is supported ... PosixPath`로 크래시.
`predict.py`의 `run_predict`가 쓰는 패턴(`images = s.load_images()`)과 동일하게
`scripts/measure_caption_dependence.py`의 for-loop 안에서 샘플당 1회 `load_images()` 호출로 수정.
**이 수정이 origin에 아직 push 안 됐다면 다른 머신에서 그대로 실행 시 동일하게 크래시함.**

## 3조건 정의

- `normal`: 실제 캡션
- `empty`: 빈 문자열 `""` (이미지만으로 얼마나 맞히는지 = 암기/이미지단서 하한)
- `swapped`: 표본 내 다른 샘플의 캡션을 결정적 시드로 1:1 배정(자기 자신 매칭 없음)

## 결과 (n=200)

| 조건 | EM 정확도 |
|---|---|
| normal | 126/200 = **0.6300** |
| empty | 42/200 = **0.2100** |
| swapped | 33/200 = **0.1650** |
| (참고) 24-way 균등 랜덤 기준선 | ≈0.0417 |

### 교차표

- normal correct ∩ empty correct: 34
- normal correct, empty **wrong** (캡션 제거 시 틀림): 92
- empty correct, normal **wrong** (캡션이 오히려 방해): 8 ← 소수지만 0은 아님
- normal correct ∩ swapped correct: 32
- swapped correct, normal wrong (엉뚱한 캡션인데 오히려 맞음): 1
- **normal에서 맞고 empty·swapped 둘 다에서 틀리는 표본**: 76/126 = 0.603
  (가장 보수적인 "진짜 캡션을 읽고 쓴다"는 신호)

### 캡션 실제 사용 하한 추정치

- normal→empty로 캡션 제거 시 정답이 틀리는 비율: 92/126 = **0.7302**
- normal→swapped로 캡션 교체 시 정답이 틀리는 비율: 94/126 = **0.7460**

## 해석

1. **캡션은 실제로 강하게 쓰인다.** normal(0.63) vs empty(0.21)/swapped(0.165) 격차가
   크고, 24-way 랜덤 기준선(0.042)보다 image-only(0.21)도 훨씬 높음 — 즉 모델이
   (a) 이미지만으로도 어느 정도 암기/추론하지만 (b) 캡션이 있으면 성능이 3배 가까이 뛴다.
2. empty와 swapped의 차이(0.21 vs 0.165)가 크지 않다 — 캡션이 없을 때와 **틀린** 캡션을
   줄 때 성능이 비슷하게 나쁘다는 뜻으로, "아무 텍스트나 있으면 도움" 같은 얕은 패턴이
   아니라 **캡션 내용 자체**를 실제로 활용한다는 근거로 해석 가능.
3. empty-only 8건(캡션이 오히려 방해)은 캡션과 이미지가 미묘하게 어긋나는 라벨링
   케이스이거나 모델이 캡션에 낚이는(distracted) 실패 모드일 수 있음 — 표본이 작아
   추가 조사 없이는 결론 보류.

## 캐비엇 (스크립트 docstring에 이미 명시된 것, 반드시 인용할 것)

> train 전량학습이라 암기로 caption 없이도 맞힐 수 있어 의존도 과소측정 가능성 있음

즉 위 수치는 "캡션 의존도의 하한"이지 정확한 값이 아님. 특히 `empty` EM 0.21이 실제보다
높게 나왔을 수 있음(암기로 이미지만 보고도 맞히는 비율이 섞여 있음) — 진짜 캡션 의존도는
표에 나온 값보다 더 클 가능성이 있다는 뜻(과소측정 방향이므로 결론 1의 방향은 안전).

## 원본 데이터 위치

- `runs/caption_dependence/progress.jsonl` — 200건 샘플별 원본 예측(normal/empty/swapped 각각의 pred rank·correct)
- `runs/caption_dependence/summary.json` — EM 요약
- 재현 커맨드:
  ```bash
  cd SNU-AI-Challenge_Ver8 && export PYTHONPATH=$PWD/src
  python3 scripts/measure_caption_dependence.py \
      --csv data/train.csv --image-dir data/train \
      --adapter runs/DPO-checkpoint-600 \
      --model-id unsloth/Qwen3-VL-32B-Instruct-bnb-4bit \
      --n 200 --out runs/caption_dependence
  ```
  (`--model-id` 기본값은 다른 머신의 `/home/yhmin/...` 경로이므로 HF 캐시 id로 override 필요)

## 별도 미해결 이슈 (이 세션에서 발견, 참고용)

`bf1043f`(origin의 BALANCED8)와 별개로, 이 세션은 동일한 설계 브리프로 **독자적인 BALANCED8
변형**(코셋 대표원을 4-cycle 대신 전치 g=(1,0,2,3)로 선택)을 먼저 구현해 실제 test 819건
추론까지 완료했었음(`runs/test_champ_tta8b/submission.csv`, TTA4 챔피언 대비 EM 일치율
0.9658). 이후 origin이 이미 공식 버전을 push한 것을 발견해 origin 버전으로 되돌렸으나
(로컬 변형은 `git stash` 보관, 미삭제), **완료된 submission.csv는 이제 origin의 공식
BALANCED8이 아닌 이전(로컬) 변형으로 만들어진 결과물**임 — 재사용 전 재확인 필요.
