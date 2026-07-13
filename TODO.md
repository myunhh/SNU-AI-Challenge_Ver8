# Ver8 TODO (2026-07-12 작성 — `Ver7/TODO_VER8.md`를 승계·대체)

**출발점**: 현 제출 구성 Ver3+TTA3 bf16 = **LB 0.865**. 마감 **07-24 23:59**
(잔여 ~12일, LB 슬롯 ~23회). 게이트·bf16 원칙·LB 예측자는 `CLAUDE.md` 검증 체계 참고.

**Ver8의 논지**: 추론 단 개선(재질의·디코딩·TTA 확장·서빙 정밀도)은 07-12까지
전부 실측 소진됐다. 남은 합법 레버는 **학습 단**(가중치를 바꾸는 것)뿐이고,
오류 구조 분석(홀드아웃 KT 히스토그램: 오답의 절반이 KT거리 1~2 근접 오답)이
그 타깃을 명확히 가리킨다.

> **2026-07-13 갱신 — base 모델을 32B-bnb-4bit(VESSL A100)로 전환**: §2 전체를
> 8B/4090 가정 대신 **Ver4와 동일 base**(unsloth/Qwen3-VL-32B-Instruct-bnb-4bit,
> 사전양자화 A안)로 진행. `Ver4`에서 발견된 vision 재양자화 버그 수정
> (`patch_prequant_vision_skip`)을 Ver8의 `qlora.py`/`engine.py`/`train_sft.py`에
> 이식 완료(Ver8은 Ver7에서 분기해 이 수정이 없었음). `train_sft.py --model-id`
> 기본값도 32B-4bit 로컬 경로로 변경. §2a/§2b는 이 base 위에서 구현·테스트
> 완료(아래 각 절 참고) — **실제 GPU 학습 실행은 아직 안 함** (VESSL 발사는
> 사용자 확인 후).

---

## §0. 부트스트랩 (코드 작성 없이 복사·링크만, ~30분) — ✅ 완료 2026-07-13

- [x] Ver7 코드 복사: `src/ scripts/ tests/ conftest.py pyproject.toml requirements*.txt`
  (Ver7의 dup=1 경로가 Ver3와 byte-identical + 신규 도구 superset이라 Ver7이 베이스)
- [x] `data/` 구성: Ver7과 동일하게 공유본 심볼릭 링크 + `adapter.py` 복사
- [x] `pytest tests/ -q` 55개 통과 확인 (환경 함정 2개는 CLAUDE.md 참고)
- [x] `runs/` gitignore, git init (본선 검증자료 관례) — 로컬 git repo만, 원격 미연결

## §1. 채점 방식 확정 — ⚠️ 최우선, §2의 지표 선택이 여기 걸림

규정 문서(Notion)는 "예선 **EM만**(부분점수 없음)"인데, LB 실측 정합은
**쌍순서 채점**을 시사(점수차가 쌍순서 상한의 98.6%로 포화 + bf16 오프셋 일정).
test가 훨씬 쉬워서(margin 중앙값 0.99) EM 가설도 LB 스케일은 설명 가능 — 미결.

- [ ] Kaggle 대회 페이지 Evaluation 탭·규정 원문 재확인 (사용자 — 로그인 필요라 위임)
- [x] **진단 제출 파일 생성 스크립트 구현 완료**: `scripts/diag_swap_submission.py`
  (오프라인, GPU 불필요, 테스트 3개). 기존 제출의 `progress.jsonl`에서 고마진
  K=60건에 인접 스와프 1개를 적용한 제출 CSV를 만든다. Ver7 ckpt1800+TTA3
  progress.jsonl(819건)로 실행 확인: EM 가설 −7.33%p / 쌍순서 가설 −1.22%p.
  예 `python scripts/diag_swap_submission.py --progress ../Ver7/runs/test_v7_ckpt1800_tta3/progress.jsonl --k 60 --out runs/diag_swap60/submission.csv`
- [ ] **미실행**: 위에서 생성한 파일을 실제 Kaggle에 제출하는 것은 1일 2회 슬롯을
  쓰므로 사용자 확인 후 진행 (검증용 슬롯 소모 — LB_LOG.md에 기록)
- [ ] 판별 결과를 `CLAUDE.md`·`Ver7/LB_LOG.md`에 기록, EM으로 판명 시 게이트
  주지표를 EM으로 되돌림 (쌍순서는 참고 지표로 유지)

## §2. 학습 단 개선 후보 (본 게임 — 32B-4bit·VESSL A100, 위 갱신 참고)

공통 규칙: Ver3 레시피(이미지·legend·2000스텝·uniform 증강)를 기준선으로 **한 번에
한 변수만**. 홀드아웃 945 split 유지(과거 런과 비교 가능성). 게이트는 EM+쌍순서
병행, 통과 시에만 test 추론(bf16) → LB 슬롯. 학습 전 `audit_input.py` 감사.

### 2a. DPO — 인접 스와프 hard negative (1순위 추천) — ✅ 코드 완료, 학습 미실행

- 근거: 홀드아웃 오답의 최빈 유형이 KT거리 1(인접 스와프 한 개 차이, 189/945건).
  `perm.adjacent_swap_ranks()`가 이미 rejected 후보 생성기로 존재(설계 당시부터
  DPO 용도로 주석됨). SFT 어댑터에서 이어서 DPO 1~2k스텝.
- [x] chosen=정답 letter, rejected=인접 스와프 letter 3종 중 모델 점수 최고인
  것(hard negative)으로 pair 구성 — `dpo_pairs.build_dpo_records(..., scorer=...)`
  에 scorer 콜백 추가(없으면 기존처럼 무작위 폴백). 캐탈로그 "DPO" 항목, 규정상
  합법(생성 데이터 아님, 라벨 변환일 뿐)
- [x] `src/snuai/train/train_dpo.py` 신규: score24가 단일토큰이라 TRL DPOTrainer
  없이 자체 구현(같은 프롬프트의 다음 토큰에서 chosen/rejected 두 로그확률만
  비교) — 참조 로그확률은 `model.disable_adapter()`로 얻어 별도 참조모델 불필요.
  `--hard-negative`(기본 on) 플래그로 학습 시작 전 스코어링 프리패스 자동 실행.
  32B-4bit(A100) 기본, `--adapter runs/sft32b_v8/adapter_final` 필수 인자.
  테스트 10개(`test_dpo_pairs.py` 6 + `test_train_dpo.py` 4, 전부 CPU).
- [ ] **미실행**: 실제 A100 학습 발사(§2a 전체를 태우는 6.5h~ 상당 GPU 작업) —
  Ver4의 32B SFT(runs/sft32b_v4/adapter_final)가 완료되면 그걸 `--adapter`로
  넘겨 이어서 DPO 하는 경로가 유력(Ver8에서 별도로 32B SFT를 처음부터 돌릴
  필요 없이 재사용 가능 — 사용자 확인 필요)
- [ ] 게이트 통과 시 TTA3 재게이트(레시피 바뀌면 재검증 원칙)

### 2b. metric-aligned soft label SFT (2순위 — §1이 쌍순서로 판명될 때만) — ✅ 코드 완료

- 근거: 채점이 부분점수라면 CE의 one-hot 타깃 대신 **쌍순서 유사도 커널로
  스무딩한 soft 타깃**(24클래스)이 채점 함수와 정렬됨.
- [x] `perm.soft_target_distribution(rank, temperature)` 추가(pairwise_score 커널
  softmax) + `SFTDatasetConfig.soft_label_temperature`(기본 None=one-hot,
  byte-identical) + `SFTCollator`가 letter 위치만 hard label에서 빼고
  soft_targets/answer_pos/letter_token_ids를 enc에 실음 + `train_sft.py`의
  `_make_soft_label_trainer_cls`(letter 위치 soft CE + 나머지 표준 CE 합산).
  `--soft-label-temperature` CLI 플래그. 테스트 7개(perm 3 + dataset/loss 4).
- [ ] EM으로 판명되면 이 항목은 폐기(플래그 기본 None이라 안 쓰면 영향 없음)

### 2c. 색상 증강 A/B (3순위 — 카탈로그 대기 항목 소화) — 코드는 이미 존재, 실행만 필요

- 증강 코드(`grayscale_p`/`jitter_p` 등)는 Ver3부터 이미 `augment.py`/`train_sft.py`에
  구현돼 있음 — 신규 코드 불필요. 남은 건 A/B 실측(끄고/켜고 재학습 비교)뿐.
- [ ] uniform 순열 증강에 색상 지터 추가 재학습 — 저비용이지만 기대치도 보통.
  2a/2b 학습 사이 GPU 유휴 시간에만

### ❌ 폐기 (착수 금지 — 근거 기록)

- ~~SigLIP2 임베딩 힌트~~: **앙상블 전면 금지 규정** 위반(별도 모델 출력 융합 =
  실격 사유). 전처리 한정 변형(R3 소프닝)은 이미 −3.07pp 기각
- ~~추론 단 개선 일체~~ (재질의·디코딩·TTA 확장·서빙 정밀도): 07-12까지 3계열
  독립 실험으로 소진 — `Ver7/TODO_VER8.md` 참고

## §3. 32B 게이트 판단 — Ver4 트랙과 통합 (중복 진행 금지)

- Ver4 저장소가 이미 이 게이트를 진행 중(2026-07-13 A100에서 32B SFT 본학습
  시작, ETA ~07-14). Ver8은 **같은 32B-4bit base를 §2a/§2b 실험에 재사용**하는
  것이 목적이지 게이트 0/1/32B SFT를 별도로 다시 도는 게 아니다 — 중복 학습
  방지를 위해 Ver4의 게이트 결과·`runs/sft32b_v4/adapter_final`를 그대로 참조.
- [ ] Ver4의 게이트 1(32B zero-shot vs 8B zero-shot) 통과 여부 확인 후 §2a
  DPO의 `--adapter`로 Ver4 산출물을 사용할지 결정 (Ver4 실패 시 Ver8 §2도
  8B로 축소 재검토)

## §4. 마감 전 필수 (시기 무관, 신규 실험과 병행)

- [ ] **오프라인 리허설**: `HF_HUB_OFFLINE=1` test 819 풀런(현 제출 구성 bf16) +
  3090 환산 시간 리포트 — D9까지는 1회 완료해둘 것
- [ ] 본선 검증자료 정리: 게이트 JSON·audit 덤프·포렌식 리포트·LB_LOG·
  버전별 VER*.md 기각 기록 일람화
- [ ] `Ver7/LB_LOG.md` 계속 갱신 (제출마다 예측치 → 실측 기록)
- [ ] D11~12는 신규 실험 동결, 제출 구성 확정·재현 확인만

## 일정 제안 (12일)

| 구간 | 내용 |
|---|---|
| D1 (07-13) | §0 부트스트랩 + §1 채점 확정 (+필요시 진단 제출) |
| D2~4 | §2a DPO 구현→학습→게이트 (+2b는 §1 결과에 따라) |
| D5~7 | §2 2차 후보 or §3 32B 게이트 0·1 |
| D8~10 | 승자 test 제출·LB 검증, §4 리허설 |
| D11~12 | 동결·검증자료 마무리 |
