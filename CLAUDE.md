# CLAUDE.md — Ver8

셔플된 비디오 프레임 4장 + 캡션 → 시간순 재배열(4!=24). SNU AI Challenge 2026.

**이 저장소 = Ver7을 베이스로 시작한 학습 단 개선 트랙 — 현재 대회 챔피언(가중치는
DPO checkpoint-600 그대로, 서빙 TTA만 계속 갱신). 트랙(학습) 자체는 2026-07-20 종료**
(챔피언 가중치·제출 구성은 그대로 유지, 예정됐던 hard-negative 본런은 미착수 — 마감까지
Ver11/Ver12에 자원 집중, 서빙 단 실험만 계속).
추론 단 개선(재질의·디코딩·TTA 확장·서빙 정밀도)이 Ver1~7에서 전부 실측 소진된 뒤, 남은 합법
레버인 학습 단(가중치를 바꾸는 것)에 집중. **DPO(인접 스와프 hard-negative)**를 Ver4 ckpt1600
base 위에 A100에서 실행해 checkpoint-200이 LB 0.90401(Ver4 ckpt1600의 0.90226 대비 +0.175pp)
찍고, 2026-07-18 스텝 스윕 홀드아웃 945건 페어드 비교(맥니마)에서 **checkpoint-600**만 유의
(EM 0.6063 vs 0.5958, p=0.041)해 LB 검증 → **0.91099로 승격**. ckpt400/800/1000은
비유의(각 0.5979/0.6032/0.6032). 이후 가중치 변경 없이 서빙(TTA)만 계속 갱신 — **2026-07-20
TTA3→균형 TTA4**(Klein 4원군, `scripts/tta.py` BALANCED4) **LB 0.91623**(+0.524pp) →
**2026-07-21 밤 TTA4→TTA5**(항등+4랜덤셔플, 균형 아님) **LB 0.91797**(+0.174pp, 573그리드
526/573) → **2026-07-22 TTA8**(`--tta 8 --tta-balanced8`, 균형 코셋 버전 확인됨 — 2026-07-22 사용자
확인) **LB 0.93019 — 현 챔피언**(573그리드 533/573, TTA5 대비 +7EM/+1.22pp —
이번 트랙에서 나온 서빙 변경 중 최대 단일 증분, TTA3→TTA4의 두 배 이상). 이후
**TTA8→TTA24**(`--tta 24`, 완전균형 S4 전체=`BALANCED24`, Klein V→Sylow-2(D4)
계열의 "대수적 완전성" 가설 검증용, `scripts/run_champ_tta24.sh`)를 LB 제출까지
실측했으나 **TTA8-balanced 대비 열세로 확인(2026-07-22, 사용자 확인 — 정확 LB
수치는 세션 로그 참고)** → 대수적 완전성 가설 기각, **TTA8이 이 계열의 실질
종점**으로 결론. 곧바로 별도 가지로 **TTA12**(교대군 A4=짝순열 12개 전부,
`ALTERNATING12`, `scripts/run_champ_tta12_alt.sh`)를 "홀순열이 모델을 헷갈리게
하는가" 가설을 격리 검증하기 위해 구현·실측(test 819건 완주, 에러 0, VRAM
~2h48m 중 대부분 23GB 초과 — 워치독 재기동 발동 없이 안정) — 챔피언(TTA8) 대비
로컬 일치율 EM 0.9768(19/819 다름, 쌍순서 0.9939)로 우위가 뚜렷하지 않아 **홀순열
가설도 기각 쪽**, "8 근처가 최적점"이라는 결론이 강화됨. **LB 미제출**(로컬
프록시상 우위 불분명해 우선순위 낮음, 제출 여부는 사용자 판단 보류). 상세 경위·게이트
판정 이력은 `../PROJECT_SUMMARY.md` §2 Ver8 항목, 공통 규정·함정은 `../CLAUDE.md` 참고.

⚠️ **스텝 스윕 추가 제출 금지(07-20 실측)**: ckpt600/800/1000의 test 819건 예측이 서로 4~7건만
달라, 스텝 변경으로 움직일 수 있는 LB 상한이 쌍순서 기준 ~0.14pp에 불과하다(서빙 변경 +0.52pp의
1/4). 하루 2회 제한 대비 기대값 미달 — 남은 슬롯은 Ver11/Ver12에 배분한다.

## 진입점

```bash
pytest tests/ -q
python -m snuai.train.train_dpo --adapter runs/sft32b_v4/adapter_final --no-hard-negative \
    --max-steps 1000 --save-steps 200 --out runs/dpo_v8
python -m snuai.infer.predict --csv data/test.csv --image-dir data/test \
    --strategy score24 --adapter <dpo checkpoint> --tta 3 --out runs/test_v8

# 멀티GPU(DDP, 2026-07-16) — device_map을 rank의 로컬 GPU로 고정, HF Trainer가 DDP 처리.
# --grad-accum(기본16)은 **전역** 누적으로 해석해 world_size로 나눔 → 유효배치는 단일GPU와
# 동일(=현 챔피언 레시피 그대로)하고 스텝당 forward가 rank로 분산돼 ~world_size배 빨라짐.
# 즉 2장으로 "2000스텝"은 단일GPU 2000스텝과 같은 데이터/배치를 절반 wall-clock에 처리.
torchrun --nproc_per_node=2 -m snuai.train.train_dpo --adapter runs/sft32b_v4/adapter_final \
    --max-steps 2000 --save-steps 200 --out runs/dpo_v8_ddp
# 현 챔피언(ckpt200)은 --no-hard-negative였음. DDP도 --no-hard-negative가 가장 단순·안전
# (records를 rank마다 동일하게 독립 생성, 스코어링 배리어 없음). --hard-negative를 쓰면
# rank0가 수십 분 스코어링하는 동안 다른 rank가 배리어 대기 → PG timeout 2h로 견디게 해둠.
```

## 저장소 구조

- `src/snuai/train/train_dpo.py` — 단일토큰 DPO(TRL 미사용, 참조모델은 `disable_adapter()`로 대체)
- `src/snuai/train/dpo_pairs.py` — 인접 스와프 hard-negative 선호쌍 생성. `--reversal-rejected`
  (2026-07-18)로 완전역전(d=6) rejected 추가 가능 — ckpt600 홀드아웃 오답 분석에서 확인된
  "확신에 찬 방향 역전"(d≥4 64건/완전역전 12건) 억제용. 결정적 생성이라 DDP 캐시 경로 호환.
- `scripts/diag_swap_submission.py` — 고마진 K건 인접 스와프로 EM/쌍순서 채점방식 가설 구분용 진단 제출 생성기. **2026-07-21 실제 제출 완료** — `runs/diag_swap60`(Ver7 ckpt1800+TTA3 베이스, K=60) 실측 LB 0.78534가 EM 가설(0.78537)과 일치, **EM 단독 채점 확정**(쌍순서 가설 0.84642는 기각). 상세는 `../CLAUDE.md` 공통 함정 절.

## 다음 액션

- checkpoint-400/600/800/1000이 A100에서 저장 중 — ckpt200보다 나은지 홀드아웃으로 먼저 비교 후
  LB 슬롯 사용(`../TODO.md` 참고).

## 이 버전 고유 함정

- DPO 참조 로그확률은 별도 참조모델 없이 `disable_adapter()`로 얻음(reference-free, 메모리 절약).
  ⚠️ **DDP에서 `model`은 `DistributedDataParallel` 래퍼라 `.disable_adapter()`가 없다**(래퍼는 이
  메서드를 forward하지 않음) → `model.disable_adapter()`는 AttributeError로 첫 스텝 크래시. 반드시
  `self.accelerator.unwrap_model(model).disable_adapter()`로 내부 PeftModel을 꺼내 호출해야 한다(TRL
  DPOTrainer도 동일). 단일프로세스에선 model이 PeftModel이라 원래도 동작했어서 이 버그가 안 드러났던 것
  (2026-07-16 재검토로 발견·수정). 참조 forward는 no_grad라 DDP를 안 거쳐도 결과 동일.
- `--hard-negative`(기본 on) 사전 스코어링이 로컬 4090에서 8,600건 순회에 80분+ 걸려 사실상 못 돎 —
  A100 또는 `--no-hard-negative`(스코어링 스킵, 무작위 rejected) 필요.
- **DDP(멀티GPU)에서 hard-negative 스코어링은 rank0만 수행**(그 비싼 forward pass를 world_size배
  중복하지 않으려고) — `dpo_pairs.build_dpo_records`의 `rejected_ranks_cache` 인자로 rank0가 만든
  `runs/.../hard_negative_cache.json`을 다른 rank들이 읽어 동일 records를 재구성(augment_sample이
  seed 고정 rng만 써서 rank 간 결정적으로 동일함을 이용). **`--include-random-rejected`는 이 DDP
  캐시 경로와 미지원**(rng 스트림이 rank마다 갈라질 위험) — 켜려면 단일 프로세스로 돌릴 것.
  `--ddp-find-unused-parameters`는 gradient-checkpointing+PEFT+DDP 조합에서 크래시 시에만 켤 것.
- **⚠️ 2026-07-16 재검토로 발견·수정된 메모리 누수**: 스코어링 후 `del scorer_eng`만으로는 GPU
  메모리가 안 풀렸다 — `scorer_fn`이 `Score24Scorer`(→ 그 안의 VLMEngine)를 클로저로 캡처하고 있어
  refcount가 안 떨어졌기 때문. `del scorer_eng, scorer_fn`으로 같이 지워야 `empty_cache()`가 실제로
  비운다(단일프로세스·DDP rank0 양쪽 다 수정). 특히 DDP에서 안 고쳤으면 rank0가 스코어링 엔진 메모리를
  들고 있는 채로 그 위에 학습용 모델을 또 올려야 해서, VRAM이 빠듯하면 rank0 학습 모델 로딩 시점에
  OOM 위험이 있었다.
- **⚠️ 2026-07-16 재검토로 발견·수정된 NCCL 배리어 타임아웃**: `--hard-negative` DDP 경로에서 rank0가
  수십 분 스코어링하는 동안 다른 rank는 `state.wait_for_everyone()`(NCCL 배리어)에서 대기하는데, NCCL
  워치독 **기본 타임아웃(~10분)**이면 스코어링이 끝나기 전에 그 배리어가 타임아웃돼 죽는다. main()
  진입 시 프로세스그룹을 **직접 `timeout=2h`로 초기화**(그 뒤 PartialState/HF Trainer가 재사용)해서
  스코어링을 견디게 고쳤다. `--no-hard-negative`면 스코어링 자체가 없어 이 경로를 안 탐(가장 안전).
- **유효배치/속도(2026-07-16)**: `--grad-accum`을 world_size로 나눠 per-device 누적으로 넘긴다 — HF
  Trainer+DDP는 rank 간 grad를 평균하므로, 안 나누면 유효배치가 world_size배가 되고(다른 레시피) 스텝당
  forward도 그대로라 wall-clock이 안 빨라진다. 나누면 유효배치는 단일GPU와 동일하고 스텝이 실제로
  빨라진다. 2x 배치를 원하면 `--grad-accum`을 2배로.
- **잔여 리스크(코드로 못 막음)**: rank0가 스코어링/저장 도중 죽으면 다른 rank는 배리어에서 PG
  타임아웃(2h)까지 대기 — 오래 멈춘 것 같으면 rank0 로그부터 확인할 것. 본런 전에 반드시 스모크(몇
  스텝)로 DDP 경로가 실제로 도는지 먼저 확인 권장.
- **score24는 순수 forward pass라 원칙적으로 결정적이어야 하는데 실측은 아니다**(2026-07-21~22
  밤 확인): 완전히 동일한 설정(`--tta 8 --tta-balanced8`, cascade 유무 무관)으로 test 819건을
  두 번 따로 돌렸더니 819건 중 15건(1.8%)이 그냥 달라짐 — GPU 커널(sdpa attention 등) 비결정성
  추정, 확정 원인 규명은 안 함. `score_agreement.py` 같은 로컬 프록시 일치율 지표를 해석할 때
  이 배경 노이즈(~1.8~2%)를 감안할 것 — 후보 간 일치율 차이가 이 폭 이내면 실질적으로 구분이
  안 될 수 있음(예: TTA12 vs 챔피언 EM 불일치 19/819=2.3%가 이 노이즈 폭과 크게 다르지 않음).
