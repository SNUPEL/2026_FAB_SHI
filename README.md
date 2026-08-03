# PMSP 절단 공장 스케줄링

NP/FN/FL/NC 물리 블록을 대상으로 Phase 1 Block-Series-to-Bay 배정과 Phase 2
W/O batch-to-Machine 스케줄링을 수행하는 프로젝트입니다. 공개 학습 경로는
`MIXED` 하나이며, 과거 NP-only 학습·imitation·3-Bay 정책은 지원하지 않습니다.

문서는 다음 순서로 읽습니다.

1. 진행 상태와 남은 작업: [`현재과제_진행체크리스트.md`](현재과제_진행체크리스트.md)
2. 실행 코드·state·action·feature·학습·데이터 계약:
   [`다계열_Phase1_정책_및_합성데이터_생성_계약.md`](다계열_Phase1_정책_및_합성데이터_생성_계약.md)
3. 장시간 학습·재개·Phase 2 upstream 비교 실행:
   [`다계열_Phase1_Phase2_학습_실행_가이드.md`](다계열_Phase1_Phase2_학습_실행_가이드.md)
4. 빠른 실행 방법: 이 README

과거 설계와 실행 이력은 별도 문서로 중복 보관하지 않고 Git history에서 확인합니다.

## 문제 정의

### Phase 1

- 의사결정: `(PROJ_NO, GYEL, BLK_NO) -> Bay`
- 동일 물리 블록이라도 계열이 다르면 별도 block-series 작업으로 취급합니다.
- 동일 block-series의 모든 W/O는 반드시 같은 Bay로 갑니다.
- Bay: `22`, `23`, `24`, `25`, `trans`
- 설비 수 기반 용량비: `4:3:4:2:2`
- 부모 episode 입력은 5개 Bay를 모두 포함하지만 학습 문제는 두 개로 분리:
  - `NP_NC`: NP/NC block-series만 Bay 22/23/24에 배정
  - `FN_FL`: FN/FL block-series만 Bay 25/trans에 배정
- 평준화 계열: `NP`, `NC`, `FN`, `FL` 네 개를 서로 따로 계산
- `series_only` teacher score:
  - `NP_NC = (NP W/O gap + NC W/O gap, NP CUT gap + NC CUT gap, NP BV gap + NC BV gap)`
  - `FN_FL = (FN W/O gap + FL W/O gap, FN CUT gap + FL CUT gap, FN BV gap + FL BV gap)`
- 두 score는 서로 비교하거나 합쳐 teacher를 선정하지 않습니다. 두 서브문제가
  있으면 하나의 부모 episode에서 같은 policy를 순서대로 2번 CE update합니다.
- hard mask:
  - NP/NC: Bay 22/23/24
  - FN/FL: Bay 25/trans
  - NP block-series의 W/O `CUT_LTH` 합 `>= 1000`, `BTH > 4500`, CNT block: Bay 22/23

Phase 1 action은 현재 서브문제에서 가능한 `(block-series, Bay)` edge 하나를
선택합니다. 정책 입력은 NP/NC/FN/FL을 별도 flag로 표현한 20차원 pair
feature와 8차원 환경 feature이며, 후보 수와 Bay 수에 독립적인
pointer-style scorer입니다.

### Phase 2

- 학습 action: 환경이 확정한 설비의 open batch에 추가할 `W/O` 하나 선택
- 설비 dispatch: 현재 전역 시각에 실행 가능한 유휴 설비 중 `machine_id` 오름차순
- 이벤트 시계: 유휴 설비가 없을 때만 전체 설비의 다음 최소 완료 시각으로 점프
- batch W/O 수: 1~3개
- batch `LTH` 합: 55,000 이하
- batch 처리시간: 포함 W/O `TACT_TIME`의 최댓값
- 동일 batch의 W/O는 같은 시점에 시작하고 종료합니다.
- 사전식 score:
  `hard violation -> makespan -> Bay 내부 CUT_LTH gap 합 -> W/O 수 gap 합 -> BV_QTY gap 합 -> 점유시간 gap 합`

각 부하 gap은 Bay 안 설비의 `max-min`으로 계산한 뒤 Bay별 값을 합합니다. 점유시간
gap도 마지막 사전식 tie-break 항목이며 진단 CSV/PNG에 함께 저장합니다.

설비 선택은 sampling·logit·CE target이 아닙니다. 환경이 설비와 목표 batch 크기를
결정한 뒤, 단일 Set-Pointer policy가 해당 설비에 투입 가능한 W/O 후보만 scoring합니다.
batch close는 해당 설비의 `machine_available_at`만 갱신하므로 여러 설비와 여러 Bay의
첫 batch가 `t=0`에 병렬 시작할 수 있습니다. 모든 설비가 점유된 뒤에만 전역 시계가
다음 완료 이벤트로 이동하며, 모든 W/O 배정 후에는 남은 완료 이벤트를 drain합니다.

Phase 2는 확정된 PLS/PLP 15대를 사용합니다.

```text
Bay 22 : PLS21, PLS22, PLS23, PLS24
Bay 23 : PLS31, PLS32, PLS33
Bay 24 : PLS41, PLS42, PLS43, PLS44
Bay 25 : PLS51, PLS52
trans  : PLP01, PLP02
```

위 15대가 유일한 factory topology입니다. config가 아니라
`Utils/data/multi_series_cutting_data.py`의 `MIXED_PLANNING_MACHINE_IDS_BY_BAY`가
단일 기준이며, `config_mixed.yaml`은 Phase 2 제약 profile만 제공합니다.

실적 `EQP_1~EQP_16`은 위 identity로 엄격히 매핑합니다. 단 `EQP_3`은 신규 데이터
403건 모두 `NC + trans`인 실적 전용 설비이므로 actual identity에는 보존하지만,
NC planning 후보에는 넣지 않습니다. NC는 Phase 1에서 Bay 22/23/24로 배정되고
Phase 2에서 해당 Bay의 PLS 설비만 선택합니다.

## MIXED 합성데이터

공개 생성기는 `Utils/data/multi_series_formula_data_generator.py`이며, 학습 파라미터는
`Utils/data/multi_series_generation_profile.json`에 고정되어 있습니다.

1. 발표자료 고정식으로 물리 블록 목표 특성과 전체 `WO_QTY`를 먼저 생성합니다.
2. 전체 `WO_QTY`로 배분 가능한 실적 계열 조합 하나를 실적 확률로 표본화합니다.
3. 물리 블록 특성이 가까운 실적 count-vector profile을 이용해 전체 `WO_QTY`를
   선택된 계열에 양의 정수로 배분합니다.
4. 확정된 계열별 `WO_QTY`를 각 계열 생성기에 입력해 W/O 특성을 생성합니다.
5. 생성 W/O를 `(PROJ_NO, BLK_NO, GYEL)`로 역집계하고 block-series identity를
   엄격히 검증합니다.
6. Phase 1과 Phase 2는 같은 생성 episode의 W/O를 사용합니다.

```text
WO_QTY = max(
    1,
    round((0.01203 * CUT_LTH + 2.014) * Gamma(shape=7.616, scale=0.128))
)

N_NP + N_FN + N_FL + N_NC = WO_QTY
```

계열별 생성기가 W/O 수를 독립 생성한 뒤 합치거나, 생성 후 block swap·Hungarian
매칭·임의 비율 보정을 수행하지 않습니다. 배분이나 집계가 불가능하면 다른 값으로
대체하지 않고 원인을 출력한 뒤 실패합니다.

NP component는 발표자료의 고정 수식을 사용합니다. FN/FL/NC는 동일한 계층 생성
흐름과 계열별 empirical profile을 사용합니다. 전 계열 `TACT_TIME`은 현재 확정된
Case 6 식을 공통 적용합니다.

```text
TACT_TIME = 0.3037*CUT_LTH + 0.1325*MARK_LTH
            + 0.4790*THK + 0.3840*PTLST_QTY
```

`WO_QTY`와 `STL_QTY`는 서로 다른 변수이며 대체하지 않습니다.

### 고정 generation profile

MIXED Phase 1/2 학습은 원천 Excel을 실행 중 다시 적합하지 않고 다음 JSON만
읽습니다.

```text
Utils/data/multi_series_generation_profile.json
```

이 파일에는 NP 고정식 계약, NP BTH/STL profile, FN/FL/NC 계수·입력분포·잔차분포,
물리 블록 계열 조합과 조건부 W/O 수 분포, 원천 파일 SHA256이 포함됩니다.
profile 누락, schema 불일치, 계열 누락은 Excel fallback 없이 실패합니다.

원천 Excel 또는 적합 방법이 변경된 경우에만 profile을 명시적으로 재생성합니다.

```bash
python scripts/build_multi_series_generation_profile.py
```

같은 원천에서 재생성한 JSON은 byte 단위로 동일해야 하며, 변경된 JSON은 코드와
함께 Git에서 검토합니다.

## 주요 실행

Windows conda 환경에서는 아래 명령의 `python`을 그대로 사용합니다.

### 다계열 실제 데이터 검증 및 Phase 1 일별 계획

한국 법정·대체 공휴일 계산에 `holidays`가 필요합니다. 없으면 기본 달력으로
대체하지 않고 실패합니다.

```bash
python -m pip install holidays
python main.py phase1-plan-multi-series \
  --block-xlsx "input/260724_절단블록_데이터_None.xlsx" \
  --wo-xlsx "input/260724_절단WO_데이터_None.xlsx" \
  --output-dir output/generated/multi_series_260711
```

### MIXED 합성데이터 생성

```bash
python main.py generate-phase1-blocks \
  --n-blocks 100 \
  --seed 2026 \
  --output-dir output/generated/phase1_mixed_joint
```

### Phase 1 self-labeling 학습

```bash
python main.py phase1-train-pair-self-labeling \
  --config config_mixed.yaml \
  --episodes 20000 \
  --min-blocks 12 \
  --max-blocks 80 \
  --rollout-samples 64 \
  --rollout-samples_validation 64 \
  --heuristic-algorithms all \
  --hidden-dim 128 \
  --checkpoint-every 100 \
  --validation-every 100 \
  --validation-episodes 20 \
  --device cuda \
  --objective-scope series_only \
  --output-dir output/phase1_mixed_resource_pool_v1
```

재개 학습은 같은 인자와 output 경로를 유지하고 아래 옵션을 추가합니다.

```bash
--resume-checkpoint latest
```

### Phase 2 self-labeling 학습

Phase 1 휴리스틱을 upstream으로 고정하는 예시입니다.

```bash
python main.py phase2-train-batch-machine-self-labeling \
  --config config_mixed.yaml \
  --phase1-heuristic bevel_first_balanced \
  --episodes 20000 \
  --min-blocks 12 \
  --max-blocks 80 \
  --rollout-samples 64 \
  --rollout-samples_validation 64 \
  --phase2-score-mode raw \
  --hidden-dim 128 \
  --action-pool-limit None \
  --checkpoint-every 100 \
  --validation-every 100 \
  --validation-episodes 20 \
  --device cuda \
  --output-dir output/phase2_mixed
```

Phase 2는 매 checkpoint마다 **두 개의 고정 validation**을 수행합니다.

- **MAIN validation (정식 지표, in-distribution)** — 학습분포(블록 `--min-blocks..--max-blocks`,
  target-matching 없는 자연표본)의 고정 문제. `<output>/validation/`에 기록하며 **best checkpoint 선택
  기준**입니다. `--main-validation-{min-blocks,max-blocks,block-gap,episodes}`로 조정(기본값은 학습 블록범위와
  `--validation-episodes`).
- **generalization test (리포팅 전용)** — 기존 `--validation-*` 플래그가 만드는 넓은 grid(예: blocks 10-100,
  distribution-type target-matched). `<output>/validation_generalization/`에 기록하며 best 선택에는 관여하지
  않습니다.

MAIN validation 문제는 held-out seed로 생성되어 **학습 데이터·generalization 문제와 겹치지 않음이 보장**되며
(학습 시작 시 `validation_overlap_check`로 assert), 겹치면 즉시 오류를 냅니다.

Phase 1 checkpoint를 upstream으로 고정할 때는 `--phase1-heuristic` 대신 다음을
사용합니다.

```bash
--phase1-checkpoint output/phase1_mixed_resource_pool_v1/phase1_pair_pointer_best.pt \
--phase1-samples 32
```

Phase 2 재개 학습도 같은 RunSpec 인자를 유지하고 `--resume-checkpoint latest`를
추가합니다.

### Phase 1 -> Phase 2 full flow

휴리스틱 smoke 예시:

```bash
python main.py phase2-run-full-workflow \
  --config config_mixed.yaml \
  --phase1-heuristic bevel_first_balanced \
  --batch-machine-heuristic lpt_batch \
  --synthetic-blocks 30 \
  --output-dir output/full_flow_mixed
```

학습 checkpoint 평가 예시:

```bash
python main.py phase2-run-full-workflow \
  --config config_mixed.yaml \
  --phase1-checkpoint output/phase1_mixed_resource_pool_v1/phase1_pair_pointer_best.pt \
  --phase1-samples 64 \
  --batch-machine-checkpoint output/phase2_mixed/phase2_batch_machine_policy.pt \
  --synthetic-blocks 30 \
  --output-dir output/full_flow_mixed_checkpoint
```

Checkpoint 실행은 저장된 RunSpec의 score mode, batch limit, action pool,
heuristic bank, sampling 수, Phase 1 용량비와 제약 profile이 다르면 실패합니다.
과거 `joint_five_bay_*` checkpoint는 19차원 pair와 단일 teacher 계약을 사용하므로
재사용하지 않습니다. 현재 공개 scope는
`resource_pool_subproblems_v1`, feature는 `20+8`입니다. frozen checkpoint 추론은
각 자원군에서 best-of-K를 독립 선정한 뒤 두 assignment를 합칩니다.

## 구조

- `Environment/`: 공통 state, 제약, metric, custom event-driven DES
- `Phase1/`: MIXED Block-Series-to-Bay 휴리스틱·pair policy·self-labeling
- `Phase2/`: merged batch-machine state·set pointer policy·self-labeling·full flow
- `Utils/data/`: strict loader, MIXED 공동분포 생성, 고정 generation profile
- `Utils/phase1/`: 다계열 규칙, Bay balancer, episode builder
- `Utils/learning/`: Phase 간 message/checkpoint 계약
- `Train/network/mlp.py`: Phase 1/2 정책이 공유하는 MLP builder

SimPy는 사용하지 않습니다. 현재 환경은 decision epoch와 machine clock을 직접
전진시키는 custom event-driven DES이며 generated schedule과 replay가 동일 event
schema를 사용합니다.

## 검증

코드 변경 후 최소 회귀검증:

```bash
python -X pycache_prefix=/tmp/pmsp_pycache -m py_compile \
  main.py Environment/*.py Environment/constraints/*.py \
  Phase1/*.py Phase2/*.py Utils/data/*.py Utils/learning/*.py Utils/phase1/*.py \
  Train/network/*.py scripts/*.py
python -m pytest tests/ -q
python main.py generate-phase1-blocks --n-blocks 40 --seed 0 --output-dir output/_check/blocks
python main.py phase1-plan-multi-series \
  --block-xlsx input/260724_절단블록_데이터_None.xlsx \
  --wo-xlsx input/260724_절단WO_데이터_None.xlsx \
  --output-dir output/_check/plan
git diff --check
```

성공 기준은 pytest 전건 통과, `generate-phase1-blocks`가 NP/FN/FL/NC 4계열을 모두
생성, `phase1-plan-multi-series`가 전처리본 블록 1,137건과 W/O 6,644건을 읽고
problem을 만드는 것입니다. 학습 경로까지 확인하려면 `AGENTS.md` §12의 짧은 Phase 1/2 학습
명령을 실행합니다.

## 엄격한 실패 원칙

- 누락 컬럼, 날짜 파싱, 처리시간, machine/Bay mapping 실패는 예외로 종료합니다.
- 없는 데이터를 평균·0·첫 machine으로 대체하지 않습니다.
- hard constraint 평가 실패를 pass로 처리하지 않습니다.
- 미등록 EQP, `EQP_3`의 NC/trans 계약 위반, machine/Bay 매핑 실패는 예외로 종료합니다.
- source `CUT_BAY`와 매핑 설비의 home Bay가 다르면 원본을 바꾸지 않고 audit 필드로 남깁니다.
