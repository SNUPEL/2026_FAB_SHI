# Phase 1 / Phase 2 공통 DES 통합 개발 체크리스트

이 문서는 Phase 1과 merged Phase 2를 하나의 공통 환경, 제약 엔진, DES 상태, metric contract 위에 통합하기 위한 전용 실행 체크리스트다. 이 작업을 시작하는 개발자는 다른 설계 문서를 먼저 추론하지 말고 이 문서의 순서와 완료 조건을 따른다.

> 상태 관리: `현재과제_진행체크리스트.md`가 단일 master다. 이 문서는 공통 DES 통합의 상세 항목과 검증 증거를 대조하는 보조 체크리스트이며, 완료 표시는 코드·테스트·실행 결과가 모두 확인된 항목에만 한다.

## 0. 최종 목표

- [x] Phase 1과 Phase 2가 동일한 scenario Job/Machine mapping, Bay 용량, 제약 profile을 공통 환경에서 사용한다.
- [x] Phase 2 action mask와 final audit가 동일한 `PhaseConstraintProfile -> ConstraintManager` 경로를 사용한다. Phase 1 장척 mask는 block-Bay 전처리 계약으로 별도 유지한다.
- [x] Phase 1은 공통 planning state를 갱신하되 simulation clock은 전진시키지 않는다.
- [x] Phase 2는 별도 권위 simulator가 아니라 `CommonHierarchicalEnvironment` runtime/event state를 갱신한다.
- [x] Phase 2 batch close 시 공통 환경이 W/O별 `MACHINE_ASSIGN`, `CUT_BAY_ASSIGN`, `PROCESS_START`, `PROCESS_FINISH` event와 `batch_id` payload를 직접 생성한다.
- [x] 학습, validation, actual-8days, full-flow report가 `Environment.metrics.calculate_phase2_schedule_metrics()`를 공유한다.
- [x] actual-8days 평가는 `착수일 후보 블록.xlsx`의 8개 sheet를 authoritative block universe로 사용한다.
- [x] 독립 학습/full-flow 추론을 먼저 검증했고, 계층 학습은 frozen Phase 2 feedback 방식으로 구현했다. 동시 gradient joint update는 현재 범위에서 제외한다.

## 1. 변경하지 않을 확정 조건

- [x] Phase 1 action은 직접 pair action `(Block, Bay)`다.
- [x] Phase 1은 Bay별 설비 수를 capacity weight로 반영한다. NP 22/23/24 기준은 `4/3/4`다.
- [x] Phase 1의 현재 pair-pointer 구조는 유지한다.
- [x] Phase 2 action은 `SELECT_MACHINE -> SELECT_WO 반복 -> 자동 batch close`다.
- [x] Phase 2는 Phase 1의 block-to-Bay 결과를 기준으로 Bay별 subproblem으로 나눈다.
- [x] Phase 2 policy는 Bay별로 따로 만들지 않고 하나를 공유한다.
- [x] 활성 Bay가 3개면 parent episode 하나에서 최대 3개 Bay subproblem을 풀고 Bay별로 optimizer update를 수행한다.
- [x] batch는 W/O 1~3개다.
- [x] batch W/O `LTH` 합은 55,000 이하다.
- [x] batch 처리시간은 batch 내부 W/O `TACT_TIME`의 최댓값이다.
- [x] hard constraint 위반 action은 학습 대상 feature가 아니라 action mask로 제거한다.
- [x] family eligibility와 thickness range는 현재 확정된 Phase 2 제약이 아니므로 기본 비활성화한다.

## 2. 현재 구현 사실과 score contract

### 2.1 현재 Phase 2 학습은 Bay별 teacher 학습이다

현재 학습은 parent episode 전체 후보를 한 번에 비교하지 않는다. Phase 1 배정 후 각 Bay를 독립 subproblem으로 나누고 다음을 반복한다.

```text
for each active Bay b:
    heuristic candidates + agent greedy + agent samples 생성
    Bay b 내부 score로 best_b 선택
    best_b trajectory로 CE loss 계산
    optimizer update 1회
```

- [x] Bay별 teacher score의 현재 사전식 순서는 아래와 같다.

```text
S_b = (
    hard_violation_count_b,
    makespan_b,
    wo_count_gap_b,
    cut_length_gap_b,
    bevel_quantity_gap_b,
    occupancy_gap_b,
)
```

- [x] `makespan_b`는 해당 Bay 설비들의 완료시간 최댓값이다.
- [x] 각 gap은 해당 Bay 내부 설비들의 `max-min`이다.
- [x] normalized mode에서는 각 Bay 내부에서 `(max-min)/mean`을 사용한다.
- [x] Bay subproblem 하나만 평가하므로 teacher 선택 시에는 Bay gap의 합이 아니라 그 Bay의 gap 하나를 사용한다.

### 2.2 Parent episode score는 Bay별 결과를 합친다

모든 Bay의 `best_b`를 결합한 parent 결과는 아래처럼 집계한다.

```text
S_parent = (
    sum_b(hard_violation_count_b),
    max_b(makespan_b),
    sum_b(wo_count_gap_b),
    sum_b(cut_length_gap_b),
    sum_b(bevel_quantity_gap_b),
    sum_b(occupancy_gap_b),
)
```

- [x] 여기서 `max_b(makespan_b)`는 전체 공장 makespan과 같다.
- [x] Phase 2 부하평준화 metric은 모든 설비의 global max-min이 아니라 `Bay 내부 설비 gap의 합`이다.
- [x] “현재 score 순서를 유지한다”는 말은 위 tuple의 우선순위를 유지한다는 뜻이다.
- [x] 현재 CE teacher 선택은 Bay별로 수행되며, parent tuple 전체로 모든 Bay 후보 조합을 공동 탐색하는 방식은 아니다.
- [x] parent-global Cartesian teacher 조합은 Bay 후보 수의 곱으로 증가하며 현재 확정 구조와 다르므로 구현하지 않는다. 현재 계약은 Bay-local teacher + parent audit이다.

### 2.3 Full-flow report metric 통일

- [x] 전체 machine global max-min은 Phase 2 학습 순위 metric으로 사용하지 않는다.
- [x] global machine max-min은 `diagnostic_global_*` 필드로만 저장한다.
- [x] individual `processing_time_sum` gap과 batch max TACT 누적인 occupancy gap을 별도 필드로 구분한다.
- [x] 학습 순위에는 batch max TACT 누적으로 계산한 machine occupancy gap을 사용한다.
- [x] full-flow report는 `bay_metric_summary.csv`에 Bay별 raw/normalized gap을 출력하고 parent 합계를 evaluation summary에 남긴다.
- [x] full-flow에서 공통 metric으로 재계산한 score와 candidate `score_tuple`이 다르면 예외로 실패한다.

## 3. 목표 공통 환경 구조

### 3.1 단일 상태 소유자

공통 환경만 mutable state를 소유한다.

```text
Common Hierarchical Environment
├─ static state
│  ├─ Block
│  ├─ W/O
│  ├─ Bay
│  ├─ Machine
│  └─ config
├─ planning state
│  ├─ block_to_bay
│  ├─ Bay별 block/W/O/steel/cut/bevel 누적
│  └─ Phase 1 완료 여부
├─ runtime state
│  ├─ machine availability/clock
│  ├─ machine별 batch/WO/cut/bevel 누적
│  ├─ open batch
│  └─ completed batches
└─ event state
   ├─ simulation clock
   ├─ timeline
   └─ event log
```

- [x] Phase 1 pair rollout은 `HierarchicalPlanningState` 전이로 Bay load와 block assignment를 갱신한다.
- [x] Phase 2는 `CommonHierarchicalEnvironment.state.runtime` view에서 machine availability/load/open batch를 읽고 갱신한다.
- [x] static Job/Machine mapping은 fork간 동일 객체를 공유한다.
- [x] self-labeling 후보는 동일한 pristine post-Phase1 snapshot에서 시작한다.
- [x] snapshot/fork는 mutable planning/runtime/event state만 `deepcopy`한다.
- [x] rollout 종료 후 parent state가 오염되지 않는지 테스트한다.
- [x] 최종 결과는 선택 candidate의 완성 state만 export하며 다른 후보 state는 폐기한다.

### 3.2 Phase별 view

- [x] `Phase1View`는 Job/Bay ID와 planning load/assignment만 노출한다.
- [x] Phase 1 action은 simulation clock을 전진시키지 않는다.
- [x] `commit_phase1()`은 block-to-Bay, Bay load 필수 필드, block/W/O count, capacity weight를 검증한 뒤 고정한다.
- [x] Phase 1 planning 입력의 capacity weight, W/O/강재/베벨 정수값, 절단장 수치는 `NaN/inf`와 암묵적 정수 절삭 없이 검증한다.
- [x] `Phase2BayView`는 해당 Bay에 배정된 미스케줄 W/O와 해당 Bay 설비만 노출한다.
- [x] action이 Bay view 밖 machine/W/O를 참조하면 state builder가 실패시킨다.
- [x] Phase 2 state의 필수 machine load와 W/O 수치/TACT가 누락되거나 비정상이면 0으로 대체하지 않고 실패시킨다.
- [x] Bay 순서는 ID 정렬로 고정하고 설비/runtime은 Bay별로 격리한다.
- [x] 향후 Bay 간 공유자원 제약이 확정되면 독립 subproblem 가정을 재검토해야 함을 명시한다.

## 4. 공통 제약 엔진과 Phase별 profile

### 4.1 Phase 2 기본 hard profile

```yaml
phase2:
  hard_constraints:
    machine_enabled: true
    machine_bay_consistency: true
    block_set_same_bay: true
    machine_single_processing: true
    batch_wo_count_limit: true
    batch_length_sum_limit: true

    family_eligibility: false
    thickness_range: false
    table_length_limit: false
```

- [x] `config_np_100.yaml`에 `constraints.profiles.phase2` 계약을 명시한다.
- [x] `machine_enabled`는 비활성 설비 차단 안전장치로 유지한다.
- [x] `machine_bay_consistency`는 Phase 1이 정한 Bay 밖의 설비를 차단한다.
- [x] `block_set_same_bay`는 같은 블록의 W/O가 Phase 1 배정 Bay를 유지하도록 검사한다.
- [x] `machine_single_processing`은 동일 설비 batch interval 중복을 차단한다.
- [x] `batch_wo_count_limit`은 3개 이하를 검사한다.
- [x] `batch_length_sum_limit`은 W/O LTH 합 55,000 이하를 검사한다.
- [x] family, thickness, table length, calendar, breakdown은 Phase 2 profile에서 비활성화한다.
- [x] global config와 별개로 Phase 2 profile의 false rule은 action mask에 적용하지 않는다.

### 4.2 공통 평가 흐름

```text
candidate action 생성
→ ConstraintContext 생성
→ ConstraintManager hard rule 평가
→ hard 위반 후보 mask
→ policy/heuristic 선택
→ 공통 환경 transition
→ 종료 후 동일 규칙으로 final audit
```

- [x] 후보 mask와 최종 audit가 같은 profile/registry rule 함수를 사용한다.
- [x] 제약 평가 오류는 fail-open하지 않고 원인 출력 후 예외를 발생시킨다.
- [x] batch count/LTH는 공통 registry rule로 mask/audit하고, final audit에서 W/O 원본으로 count/LTH/max TACT를 독립 재계산한다.
- [x] hard violation candidate가 teacher bank에 들어가지 않는지 테스한다.

## 5. Phase 1 migration

### 5.1 유지할 state와 network

현재 Phase 1 pair feature 10개와 env feature 5개를 1차 migration에서 유지한다.

```text
Pair feature
- block steel quantity
- block cut length
- block bevel quantity
- candidate Bay current steel load
- candidate Bay current cut load
- candidate Bay current bevel load
- Bay capacity weight
- projected steel gap
- projected cut gap
- projected bevel gap

Environment feature
- progress ratio
- remaining block ratio
- current steel gap
- current cut gap
- current bevel gap
```

- [x] Phase 1 pair-pointer network 구조는 변경하지 않았다.
- [x] pair/env feature 계산과 action 반영은 공통 `HierarchicalPlanningState`를 사용한다.
- [x] Phase 1 checkpoint feature schema를 유지한다.
- [x] 100-block 고정 seed 회귀 테스트에서 migration 전 feature/action/assignment/load가 동일하다.

### 5.2 Phase 1 등가 회귀검증

- [x] 동일 100-block 입력의 candidate pair/feature 목록이 같다.
- [x] 동일 100-block 입력의 장척 hard mask가 같다.
- [x] projected steel/cut/bevel gap feature가 같다.
- [x] 동일 seed의 sample action index 열이 같다.
- [x] 최종 block-to-Bay assignment가 같다.
- [x] 최종 Bay별 부하와 score 입력이 같다.

## 6. Phase 2 state와 policy 교체

기존 12차원 flat action feature는 폐기한다. 기존 Phase 2 checkpoint 호환은 요구하지 않는다.

### 6.1 Bay context

- [x] 현재 action stage: `SELECT_MACHINE` 또는 `SELECT_WO`.
- [x] 전체 진행률.
- [x] 남은 W/O 비율.
- [x] 남은 TACT 합 비율.
- [x] 남은 LTH 합 비율.
- [x] 남은 절단장 합 비율.
- [x] 남은 베벨수량 합 비율.
- [x] 현재 Bay 내부 normalized W/O gap.
- [x] 현재 Bay 내부 normalized cut gap.
- [x] 현재 Bay 내부 normalized bevel gap.
- [x] 현재 Bay 내부 normalized occupancy gap.
- [x] 현재 Bay 설비 수 비율.

### 6.2 Machine node state

- [x] 현재 machine clock / 목표시간.
- [x] 현재 W/O 수 / 설비별 목표 W/O 수.
- [x] 현재 절단장 누적 / 설비별 목표 절단장.
- [x] 현재 베벨수량 누적 / 설비별 목표 베벨수량.
- [x] 현재 batch 수 / 예상 batch 수.
- [x] 해당 설비에서 처리 가능한 남은 W/O 비율.
- [x] 현재 선택된 설비 여부.
- [x] disabled 설비는 공통 hard mask에서 action set에 들어가지 않는다.

### 6.3 Remaining W/O node state

- [x] TACT_TIME 비율.
- [x] LTH / 55,000.
- [x] CUT_LTH 비율.
- [x] BV_QTY 비율.
- [x] 현재 open batch 포함 여부.
- [x] infeasible W/O는 node 값으로 우회하지 않고 action mask로 제거한다.
- [x] 계열/선호 edge는 확정 데이터가 없으므로 임의 값을 만들지 않는다.

### 6.4 Open batch state

- [x] 현재 W/O 수 / 3.
- [x] 현재 LTH 합 / 55,000.
- [x] 현재 최대 TACT 비율.
- [x] 현재 절단장 합 비율.
- [x] 현재 베벨수량 합 비율.
- [x] 남은 slot 수 / 3.
- [x] 남은 LTH 여유 / 55,000.

### 6.5 Action-conditioned projected feature

- [x] 선택 후 batch duration 증가량.
- [x] 선택 후 후보 설비 완료시간.
- [x] 선택 후 예상 makespan.
- [x] 선택 후 Bay 내부 normalized W/O gap.
- [x] 선택 후 Bay 내부 normalized cut gap.
- [x] 선택 후 Bay 내부 normalized bevel gap.
- [x] 선택 후 Bay 내부 normalized occupancy gap.
- [x] projected feature는 현재 state/action으로만 계산하며 teacher 결과를 누출하지 않는다.

### 6.6 Policy network

```text
Machine shared MLP encoder
+ W/O shared MLP encoder
+ Bay context encoder
+ Open batch encoder
+ machine/W/O embedding mean+max pooling
+ action-conditioned projected feature encoder
→ pointer action scorer
```

- [x] shared set encoder가 가변 machine 수를 처리한다.
- [x] shared set encoder가 가변 W/O 수를 처리한다.
- [x] machine/action 순서 permutation equivariance 테스트를 통과한다.
- [x] 현재 complete bipartite 관계에서는 GNN을 추가하지 않는다.
- [x] sparse eligibility/선호 edge가 확정될 때만 bipartite GNN을 별도 실험한다.
- [x] Phase 2 feature schema version을 checkpoint RunSpec에 저장한다.
- [x] legacy edge/schema checkpoint를 로드하면 명시적 mismatch 오류로 실패한다.

## 7. Phase 2 공통 DES transition

### 7.1 Transition 규칙

```text
SELECT_MACHINE
→ SELECT_WO 반복
→ 목표 batch 크기 또는 LTH 제약으로 자동 close
→ DES dispatch_batch
→ 설비 점유
→ batch start/end event 생성
→ machine availability 갱신
```

- [x] Phase 2가 `CommonHierarchicalEnvironment.open_batch/add_wo/close_batch` transition API를 사용한다.
- [x] Phase 2의 `machine_clock`/`machine_loads` 참조는 공통 runtime state view이며 별도 권위 state가 아니다.
- [x] W/O 추가 때마다 open batch count/LTH/max TACT/cut/bevel state를 갱신한다.
- [x] batch close 때 machine load와 occupancy를 한 번만 commit한다.
- [x] batch duration은 내부 W/O TACT 최댓값과 일치한다.
- [x] 같은 batch의 모든 W/O start/end 시각이 같다.
- [x] 같은 설비의 batch interval이 겹치지 않는다.
- [x] event log를 timeline에서 사후 재구성하지 않고 close transition에서 직접 생성한다.
- [x] 1분 tick을 사용하지 않고 machine availability와 batch start/finish event 시각을 직접 갱신한다.

### 7.2 기존 dispatch simulator 정리 결과

- [x] 현재 public Phase 2 후보 생성은 공통 environment transition만 사용한다.
- [x] dispatch cache 적용 전후 fixed candidate assignment/batch/score 동일성을 회귀 테스트한다.
- [x] assignment, batch, max TACT duration, start/finish, event identity, metric을 서로 독립된 audit/test로 검증한다.
- [x] 기존 별도 Phase 2 clock/timeline 생성 public path는 제거했다.

## 8. actual-8days 평가 조건 통일

### 8.1 단일 RunSpec

checkpoint/manifest에 아래 값을 저장한다.

- [x] `score_mode`.
- [x] score field 순서와 metric schema version.
- [x] `action_pool_limit`.
- [x] Phase 1 Bay capacity weights.
- [x] Phase 1 long-cut hard mask 설정.
- [x] Phase 2 constraint profile.
- [x] heuristic bank.
- [x] train rollout sample 수.
- [x] validation rollout sample 수.
- [x] feature schema version.

### 8.2 evaluator 적용

- [x] actual evaluator는 checkpoint RunSpec을 기본값으로 적용한다.
- [x] CLI override가 checkpoint 조건과 다르면 차이 필드를 출력하고 실패시킨다.
- [x] `action_pool_limit` 학습/평가 불일치를 제거한다.
- [x] Phase 1 `4/3/4` capacity weight를 RunSpec으로 평가에 전달한다.
- [x] normalized checkpoint를 raw ranking으로 평가하지 않는다.
- [x] validation/actual best-of-K sample 수를 RunSpec/manifest에 남긴다.
- [x] actual-8days block universe는 `착수일 후보 블록.xlsx`의 8개 sheet를 사용한다.
- [x] 해당 block의 전체 W/O는 WO 수정 원본에서 확장한다.

## 9. 공통 metric 모듈

- [x] Phase 2 학습, validation, actual evaluator, full-flow report가 동일한 score 함수를 import한다.
- [x] Bay별 metric을 먼저 계산한다.
- [x] parent metric은 Bay별 gap을 합산한다.
- [x] global makespan은 모든 machine clock의 최댓값으로 계산한다.
- [x] global machine max-min은 `diagnostic_global_*` 이름으로만 저장한다.
- [x] individual W/O TACT 합 gap과 batch occupancy gap을 다른 필드로 저장한다.
- [x] raw/normalized를 동시에 audit에 저장하되 ranking에는 `score_mode` 하나만 사용한다.
- [x] per-Bay CSV에 Bay ID, machine count, makespan, raw/normalized W/O/cut/bevel/occupancy gap을 저장한다.
- [x] parent long-format CSV/report에 Bay metric 합과 global makespan을 저장한다.

## 10. Full-flow inference와 학습

### 10.1 먼저 검증할 실행 모드

- [x] Phase 1 heuristic 단독.
- [x] Phase 1 frozen agent 단독.
- [x] Phase 1 heuristic -> Phase 2 heuristic.
- [x] Phase 1 frozen agent -> Phase 2 agent.
- [x] 공통 planning/runtime/event 계약으로 Phase 1 -> Phase 2 full-flow inference.

### 10.2 Full-flow용 Phase 1 lookahead feature: 후속 실험

현재 feature schema/checkpoint를 깨는 변경이며, 현재 frozen feedback이 이미 Phase 2 terminal score를 Phase 1 teacher에 반영한다. 따라서 아래 feature는 eligibility/선호 데이터가 확정된 후 ablation으로만 추가한다.

- 블록 W/O 수, TACT 합/최댓값/편차, LTH 합, 예상 batch 수.
- 후보 Bay 설비 수/capacity weight, Phase 2 lower bound.
- 확정된 eligibility가 있을 때의 처리 가능 설비 수.
- 없는 선호/eligibility 데이터는 0이나 임의값으로 생성하지 않는다.

### 10.3 Feedback/joint 학습

- [x] Phase 2 validation에서 휴리스틱, greedy, best-of-K를 동시 비교한다.
- [x] Phase 1 candidate마다 frozen Phase 2 Bay subproblem을 실행해 parent score를 계산한다.
- [x] Phase 1 self-label 후보 비교에 Phase 2 parent score를 명시적으로 prepend한다.
- [x] feedback on/off score 계약과 checkpoint SHA256/RunSpec pairing을 audit한다.
- [x] 독립 checkpoint를 frozen 상태로 연결하는 full-flow를 검증한다.
- [x] alternating/double-gradient update는 검증되지 않은 별도 알고리즘이므로 현재 범위에서 구현하지 않는다.

## 11. TDD 구현 순서

### Step 1: metric contract 고정

- [x] Bay 내부 gap과 parent 합계 테스트.
- [x] global gap이 ranking이 아닌 diagnostic인 테스트.
- [x] batch occupancy와 individual TACT 합이 다른 예제 테스트.
- [x] 공통 metric 함수 단일 경로 테스트.

### Step 2: Phase별 constraint profile

- [x] profile on/off 테스트.
- [x] batch 4개와 LTH 55,001 후보 mask 테스트.
- [x] family/thickness rule가 Phase 2 profile false일 때 비활성인 테스트.
- [x] disabled machine mask 테스트.

### Step 3: 공통 planning/runtime state

- [x] Phase 1 planning transition 테스트.
- [x] Phase 1 action이 clock을 변경하지 않는 테스트.
- [x] Phase 2 Bay view 격리 테스트.
- [x] snapshot/fork 무오염 테스트.

### Step 4: Phase 1 adapter migration

- [x] 기존 feature와 공통 state feature 등가 테스트.
- [x] 100-block Phase 1 고정 seed 회귀검증.

### Step 5: Phase 2 DES transition

- [x] batch start/end, max TACT, machine overlap 테스트.
- [x] dispatch cache 적용 전후 후보 결과 동일성과 최종 schedule 독립 audit 테스트.

### Step 6: Phase 2 state/policy

- [x] feature schema 테스트.
- [x] open batch/projected gap 수치 테스트.
- [x] permutation equivariance 테스트.
- [x] 고정 소형 문제 overfit loss 감소 테스트.

### Step 7: actual/full-flow parity

- [x] checkpoint RunSpec mismatch 실패 테스트.
- [x] actual-8days 8문제 checkpoint 평가 최종 재실행. 8문제·405후보, common audit 24,038행, hard 실패 0건.
- [x] 학습 score와 full-flow report 재계산 score 일치 테스트.

### Step 8: feedback/joint

- [x] frozen full-flow를 먼저 검증한다.
- [x] Phase 2 feedback off/on 시 Phase 1 learning score contract이 의도대로 달라지는지 테스트한다.
- [x] joint update는 frozen feedback과 다른 미검증 알고리즘이므로 현재 public path에 추가하지 않는다.

## 12. 예상 코드 소유 경계

- [x] `Environment/`: 공통 state, constraint context, DES transition, event, Phase 2 metric을 소유한다.
- [x] `Phase1/`: Phase 1 observation/action adapter, pair policy, self-labeling을 소유한다.
- [x] `Phase2/`: Bay view observation/action adapter, set-pointer policy, self-labeling/evaluation을 소유한다.
- [x] `Utils/`: Excel/CSV/scenario 변환과 순수 데이터 helper를 소유한다.
- [x] `scripts/`: 실행/분석/report 진입점만 소유하고 scheduling 규칙은 Phase 모듈을 호출한다.
- [x] Phase 2 metric 계산은 `Environment/metrics.py` 하나가 소유한다.
- [x] Phase 2 rollout용 상태 복제는 공통 환경 snapshot/fork API를 사용한다.

## 13. 검증 명령과 완료 조건

### 13.1 기본 회귀검증

```bash
python3 -X pycache_prefix=/tmp/pmsp_pycache -m py_compile main.py Phase1/*.py Phase2/*.py Environment/*.py Environment/constraints/*.py Utils/*.py
python3 -m unittest discover -v
python3 main.py show-config --config config_np_100.yaml
python3 main.py factory-summary --config config_np_100.yaml
python3 main.py simulate --config config_np_100.yaml --heuristic spt
python3 main.py simulate --config config_np_100.yaml --heuristic load_balance
python3 main.py factory-replay --config config_np_100.yaml
git diff --check
```

### 13.2 Phase 1 완료 조건

- [x] 100-block migration 전후 candidate/action/assignment/load 동일.
- [x] scheduled block 누락 없음.
- [x] 같은 블록의 W/O Bay 분리 없음.
- [x] Phase 1 checkpoint 재개/추론 가능.

### 13.3 Phase 2 완료 조건

- [x] generated schedule은 final common audit에서 hard violation 0일 때만 정상 결과로 처리한다.
- [x] 모든 W/O가 정확히 한 batch와 한 machine에 배정된다.
- [x] batch W/O 수 1~3.
- [x] batch LTH 합 55,000 이하.
- [x] batch duration이 내부 TACT 최댓값과 동일.
- [x] machine interval overlap 0.
- [x] event/timeline identity error 0.
- [x] Bay별 subproblem metric과 parent 합계가 정확히 일치.
- [x] Phase 2 고정 소형 문제 overfit loss 감소 확인.

### 13.4 actual/full-flow 완료 조건

- [x] actual-8days 입력 count 재확인: block `25/27/25/18/28/19/12/13`, W/O `276/207/244/164/184/137/121/98`.
- [x] checkpoint RunSpec과 evaluator RunSpec 일치.
- [x] 학습/validation/actual/full-flow의 score field와 계산식 일치.
- [x] global gap과 Bay 내간 gap 합을 구분하는 CSV/그래프 생성.
- [x] 실행 실패 시 fallback 없이 원인 출력 후 예외 발생.

## 14. 구현 중단 조건

- [x] Phase 1 100-block 회귀가 달라지면 Phase 2 작업을 멈추고 원인을 먼저 수정한다.
- [x] 공통 transition 결과/identity가 다르면 기존 경로 정리를 중단한다.
- [x] metric 재계산 결과가 candidate score와 다르면 예외로 실패한다.
- [x] actual evaluator 조건이 checkpoint와 다르면 실패한다.
- [x] hard violation이 하나라도 있으면 정상 결과로 표시하지 않는다.

## 15. 최종 구현 순서 요약

- [x] 1. 공통 metric contract와 테스트.
- [x] 2. Phase별 constraint profile과 공통 action mask.
- [x] 3. 공통 planning/runtime/event state와 snapshot.
- [x] 4. Phase 1 state 출처를 공통 planning state로 migration.
- [x] 5. Phase 2 dispatch를 공통 transition으로 migration.
- [x] 6. Phase 2 state와 set-pointer policy 교체.
- [x] 7. actual-8days RunSpec 통일.
- [x] 8. full-flow report metric 통일.
- [x] 9. Phase 1/Phase 2 독립 smoke 학습 및 validation.
- [x] 10. frozen full-flow inference.
- [x] 11. Phase 2 feedback 기반 Phase 1 self-labeling.
- [x] 12. joint/alternating은 현재 계약에 필요하지 않은 별도 연구로 판단.
