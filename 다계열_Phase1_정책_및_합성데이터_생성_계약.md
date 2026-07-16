# 다계열 Phase 1/2 정책·데이터·DES 실행 계약

이 문서는 현재 실행 코드의 authoritative contract다. 진행 상태는
`현재과제_진행체크리스트.md`, 실행 명령은 `README.md`를 함께 본다. 과거 루트 md와
발표 문서는 설계 근거 archive이며 이 문서와 충돌할 때 현재 코드 계약으로 사용하지
않는다.

## 1. 현재 공개 범위

- 공개 학습 데이터: `MIXED` physical-block joint-distribution episode
- Phase 1: `(PROJ_NO, GYEL, BLK_NO) -> Bay`
- Phase 2: `SELECT_MACHINE -> SELECT_WO`를 반복하는 batch-machine scheduling
- 공통 환경: `Environment.hierarchical.CommonHierarchicalEnvironment`
- 실행 엔진: SimPy가 아닌 custom event-driven DES
- Phase 1 scope: `joint_five_bay_v2_mapped_eqp`
- Phase 1 rule profile: `multi_series_260711`
- Phase 1 score mode: `wo_first`
- silent fallback: 허용하지 않음

과거 NP-only pair-policy, imitation, 분리 Phase 2.1/2.2/3 실행 경로와
`joint_five_bay_v1` checkpoint는 지원하지 않는다.

## 2. 설비 identity와 EQP 매핑

### 2.1 Planning factory

| Bay | planning machine | 수 | 허용 계열 |
|---|---|---:|---|
| 22 | PLS21, PLS22, PLS23, PLS24 | 4 | NP, NC |
| 23 | PLS31, PLS32, PLS33 | 3 | NP, NC |
| 24 | PLS41, PLS42, PLS43, PLS44 | 4 | NP, NC |
| 25 | PLS51, PLS52 | 2 | FN, FL |
| trans | PLP01, PLP02 | 2 | FN, FL |

Phase 1 용량비와 Phase 2 machine 수는 이 표를 별도 숫자로 복사하지 않고
`MIXED_PLANNING_MACHINE_IDS_BY_BAY`에서 직접 계산한다. 따라서 현재 용량비는
`4:3:4:2:2`, planning machine은 총 15대다.

### 2.2 Actual EQP mapping

| Actual ID | Machine | Actual ID | Machine |
|---|---|---|---|
| EQP_1 | PLS51 | EQP_9 | PLS32 |
| EQP_2 | PLS52 | EQP_10 | PLS33 |
| EQP_4 | PLS21 | EQP_11 | PLS41 |
| EQP_5 | PLS22 | EQP_12 | PLS42 |
| EQP_6 | PLS23 | EQP_13 | PLS43 |
| EQP_7 | PLS24 | EQP_14 | PLS44 |
| EQP_8 | PLS31 | EQP_15 | PLP01 |
|  |  | EQP_16 | PLP02 |

`EQP_3`은 일반 mapping에 넣지 않는다. 신규 W/O 403건에서 모두
`GYEL=NC`, `CUT_BAY=trans`로 관측된 actual-only 설비다.

- actual audit: `MAPPED_MACHINE_ID=EQP_3`, `MACHINE_HOME_BAY=trans`
- planning candidate: `False`
- Phase 1 NC candidate Bay: 22, 23, 24
- Phase 2 NC candidate machine: 선택된 Bay의 PLS 설비

미등록 EQP나 `EQP_3`의 계열/Bay가 NC/trans가 아닌 행은 예외로 종료한다. source
`CUT_BAY`와 매핑 machine home Bay가 달라도 원본을 고치거나 다른 설비로 대체하지
않고 `SOURCE_BAY_MATCHES_MACHINE_HOME=False`로 기록한다.

## 3. 데이터 identity와 집계

### 3.1 Identity

- 물리 블록: `(PROJ_NO, BLK_NO)`
- Phase 1 block-series: `(PROJ_NO, GYEL, BLK_NO)`
- W/O: `WK_ORD_NO`
- 같은 block-series의 모든 W/O는 같은 Bay로 간다.
- 같은 물리 블록에 NP/FN/FL/NC가 함께 있어도 계열별 Bay 결정은 독립적이다.

### 3.2 신규 block-W/O 집계

- 최댓값: `LTH`, `BTH`, `THK`, `WGT`, `MARK_LTH`
- 합계: `CUT_LTH`, `BVL_LTH`, `PTLST_QTY`, `STL_QTY`, `BV_QTY`, `CURVE_QTY`
- `WO_QTY`: 해당 block-series의 W/O 행 수
- `STL_QTY`: W/O `STL_QTY` 합
- `WO_QTY`와 `STL_QTY`는 서로 대체하지 않는다.

필수 컬럼 누락, block-W/O FK 불일치, 집계 identity 불일치는 원인을 출력하고
실패한다. `RET_QTY`는 모델 입력, 목적, 제약에서 사용하지 않는다.

## 4. MIXED 합성데이터

한 episode는 다음 순서로 생성한다.

1. 실적 물리 블록에서 관측한 계열 조합과 계열별 공동 percentile donor를 뽑는다.
2. NP는 발표자료 고정 수식, FN/FL/NC는 동일 계층 흐름의 계열별 empirical
   profile로 block/W/O 주변분포를 생성한다.
3. 생성 계열별 block을 donor의 공동 rank에 최소비용 일대일 매칭한다.
4. 동일 physical-block identity 아래에 계열별 block-series와 W/O를 연결한다.
5. 모든 집계 identity, 계열 조합, W/O identity를 검증한다.

실적 값을 생성 정답으로 복사하지 않는다. donor는 계열 공동관계를 정하는 데만
사용한다. 한 episode에 특정 계열이 없을 수 있으며, 이는 실적 계열 조합 분포의
일부다.

전 계열 W/O `TACT_TIME`은 현재 확정된 Case 6 식을 사용한다.

```text
TACT_TIME = 0.3037 * CUT_LTH
          + 0.1325 * MARK_LTH
          + 0.4790 * THK
          + 0.3840 * PTLST_QTY
```

`BV_QTY`를 TACT 식에 임의로 추가하지 않는다. actual start-end elapsed time도
TACT_TIME 대체값으로 사용하지 않는다.

## 5. Phase 1 계약

### 5.1 Action과 hard mask

한 step action은 현재 feasible한 `(block-series, Bay)` edge 하나다.

- NP: Bay 22/23/24
- NC: Bay 22/23/24
- FN/FL: Bay 25/trans
- NP `BTH > 4500`: Bay 22/23
- NP CNT block: Bay 22/23
- NP `CUT_LTH >= 1000`: Bay 22/23

동일 block-series를 두 Bay로 나누는 action은 생성하지 않는다. feasible edge가
없으면 임의 Bay를 복원하지 않고 실패한다.

### 5.2 Pair feature 16차원

1. block W/O 수 비율
2. block CUT_LTH 비율
3. block BV_QTY 비율
4. NP balancing-group indicator
5. FN+FL balancing-group indicator
6. NC balancing-group indicator
7. NP 광폭 indicator
8. NP CNT indicator
9. NP 장척 indicator
10. 후보 Bay의 현재 W/O/설비수 부하
11. 후보 Bay의 현재 CUT_LTH/설비수 부하
12. 후보 Bay의 현재 BV_QTY/설비수 부하
13. 후보 Bay 설비수 비율
14. 선택 후 W/O 정규화 gap
15. 선택 후 CUT_LTH 정규화 gap
16. 선택 후 BV_QTY 정규화 gap

### 5.3 Environment feature 5차원

1. 전체 진행률
2. 남은 block-series 비율
3. 현재 W/O 정규화 gap
4. 현재 CUT_LTH 정규화 gap
5. 현재 BV_QTY 정규화 gap

### 5.4 Score

각 후보 완성 계획은 `NP`, `FN+FL`, `NC` 그룹별 유효 Bay에서 설비 수로 정규화한
max-min gap을 계산하고 그룹별 gap을 합산한다.

```text
(W/O count gap, CUT_LTH gap, BV_QTY gap)
```

비교는 weighted sum이 아닌 사전식이다. Phase 1 policy는 pair와 environment
feature를 MLP로 encoding한 뒤 모든 feasible edge에 pointer-style score를 준다.
후보 수와 Bay 수가 변해도 같은 가중치를 공유한다.

## 6. Phase 2 계약

### 6.1 Action과 batch

각 Bay는 독립 subproblem으로 풀지만 하나의 Phase 2 policy를 공유한다.

1. `SELECT_MACHINE`: 현재 Bay의 eligible machine 하나를 선택한다.
2. `SELECT_WO`: 남은 eligible W/O 하나를 open batch에 추가한다.
3. batch close: 최대 3개 또는 `LTH` 합 55,000 한계와 남은 W/O 상태에 따라
   환경이 닫는다.

한 batch에는 서로 다른 block과 서로 다른 계열이 섞일 수 있다. 단 선택 machine이
batch의 모든 계열을 처리할 수 있어야 한다. batch 처리시간은 포함 W/O
`TACT_TIME`의 최댓값이며, 같은 batch는 같은 시각에 시작하고 종료한다.

### 6.2 State

Bay context 13차원은 stage, 진행률, 남은 W/O/TACT/LTH/CUT/BV 비율과 현재
W/O/CUT/BV/점유시간 gap, machine 수 비율을 포함한다.

Machine node 11차원은 clock, W/O, CUT, BV, batch 수의 target 대비 비율,
남은 feasible W/O 비율, 현재 선택 여부, NP/FN/FL/NC eligibility를 포함한다.

W/O node 9차원은 TACT, LTH limit, CUT, BV 비율, open-batch 포함 여부와
NP/FN/FL/NC family indicator를 포함한다.

Open-batch 7차원은 W/O 수, LTH 합, 최대 TACT, CUT 합, BV 합, 남은 slot,
남은 LTH capacity를 포함한다.

각 projected action은 duration 증가, finish time, makespan, 정규화
W/O/CUT/BV/점유시간 gap 7개를 추가로 계산한다. 정책은 이 node/context/action
feature를 MLP set/pointer-style scorer로 평가하며 message-passing GNN은 사용하지
않는다.

### 6.3 Score

현재 raw teacher score는 다음 사전식 순서다.

```text
(
  hard violation count,
  makespan,
  Bay 내부 CUT_LTH gap,
  Bay 내부 W/O count gap,
  Bay 내부 BV_QTY gap,
  Bay 내부 occupancy gap
)
```

`normalized` score mode도 제공하지만 checkpoint RunSpec과 실행 옵션이 정확히
일치해야 한다. 각 Bay teacher는 해당 Bay subproblem 후보 bank에서 score가 가장
작은 시퀀스를 self-label로 선택하고 CE/NLL update를 수행한다.

## 7. Self-labeling

Phase 1과 Phase 2 모두 고정 실적 정답 imitation이 아니다.

1. episode마다 새 MIXED synthetic 문제를 생성한다.
2. dispatching heuristic 후보, agent greedy, agent sampled rollout을 만든다.
3. 동일 hard mask와 score로 후보를 비교한다.
4. best candidate의 action index sequence를 pseudo-label로 삼는다.
5. 선택 action의 cross entropy/NLL을 최소화한다.

Phase 1에서 frozen Phase 2 feedback을 켤 때는 Phase 2 checkpoint와 RunSpec을
명시해야 한다. Phase 2 gradient를 Phase 1에 전달하지 않고 frozen score만 Phase 1
teacher 비교의 앞쪽에 붙인다. checkpoint SHA256과 계약을 함께 저장한다.

## 8. DES와 event

Phase 1은 시간축 없는 Bay 계획 단계다. Phase 2가 machine clock과 batch event를
생성한다. full flow는 같은 episode의 Phase 1 결과를 Phase 2에 전달하고 공통
환경에서 다음 event를 기록한다.

- batch/open 또는 dispatch decision
- W/O assignment
- machine start
- machine finish

현재 구현은 1초/1분 tick loop가 아니라 다음 decision/event 시점으로 clock을
전진하는 custom event-driven DES다. generated schedule과 actual replay는 공통
event schema를 사용한다.

## 9. 엄격한 실패와 남은 경계

- unknown EQP, missing mapping, missing column, invalid date, formula failure는 예외다.
- 평균, 0, 첫 machine, 임의 Bay로 조용히 대체하지 않는다.
- hard constraint 평가 오류를 pass로 처리하지 않는다.
- source Bay와 mapped home Bay mismatch는 actual data-quality audit이며 source를
  수정하지 않는다.
- 코드·정적 계약과 짧은 smoke 검증은 장시간 학습 완료를 의미하지 않는다.

남은 외부 검증은 두 가지다.

1. 신규 W/O 전체 actual time-axis replay에서 mapped 15대와 `EQP_3`을 분리한 비교
2. 사용자가 수행할 장시간 Phase 1/2 학습과 holdout/full-scale 성능 검증
