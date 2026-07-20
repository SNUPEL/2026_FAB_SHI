# Phase 2 MDP: W/O Batch 구성 및 설비 배정

> Phase 2는 Phase 1에서 Bay가 확정된 W/O를 해당 Bay 내부 설비에 배정하고, W/O 1~3개를 하나의 batch로 구성하는 단계이다.

## 1. 입력과 출력

| 구분 | 내용 |
|---|---|
| 입력 | Phase 1의 `block-series -> Bay` 배정과 개별 W/O |
| 의사결정 단위 | Bay 내부의 W/O, 설비, batch |
| 출력 | W/O별 설비, batch, 시작·종료 시간 |
| 처리 단위 | W/O 1~3개로 구성된 batch |

하나의 episode를 Bay별 subproblem으로 나누어 풀지만, 모든 Bay는 동일한 Phase 2 정책 하나를 공유한다.

## 2. MDP 정의

| 구성 | Phase 2에서의 의미 |
|---|---|
| State | 남은 W/O, 설비별 누적 부하, 설비 종료 시각, 열린 batch |
| Action | `SELECT_MACHINE` 또는 `SELECT_WO` |
| Transition | batch를 열고 W/O를 추가한 후 설비에 투입 |
| Objective | 제약을 지키며 makespan과 설비별 부하 편차 최소화 |
| Termination | 모든 Bay의 W/O가 설비와 batch에 배정됨 |

## 3. Action 구조

Phase 2는 W/O 조합을 한 번에 선택하지 않고 두 단계로 구성한다.

```text
SELECT_MACHINE
→ SELECT_WO
→ SELECT_WO
→ SELECT_WO
→ batch 자동 close
```

### SELECT_MACHINE

다음 batch를 처리할 설비 하나를 선택한다.

- 남은 W/O를 처리할 수 있는 설비만 후보가 된다.
- 현재 종료 예정 시각이 가장 빠른 설비만 후보가 된다.
- 사용 중지 설비와 해당 계열을 처리할 수 없는 설비는 제외한다.
- 에이전트는 설비만 선택하며 batch 크기는 환경이 계산한다.

```text
목표 batch 크기
= min(3, ceil(남은 W/O 수 / 현재 가장 빨리 비는 설비 수))
```

예를 들어 W/O 8개와 설비 3대가 있으면 batch 크기는 대체로 `3, 3, 2`가 된다. W/O 4개와 설비 3대면 `2, 1, 1`이 된다.

### SELECT_WO

선택한 설비의 열린 batch에 넣을 W/O를 **한 개씩** 선택한다.

```text
a_t = (현재 설비의 open batch에 추가할 W/O)
```

W/O를 추가할 때마다 open batch의 W/O 수, LTH 합, CUT_LTH 합, BV_QTY 합, 최대 TACT_TIME이 갱신된다.

## 4. Batch 제약

| 제약 | 기준 |
|---|---:|
| Batch W/O 수 | 최대 3개 |
| Batch LTH 합 | 최대 55,000 |
| Batch 처리시간 | Batch 내 W/O의 `max(TACT_TIME)` |
| Bay | Phase 1에서 배정된 Bay와 일치 |
| 계열 | 선택 설비가 해당 계열을 처리할 수 있어야 함 |

제약을 위반하는 W/O는 action 후보에서 제거한다. 목표가 3개여도 LTH 제약으로 세 번째 W/O를 넣을 수 없으면 2개로 batch를 자동 종료한다.

## 5. Action 예시

```text
A: TACT=30, LTH=12,000, CUT=500, BV=3
B: TACT=20, LTH=15,000, CUT=400, BV=1
C: TACT=40, LTH=18,000, CUT=900, BV=5
```

```text
SELECT_MACHINE: PLS31, 목표 크기 3
SELECT_WO: A  → batch={A},     LTH=12,000, 시간=30
SELECT_WO: B  → batch={A,B},   LTH=27,000, 시간=30
SELECT_WO: C  → batch={A,B,C}, LTH=45,000, 시간=40
AUTO CLOSE    → PLS31 시작=0, 종료=40
```

Batch가 닫힐 때 설비의 종료 시각과 누적 W/O·CUT·BV·점유시간을 확정한다. 다음 step은 갱신된 설비 상태를 사용한다.

## 6. State Feature

| 구분 | 차원 | 내용 |
|---|---:|---|
| Bay context | 13 | 진행률, 남은 물량, 현재 설비 부하 gap |
| Machine node | 설비당 11 | 종료 시각, W/O·CUT·BV·batch 부하, 계열 eligibility |
| W/O node | W/O당 9 | TACT_TIME, LTH, CUT_LTH, BV_QTY, 계열 |
| Open batch | 7 | W/O 수, LTH 합, 최대 TACT, CUT·BV 합, 남은 용량 |
| Projected action | 후보당 7 | 예상 종료 시각, makespan, W/O·CUT·BV·점유 gap |

현재 정책은 가변 길이 Machine/W/O set을 인코딩하는 **MLP 기반 Set-Pointer Policy**다. GNN message passing은 사용하지 않는다.

## 7. 사전식 목적함수

```text
1. Hard constraint 위반 수
2. Makespan
3. Bay 내부 설비별 CUT_LTH gap 합
4. Bay 내부 설비별 W/O 수 gap 합
5. Bay 내부 설비별 BV_QTY gap 합
6. Bay 내부 설비별 점유시간 gap 합
```

`raw`는 `max-min`, `normalized`는 `(max-min)/평균`으로 gap을 계산한다. 각 Bay subproblem에서 가장 좋은 후보를 따로 선택한다.

## 8. Self-labeling 학습

```text
Phase 1 배정 결과 생성
→ Bay별 subproblem 생성
→ Agent greedy/sampling + 휴리스틱 후보 생성
→ Bay별 최적 sequence를 pseudo-label로 선택
→ Cross Entropy update
```

한 Bay의 teacher sequence에는 설비 선택과 W/O 선택이 모두 포함된다. 동일한 정책을 Bay별로 update하므로 하나의 episode에서 비어 있지 않은 Bay 수만큼 학습 update가 발생할 수 있다.

## 9. 순서의 의미

Batch 내부에서 `A-B-C`와 `C-B-A`는 현재 모델에서 같은 batch다. 처리시간이 `max(TACT_TIME)`이고 순서 의존 setup time이 없기 때문이다.

현재 의미가 있는 순서는 **어떤 W/O끼리 batch를 구성하고, 어느 설비의 몇 번째 batch로 투입하는가**다.
