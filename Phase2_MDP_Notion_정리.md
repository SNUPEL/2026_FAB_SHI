# Phase 2 MDP: W/O Batch 구성 및 설비 배정

> Phase 2는 Phase 1에서 Bay가 확정된 W/O를 해당 Bay 내부 설비에 배정하고, W/O 1~3개를 하나의 batch로 구성하는 단계이다.

## 1. 입력과 출력

| 구분 | 내용 |
|---|---|
| 입력 | Phase 1의 `block-series -> Bay` 배정과 개별 W/O |
| 의사결정 단위 | 환경이 정한 설비의 open batch에 추가할 W/O |
| 출력 | W/O별 설비, batch, 시작·종료 시간 |
| 처리 단위 | W/O 1~3개로 구성된 batch |

하나의 episode를 Bay별 subproblem으로 나누어 풀지만, 모든 Bay는 동일한 Phase 2 정책 하나를 공유한다.

## 2. MDP 정의

| 구성 | Phase 2에서의 의미 |
|---|---|
| State | 남은 W/O, 설비별 누적 부하, 설비 종료 시각, 열린 batch |
| Action | 환경이 확정한 설비의 open batch에 추가할 `W/O` 하나 선택 |
| Transition | 환경이 설비와 batch 크기를 정하고, 선택 W/O를 추가한 후 batch를 투입 |
| Objective | 제약을 지키며 makespan과 설비별 부하 편차 최소화 |
| Termination | 모든 Bay의 W/O가 설비와 batch에 배정됨 |

## 3. Action 구조

Phase 2의 학습 action은 W/O 선택 하나다. 설비 선택은 policy가 아니라 환경의
결정론적 dispatch 규칙이다.

```text
환경: 현재 시각의 유휴 설비 + 목표 batch 크기 결정, open batch 생성
정책: SELECT_WO
정책: SELECT_WO
정책: SELECT_WO
환경: batch 자동 close 및 해당 machine 완료시각 예약
환경: 유휴 설비가 없을 때만 다음 완료 이벤트로 전역 clock 점프
```

### 환경의 설비 결정

남은 W/O가 있는 모든 Bay의 설비를 대상으로 다음 규칙으로 하나를 확정한다.

- 남은 W/O를 처리할 수 있는 설비만 후보가 된다.
- 현재 전역 `current_time`에 유휴인 설비를 먼저 고른다.
- 현재 유휴 설비가 여러 대면 `machine_id` 오름차순으로 고른다.
- 실행 가능한 유휴 설비가 없으면 전체 설비의 가장 빠른 완료 시각으로 `current_time`을 점프하고 다시 고른다.
- 사용 중지 설비와 해당 계열을 처리할 수 없는 설비는 제외한다.
- 이 결정은 학습 logit, sampling, CE target에 포함되지 않는다.
- batch close는 `machine_available_at`만 미래 종료시각으로 갱신하며 `current_time`은 이동하지 않는다.
- 모든 W/O 배정 후 남은 완료 이벤트를 drain하여 최종 `current_time=makespan`으로 종료한다.

```text
목표 batch 크기
= min(3, ceil(선택 Bay의 남은 W/O 수 / 현재 유휴·실행 가능 설비 수))
```

예를 들어 W/O 8개와 설비 3대가 있으면 batch 크기는 대체로 `3, 3, 2`가 된다. W/O 4개와 설비 3대면 `2, 1, 1`이 된다.

```text
t=0  PLS21: Batch A [0, 30]
t=0  PLS22: Batch B [0, 20]
t=0  PLS23: Batch C [0, 40]

모든 설비가 점유됨
-> 다음 최소 완료 이벤트 t=20으로 current_time 점프
-> PLS22가 유휴 상태가 되어 다음 batch 배정
```

### 학습 action: SELECT_WO

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
ENV DISPATCH  : PLS31, 목표 크기 3
SELECT_WO: A  → batch={A},     LTH=12,000, 시간=30
SELECT_WO: B  → batch={A,B},   LTH=27,000, 시간=30
SELECT_WO: C  → batch={A,B,C}, LTH=45,000, 시간=40
AUTO CLOSE    → PLS31 시작=0, 종료=40
```

Batch가 닫힐 때 설비의 종료 시각과 누적 W/O·CUT·BV·점유시간을 확정한다. 다음 step은 갱신된 설비 상태를 사용한다.

## 6. State Feature

| 구분 | 차원 | 내용 |
|---|---:|---|
| Bay context | 11 | 진행률, 남은 물량, 현재 설비 부하 gap |
| Machine node | 설비당 11 | 종료 시각, W/O·CUT·BV·batch 부하, 계열 eligibility |
| W/O node | W/O당 9 | TACT_TIME, LTH, CUT_LTH, BV_QTY, 계열 |
| Open batch | 7 | W/O 수, LTH 합, 최대 TACT, CUT·BV 합, 남은 용량 |
| Projected action | 후보당 7 | 예상 종료 시각, makespan, W/O·CUT·BV·점유 gap |

현재 정책은 가변 길이 Machine/W/O set을 인코딩하는 **MLP 기반 Set-Pointer Policy**다. GNN message passing은 사용하지 않는다.

### 6.1 입력과 Encoder

은닉 차원을 `H=128`, 현재 W/O 후보 수를 `K`라고 하면 실제 입력과 encoder는 다음과 같다.

| 입력 | 원시 차원 | Feedforward encoder | 출력 |
|---|---:|---|---:|
| Bay context | 11 | `Linear(11,128) -> ReLU -> Linear(128,128)` | `h_bay ∈ R^128` |
| Open batch | 7 | `Linear(7,128) -> ReLU -> Linear(128,128)` | `h_open ∈ R^128` |
| Machine set | `M×11` | 각 Machine에 동일한 `Linear(11,128) -> ReLU -> Linear(128,128)` | `H_M ∈ R^(M×128)` |
| W/O set | `N×9` | 각 W/O에 동일한 `Linear(9,128) -> ReLU -> Linear(128,128)` | `H_W ∈ R^(N×128)` |
| Projected action | `K×7` | 각 action에 동일한 `Linear(7,128) -> ReLU -> Linear(128,128)` | `P ∈ R^(K×128)` |

모든 Feedforward encoder는 마지막 Linear 뒤에 activation을 추가하지 않는다.

```text
FFN_d(x) = W_2 · ReLU(W_1 x + b_1) + b_2
```

Machine과 W/O의 개수가 바뀌어도 같은 row-wise encoder를 반복 적용하므로 parameter 수는 변하지 않는다.

### 6.2 전체 Context Fusion

Machine set과 W/O set은 순서에 영향을 받지 않도록 각각 mean pooling과 max pooling을 수행한다.

```text
r = [
    h_bay,
    h_open,
    mean(H_M), max(H_M),
    mean(H_W), max(H_W)
] ∈ R^(6×128) = R^768

c = FFN_768(r) ∈ R^128
```

Context encoder도 `Linear(768,128) -> ReLU -> Linear(128,128)`이다. `c`는 현재 Bay의 진행률, 남은 W/O, 전체 Machine 부하, 열린 batch 상태를 합친 전역 context다.

### 6.3 W/O Candidate와 Query

```text
candidate h_i = 환경이 정한 Machine에 투입 가능한 W/O embedding
selected-machine embedding m = 환경이 정한 Machine embedding
```

하나의 query encoder를 사용한다.

```text
q = FFN_256([c ; m]) ∈ R^128
FFN_256 = Linear(256,128) -> ReLU -> Linear(128,128)
```

설비는 환경이 결정하지만 해당 Machine embedding은 query에 들어간다. 따라서
Machine encoder도 CE 역전파로 학습되며, 같은 W/O라도 어느 Machine의 batch를
구성하는지에 따라 점수가 달라진다.

### 6.4 Projected Action Feature와 Pointer Scoring

각 후보를 실제 선택했을 때의 예상 결과 7개를 먼저 계산한다.

```text
duration increment
projected finish time
projected makespan
projected normalized W/O gap
projected normalized CUT_LTH gap
projected normalized BV_QTY gap
projected normalized occupancy gap
```

이를 `p_i = ProjectedEncoder(projected_i) ∈ R^128`로 인코딩한 후 다음 compatibility score를 계산한다.

```text
u_i = tanh(h_i + p_i + q)
z_i = w^T u_i
π_θ(a=i | s, machine) = softmax(z_i + M_i)

M_i = 0    if action i is feasible
M_i = -∞   otherwise
```

따라서 최종 점수에는 후보 자체의 node embedding `h_i`, 선택 후 예상 변화 `p_i`, 현재 전체 상태와 선택 Machine을 담은 query `q`가 모두 직접 들어간다.

### 6.5 한 Policy가 만드는 최종 Sequence

설비 결정은 환경 event이고 학습 token이 아니다. 다음 예에서 CE target은 W/O
7개에 대해서만 생성된다.

| Step | 주체 | 선택/결정 | 환경 변화 |
|---:|---|---|---|
| 0 | 환경 | `PLS21`, target=3 | PLS21 open batch 생성 |
| 1 | Policy | `WO1` | PLS21 batch에 WO1 추가 |
| 2 | Policy | `WO8` | PLS21 batch에 WO8 추가 |
| 3 | Policy | `WO10` | 3개가 되어 batch 자동 close |
| 3 | 환경 | `PLS22`, target=3 | PLS22 open batch 생성 |
| 4 | Policy | `WO3` | PLS22 batch에 WO3 추가 |
| 5 | Policy | `WO2` | PLS22 batch에 WO2 추가 |
| 6 | Policy | `WO12` | 3개가 되어 batch 자동 close |
| 6 | 환경 | `PLS22`, target=1 | PLS22의 다음 batch 생성 |
| 7 | Policy | `WO5` | 남은 W/O가 없어 batch close |

정책이 생성한 token sequence와 최종 해 표현은 다음과 같다.

```text
learned W/O token sequence
[WO1, WO8, WO10, WO3, WO2, WO12, WO5]

decoded schedule
π = {
    PLS21: ({WO1, WO8, WO10}),
    PLS22: ({WO3, WO2, WO12}, {WO5})
}
```

### 6.6 Self-labeling과 Cross Entropy

각 Bay subproblem에서 휴리스틱, agent greedy, agent sampling이 완성된 schedule 후보를 만든다. 사전식 score가 가장 좋은 schedule의 전체 micro-action sequence를 teacher로 선택한다.

```text
L(θ) = -(1/T) Σ_t log π_θ(WO_t* | s_t, machine_t)
```

구현에서는 각 transition의 logits에 `CrossEntropyLoss`를 적용하고 transition 수로 평균낸 뒤 한 번 `backward()`와 `optimizer.step()`을 수행한다. Bay가 바뀌어도 별도 모델을 만들지 않으며, **동일한 한 정책의 parameter를 Bay별 subproblem update가 순차적으로 갱신한다.**

정확한 명칭은 다음과 같다.

> 단일 agent, 환경의 deterministic machine dispatch, autoregressive W/O Set-Pointer Policy

`Environment/gym_wrapper.py`에 남아 있는 2-stage action adapter는 기존 DES
baseline/debug 검사용이다. merged Phase 2 공개 학습 CLI는 이 adapter를 호출하지
않으며, `SELECT_MACHINE`을 학습 transition이나 CE target으로 만들지 않는다.

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

Bay `b`의 부하 지표 `x`에 대한 gap은 다음과 같다.

```text
gap(b, x) = max(load(m, x) for m in machines(b))
          - min(load(m, x) for m in machines(b))

combined_gap(x) = sum(gap(b, x) for b in active_bays)
```

Bay별 self-labeling update에서는 해당 Bay의 `gap(b, x)`를 비교한다. 여러 Bay의
완성 계획을 합친 리포트와 full-flow score에서는 `combined_gap(x)`를 사용한다.
`makespan`만 모든 active Bay 설비의 최종 완료시각 최댓값으로 계산한다.

## 8. Self-labeling 학습

```text
Phase 1 배정 결과 생성
→ Bay별 subproblem 생성
→ Agent greedy/sampling + 휴리스틱 후보 생성
→ Bay별 최적 sequence를 pseudo-label로 선택
→ Cross Entropy update
```

한 Bay의 teacher sequence에는 W/O 선택만 포함된다. 설비와 목표 batch 크기는 모든
agent/휴리스틱 후보에 동일한 환경 규칙으로 주어진다. 동일한 정책을 Bay별로
update하므로 하나의 episode에서 비어 있지 않은 Bay 수만큼 학습 update가 발생할 수 있다.

## 9. 순서의 의미

Batch 내부에서 `A-B-C`와 `C-B-A`는 현재 모델에서 같은 batch다. 처리시간이 `max(TACT_TIME)`이고 순서 의존 setup time이 없기 때문이다.

현재 의미가 있는 순서는 **어떤 W/O끼리 batch를 구성하고, 어느 설비의 몇 번째 batch로 투입하는가**다.
