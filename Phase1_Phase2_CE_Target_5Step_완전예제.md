# Phase 1/2 CE Target와 Pointer Index: 5-Step 완전 예제

이 문서는 Phase 1과 Phase 2의 self-labeling 학습에서 다음 질문을 초보자도
직접 따라갈 수 있도록 설명한다.

1. Agent와 휴리스틱이 실제로 무엇을 선택하는가?
2. 실제 action이 왜 정수 index로 바뀌는가?
3. `index trace`와 최종 해는 무엇이 다른가?
4. 휴리스틱이 가장 좋을 때 Cross Entropy(CE)는 무엇을 정답으로 사용하는가?
5. Phase 2에서 Machine은 왜 CE target에 나타나지 않는가?

---

## 1. 가장 중요한 결론

Phase 1과 Phase 2 모두 신경망이 직접 학습하는 정답은 **현재 step의 후보
목록에서 선택된 행의 0-based local index**다.

```text
Phase 1 의미 action = (block-series, Bay)
Phase 1 CE target    = 현재 feasible pair 목록에서 위 pair가 있는 local index

Phase 2 의미 action = W/O
Phase 2 CE target    = 현재 feasible W/O 목록에서 위 W/O가 있는 local index
```

따라서 다음 두 문장은 모두 필요하다.

```text
학습 target은 index다.
최종 해는 index가 가리키는 실제 pair 또는 W/O ID다.
```

`index trace`만 저장하면 최종 해를 복원할 수 없다. 같은 숫자 `1`도 step마다
서로 다른 후보를 가리킬 수 있기 때문이다.

---

## 2. 다섯 가지 용어를 먼저 구분한다

| 용어 | 의미 | 예 |
|---|---|---|
| 의미 action | 환경에 실제 적용되는 결정 | `(P1::NP::B1, 22)` 또는 `WO5` |
| 후보 목록 | 현재 step에서 선택 가능한 action의 순서 있는 목록 | `[WO1, WO2, WO5]` |
| local pointer index | 후보 목록에서 선택 action이 있는 위치 | `WO5 -> index 2` |
| pointer trace | 한 complete solution의 step별 local index 모음 | `[4, 3, 1, 1, 0]` |
| decoded solution | index를 실제 ID로 복원한 최종 해 | `[WO5, WO4, WO2, WO3, WO1]` |

코드의 index는 항상 `0`부터 시작한다.

```text
후보 첫 번째  -> index 0
후보 두 번째  -> index 1
후보 세 번째  -> index 2
```

발표자료에서 편의상 1부터 셀 수는 있지만, 코드 target과 혼동하지 않아야 한다.

---

## 3. Pointer Policy와 CE의 공통 원리

### 3.1 현재 후보 목록

step `t`의 후보 목록을 다음과 같이 쓴다.

\[
C_t=(c_{t,0},c_{t,1},\ldots,c_{t,K_t-1})
\]

- `K_t`: 현재 feasible 후보 수
- `c_{t,i}`: 후보 index `i`가 가리키는 실제 action
- hard constraint를 위반한 action은 `C_t`에 들어오지 않는다.

### 3.2 신경망 출력

신경망은 후보마다 logit 하나를 출력한다.

\[
z_t=(z_{t,0},z_{t,1},\ldots,z_{t,K_t-1})
\]

softmax를 적용하면 후보별 선택 확률이 된다.

\[
\pi_\theta(i\mid s_t,C_t)
=\frac{\exp(z_{t,i})}{\sum_{j=0}^{K_t-1}\exp(z_{t,j})}
\]

Agent greedy는 가장 큰 logit의 index를 선택한다.

```text
selected_index = argmax(logits)
```

Agent sampling은 softmax 확률분포에서 index 하나를 표본화한다.

```text
selected_index ~ Categorical(softmax(logits))
```

### 3.3 CE target

teacher가 선택한 의미 action이 후보 `C_t`의 `i_t*` 위치에 있다면 CE target은
정수 `i_t*`다.

\[
L_t=-\log \pi_\theta(i_t^*\mid s_t,C_t)
\]

complete solution의 transition이 `T`개라면 현재 구현은 평균 CE를 사용한다.

\[
L=\frac{1}{T}\sum_{t=0}^{T-1}L_t
\]

즉 pointer trace 전체를 하나의 클래스처럼 학습하는 것이 아니다.

```text
Step 0에서는 target index의 확률을 높인다.
Step 1에서는 새 state와 새 후보 목록에서 target index의 확률을 높인다.
...
Step T-1에서도 같은 계산을 한다.
평균 loss로 한 번 backward한다.
```

### 3.4 왜 index 숫자만으로는 부족한가?

다음 두 step을 비교한다.

```text
Step 0 후보: [WO1, WO2, WO3]
target index 1 -> WO2

Step 1 후보: [WO1, WO3]
target index 1 -> WO3
```

두 target 모두 숫자는 `1`이지만 실제 W/O는 다르다. 따라서 완전한 학습
transition은 최소한 다음 정보를 포함해야 한다.

```text
(state_t, candidate_ids_t, selected_action_index_t, selected_action_id_t)
```

---

## 4. Self-labeling에서 휴리스틱도 index target이 되는 과정

Agent와 휴리스틱은 먼저 각각 **완전한 해**를 생성한다.

```text
Agent greedy complete solution
Agent sample complete solution들
휴리스틱 complete solution들
```

그다음 모든 complete solution을 같은 사전식 목적함수로 평가한다.

```text
teacher = lexicographic score가 가장 작은 complete solution
```

teacher가 Agent이면 Agent rollout에 이미 저장된 local index를 사용한다.

teacher가 휴리스틱이면 휴리스틱의 실제 action sequence를 환경에서 다시
재생하면서 매 step의 후보 목록과 local index를 만든다.

```text
휴리스틱 의미 action 선택
-> 현재 hard mask가 적용된 후보 목록 생성
-> 의미 action의 위치를 찾음
-> selected_action_index로 저장
-> 다음 state로 전이
```

휴리스틱이 hard mask를 위반한 action을 선택하면 index를 찾을 수 없다. 이 경우
다른 action으로 조용히 대체하지 않고 오류를 발생시킨다.

중요한 점은 다음과 같다.

> Agent가 LPT, SPT 또는 Phase 1 휴리스틱의 이름을 학습하는 것이 아니다.
> 해당 휴리스틱이 만든 complete solution의 step별 local index를 CE target으로
> 학습한다.

---

# Part A. Phase 1

## 5. Phase 1의 의미 action

Phase 1의 action은 다음 pair다.

\[
a_t=(\text{block-series},\text{Bay})
\]

`block-series`는 다음 identity를 사용한다.

```text
PROJ_NO::GYEL::BLK_NO
```

예를 들어 같은 물리 블록에 NP와 NC가 있으면 서로 다른 의사결정 단위다.

```text
P1::NP::B1
P1::NC::B1
```

한 block-series에 속한 모든 W/O는 pair action 한 번으로 같은 Bay를 상속한다.

```text
(P1::NP::B1, Bay22) 선택
-> P1::NP::B1에 속한 모든 NP W/O가 Bay22로 감
```

신경망 입력에서 `P1::NP::B1`을 정수 `1`로, Bay22를 정수 `22`로 넣는 것이
아니다. 각 feasible pair가 20차원 feature row 하나가 되고, 전체 환경은
8차원 vector가 된다.

```text
pair feature matrix: (현재 feasible pair 수, 20)
environment vector : (8,)
policy output       : (현재 feasible pair 수,)
```

실제 문자열 ID는 최종 해 복원과 audit를 위해 함께 보관한다.

---

## 6. Phase 1 hard constraint

### 6.1 계열별 기본 Bay

| 계열 | 기본 후보 Bay |
|---|---|
| NP | 22, 23, 24 |
| NC | 22, 23, 24 |
| FN | 25, trans |
| FL | 25, trans |

### 6.2 NP 추가 hard mask

| 조건 | 판정 | 허용 Bay |
|---|---|---|
| 장척 | 동일 block-series W/O의 `CUT_LTH` 합 `>= 1000` | 22, 23 |
| 광폭 | block-series의 `BTH` 최댓값 `> 4500` | 22, 23 |
| CNT | `BLK_NO`가 `CNT_BLK`로 시작 | 22, 23 |

위 조건은 penalty가 아니다. 위반 pair 자체가 후보 목록에서 제거된다.

### 6.3 Bay 설비 수

| Bay | 설비 수 |
|---|---:|
| 22 | 4 |
| 23 | 3 |
| 24 | 4 |
| 25 | 2 |
| trans | 2 |

Phase 1 gap은 Bay raw 합계가 아니라 `Bay 부하 / 설비 수`를 사용한다.

---

## 7. Phase 1 5-Step 입력 문제

다음은 `NP_NC` 자원군의 block-series 5개로 만든 설명용 문제다. 아래 값은
실제 코드 경로로 후보 생성과 score를 재생할 수 있도록 구성했다.

| 약칭 | 실제 block-series ID | 계열 | W/O 수 | CUT_LTH 합 | BV_QTY 합 | BTH 최대 | 추가 조건 |
|---|---|---|---:|---:|---:|---:|---|
| G1 | `P1::NP::B1` | NP | 3 | 1,200 | 7 | 4,200 | 장척 |
| G2 | `P2::NC::B2` | NC | 2 | 600 | 2 | 4,000 | 없음 |
| G3 | `P3::NP::CNT_BLK_3` | NP | 1 | 500 | 1 | 4,000 | CNT |
| G4 | `P4::NP::B4` | NP | 2 | 700 | 4 | 4,700 | 광폭 |
| G5 | `P5::NC::B5` | NC | 3 | 900 | 5 | 4,000 | 없음 |

hard mask 결과는 다음과 같다.

| 그룹 | 후보 Bay | 이유 |
|---|---|---|
| G1 | 22, 23 | NP 장척 `CUT_LTH 합=1200` |
| G2 | 22, 23, 24 | NC 기본 eligibility |
| G3 | 22, 23 | NP CNT |
| G4 | 22, 23 | NP 광폭 `BTH=4700` |
| G5 | 22, 23, 24 | NC 기본 eligibility |

이 예제의 설명용 complete pair sequence는 다음과 같다.

```text
[(G1,22), (G5,24), (G3,23), (G4,22), (G2,23)]
```

이 sequence가 Agent, W/O-first, CUT-first, Bevel-first 후보 중 최종 사전식
score가 가장 좋게 평가됐다고 가정하면 아래 local index들이 CE target이 된다.

---

## 8. Phase 1 Step 0

### 8.1 현재 후보

후보는 `block-series ID -> Bay ID` 순서로 결정론적으로 정렬된다.

| local index | 후보 action |
|---:|---|
| 0 | `G1@22` |
| 1 | `G1@23` |
| 2 | `G2@22` |
| 3 | `G2@23` |
| 4 | `G2@24` |
| 5 | `G3@22` |
| 6 | `G3@23` |
| 7 | `G4@22` |
| 8 | `G4@23` |
| 9 | `G5@22` |
| 10 | `G5@23` |
| 11 | `G5@24` |

`G1@24`, `G3@24`, `G4@24`는 hard mask 때문에 처음부터 없다.

### 8.2 선택과 target

```text
teacher 의미 action = G1@22
selected_action_index = 0
```

### 8.3 환경 전이

```text
G1의 W/O 3개, CUT 1200, BV 7을 Bay22에 누적
G1의 모든 pair 후보 G1@22, G1@23 제거
남은 block-series = G2, G3, G4, G5
```

---

## 9. Phase 1 Step 1

### 9.1 새 후보

후보 목록은 다시 0부터 index를 부여한다.

| local index | 후보 action |
|---:|---|
| 0 | `G2@22` |
| 1 | `G2@23` |
| 2 | `G2@24` |
| 3 | `G3@22` |
| 4 | `G3@23` |
| 5 | `G4@22` |
| 6 | `G4@23` |
| 7 | `G5@22` |
| 8 | `G5@23` |
| 9 | `G5@24` |

### 9.2 선택과 target

```text
teacher 의미 action = G5@24
selected_action_index = 9
```

### 9.3 환경 전이

```text
G5의 W/O 3개, CUT 900, BV 5를 Bay24에 누적
G5@22, G5@23, G5@24 제거
남은 block-series = G2, G3, G4
```

---

## 10. Phase 1 Step 2

| local index | 후보 action |
|---:|---|
| 0 | `G2@22` |
| 1 | `G2@23` |
| 2 | `G2@24` |
| 3 | `G3@22` |
| 4 | `G3@23` |
| 5 | `G4@22` |
| 6 | `G4@23` |

```text
teacher 의미 action = G3@23
selected_action_index = 4
```

환경 변화:

```text
G3의 W/O 1개, CUT 500, BV 1을 Bay23에 누적
G3@22, G3@23 제거
남은 block-series = G2, G4
```

---

## 11. Phase 1 Step 3

| local index | 후보 action |
|---:|---|
| 0 | `G2@22` |
| 1 | `G2@23` |
| 2 | `G2@24` |
| 3 | `G4@22` |
| 4 | `G4@23` |

```text
teacher 의미 action = G4@22
selected_action_index = 3
```

환경 변화:

```text
G4의 W/O 2개, CUT 700, BV 4를 Bay22에 누적
G4@22, G4@23 제거
남은 block-series = G2
```

---

## 12. Phase 1 Step 4

| local index | 후보 action |
|---:|---|
| 0 | `G2@22` |
| 1 | `G2@23` |
| 2 | `G2@24` |

```text
teacher 의미 action = G2@23
selected_action_index = 1
```

환경 변화:

```text
G2의 W/O 2개, CUT 600, BV 2를 Bay23에 누적
남은 block-series 없음
NP_NC subproblem 종료
```

---

## 13. Phase 1 pointer trace와 실제 해

### 13.1 CE target

각 step의 selected local index만 모으면 다음과 같다.

```text
Phase 1 pointer trace = [0, 9, 4, 3, 1]
```

각 step의 logit 개수와 target은 다음과 같다.

| Step | logit 수 | CE target |
|---:|---:|---:|
| 0 | 12 | 0 |
| 1 | 10 | 9 |
| 2 | 7 | 4 |
| 3 | 5 | 3 |
| 4 | 3 | 1 |

### 13.2 의미 action sequence

```text
[(G1,22), (G5,24), (G3,23), (G4,22), (G2,23)]
```

### 13.3 최종 assignment

```text
G1 -> Bay22
G2 -> Bay23
G3 -> Bay23
G4 -> Bay22
G5 -> Bay24
```

최종 해는 assignment다. pointer trace는 이 assignment를 만들기 위한 학습용
construction trace다.

---

## 14. Phase 1 예제의 최종 score

설비 수는 Bay22=4, Bay23=3, Bay24=4다.

### 14.1 NP 계열의 설비당 부하

| Bay | NP W/O/설비 | NP CUT/설비 | NP BV/설비 |
|---|---:|---:|---:|
| 22 | `5/4=1.25` | `1900/4=475` | `11/4=2.75` |
| 23 | `1/3=0.333` | `500/3=166.667` | `1/3=0.333` |
| 24 | 0 | 0 | 0 |

```text
NP W/O gap = 1.25
NP CUT gap = 475
NP BV gap  = 2.75
```

### 14.2 NC 계열의 설비당 부하

| Bay | NC W/O/설비 | NC CUT/설비 | NC BV/설비 |
|---|---:|---:|---:|
| 22 | 0 | 0 | 0 |
| 23 | `2/3=0.667` | `600/3=200` | `2/3=0.667` |
| 24 | `3/4=0.75` | `900/4=225` | `5/4=1.25` |

```text
NC W/O gap = 0.75
NC CUT gap = 225
NC BV gap  = 1.25
```

`series_only` score는 계열별 gap 합을 사전식으로 사용한다.

```text
series W/O gap = 1.25 + 0.75 = 2.0
series CUT gap = 475 + 225 = 700
series BV gap  = 2.75 + 1.25 = 4.0

score = (2.0, 700.0, 4.0)
```

`shared_and_series`라면 NP+NC 공유 자원군 전체 gap도 앞에 교대로 들어간다.

```text
score = (
  shared W/O gap=0.5,
  series W/O gap=2.0,
  shared CUT gap=250.0,
  series CUT gap=700.0,
  shared BV gap=1.75,
  series BV gap=4.0
)
```

teacher 선정은 위 complete score를 다른 Agent/휴리스틱 complete solution의
score와 비교해 수행한다.

---

## 15. Phase 1 휴리스틱과 CE

현재 Phase 1 휴리스틱은 LPT/SPT가 아니다.

| 휴리스틱 | block-series 순서 |
|---|---|
| `wo_first_balanced` | W/O 수 내림차순, CUT 내림차순, BV 내림차순 |
| `cut_first_balanced` | CUT 내림차순, W/O 수 내림차순, BV 내림차순 |
| `bevel_first_balanced` | BV 내림차순, W/O 수 내림차순, CUT 내림차순 |

각 block-series의 Bay는 hard mask를 통과한 Bay 중 **해당 block을 가상
배정한 뒤의 Phase 1 사전식 score가 가장 작은 Bay**로 결정한다.

휴리스틱의 complete assignment가 teacher가 되면 해당 assignment를 처음부터
재생해 `candidate_ids`, `selected_action_index`를 만든다. 위 예제에서는 최종
CE target이 `[0, 9, 4, 3, 1]`이다.

Phase 1은 `NP_NC`와 `FN_FL`을 서로 다른 subproblem으로 학습한다.

```text
NP_NC teacher 선택 -> NP_NC transition 평균 CE -> optimizer update
FN_FL teacher 선택 -> FN_FL transition 평균 CE -> optimizer update
```

한 parent episode에 두 자원군이 모두 있으면 같은 policy parameter에 update가
두 번 발생한다.

---

# Part B. Phase 2

## 16. Phase 2의 의미 action

현재 merged Phase 2 policy action은 `SELECT_WO` 하나다.

\[
a_t=\text{W/O}
\]

Machine은 Agent action이 아니다.

```text
환경: 현재 event time에서 실행 가능한 유휴 Machine 결정
Agent 또는 휴리스틱: 해당 Machine의 open batch에 넣을 W/O 선택
```

Machine 정보가 학습에서 사라진 것은 아니다.

```text
selected_machine_id          -> transition에 저장
selected Machine embedding   -> pointer query에 포함
Machine별 부하와 종료 시각   -> state에 포함
CE target                    -> W/O local index만 사용
```

Phase 2의 완전한 transition은 다음처럼 읽어야 한다.

```text
(state_t, selected_machine_t, feasible_wo_ids_t, selected_wo_index_t, selected_wo_id_t)
```

---

## 17. Phase 2 hard constraint와 환경 규칙

### 17.1 현재 활성 hard constraint

| 제약 | 의미 |
|---|---|
| `machine_enabled` | 중지된 설비 제외 |
| `machine_bay_consistency` | Phase 1 Bay와 Machine Bay 일치 |
| `block_set_same_bay` | 같은 block-series의 Phase 1 Bay 상속 |
| `machine_single_processing` | 사용 중인 설비에 동시 신규 batch 금지 |
| `batch_wo_count_limit` | batch W/O 최대 3개 |
| `batch_length_sum_limit` | batch LTH 합 최대 55,000 |
| `family_eligibility` | 설비가 해당 NP/FN/FL/NC 계열을 처리할 수 있어야 함 |

`thickness_range`와 `table_length_limit`은 현재 확정 Phase 2 profile에서
비활성이다. batch LTH 55,000 제약과 혼동하지 않는다.

### 17.2 Batch 처리시간

batch 안 W/O가 동시에 투입되고 동시에 완료되므로 batch 처리시간은 다음이다.

\[
p_{\text{batch}}=\max_{j\in\text{batch}}TACT\_TIME_j
\]

### 17.3 Machine 선택과 event clock

환경은 현재 시각에 실행 가능한 유휴 설비 중 `machine_id`가 가장 작은 설비를
선택한다. 실행 가능한 유휴 설비가 하나도 없을 때만 가장 빠른 완료 event로
전역 시간을 점프한다.

Machine 선택은 logit, sampling, CE target에 포함되지 않는다.

---

## 18. SPT와 LPT를 정확히 구분한다

### 18.1 정확한 정의

```text
SPT: feasible 후보 중 TACT_TIME이 가장 작은 W/O 우선
LPT: feasible 후보 중 TACT_TIME이 가장 큰 W/O 우선
```

다음 정렬은 내림차순이다.

```text
50, 25, 20, 10, 9
```

따라서 위 순서에서 `50`부터 선택하는 것은 SPT가 아니라 LPT다.

SPT의 TACT 기준 오름차순은 다음이다.

```text
9, 10, 20, 25, 50
```

### 18.2 코드의 후보 index 순서와 SPT 순서는 다르다

현재 코드의 W/O 후보 목록은 재현성을 위해 다음 순서로 정렬한다.

```text
TACT_TIME 내림차순
-> CUT_LTH 내림차순
-> W/O ID 오름차순
```

SPT 휴리스틱은 이 목록을 그대로 앞에서 고르는 것이 아니다. 목록 안의 모든
feasible action을 비교해 `TACT_TIME` 최솟값의 **현재 local index**를 찾는다.

따라서 SPT의 첫 target이 마지막 index가 될 수 있다.

---

## 19. Phase 2 SPT 5-Step 입력 문제

Bay25에 Phase 1 배정이 끝난 FN/FL W/O 5개가 있다고 가정한다.

| W/O | 계열 | TACT_TIME | LTH | CUT_LTH | BV_QTY | Phase 1 Bay |
|---|---|---:|---:|---:|---:|---|
| WO1 | FN | 50 | 20,000 | 500 | 5 | 25 |
| WO2 | FN | 25 | 25,000 | 300 | 2 | 25 |
| WO3 | FL | 20 | 35,000 | 450 | 4 | 25 |
| WO4 | FL | 10 | 15,000 | 100 | 1 | 25 |
| WO5 | FN | 9 | 10,000 | 80 | 0 | 25 |

Bay25 설비는 `PLS51`, `PLS52` 두 대이며 모두 FN/FL을 처리할 수 있다고
가정한다.

```text
현재 시각 = 0
남은 W/O = 5
실행 가능한 유휴 설비 = PLS51, PLS52
환경 선택 설비 = PLS51
목표 batch 크기 = min(3, ceil(5/2), feasible 수) = 3
```

---

## 20. Phase 2 Step 0

### 20.1 현재 상태

```text
selected Machine = PLS51
open batch = {}
open LTH = 0
```

### 20.2 후보 목록

후보 index는 코드의 결정론적 내림차순 정렬이다.

| local index | W/O | TACT | 선택 후 LTH | feasible |
|---:|---|---:|---:|---|
| 0 | WO1 | 50 | 20,000 | O |
| 1 | WO2 | 25 | 25,000 | O |
| 2 | WO3 | 20 | 35,000 | O |
| 3 | WO4 | 10 | 15,000 | O |
| 4 | WO5 | 9 | 10,000 | O |

### 20.3 SPT 선택과 target

```text
SPT 선택 W/O = WO5
selected_action_index = 4
```

### 20.4 환경 전이

```text
PLS51 open batch = {WO5}
W/O 수 = 1
LTH 합 = 10,000
CUT 합 = 80
BV 합 = 0
현재 batch 처리시간 = max(9) = 9
```

---

## 21. Phase 2 Step 1

| local index | W/O | TACT | 선택 후 LTH | feasible |
|---:|---|---:|---:|---|
| 0 | WO1 | 50 | 30,000 | O |
| 1 | WO2 | 25 | 35,000 | O |
| 2 | WO3 | 20 | 45,000 | O |
| 3 | WO4 | 10 | 25,000 | O |

```text
SPT 선택 W/O = WO4
selected_action_index = 3
```

환경 전이:

```text
PLS51 open batch = {WO5, WO4}
W/O 수 = 2
LTH 합 = 25,000
CUT 합 = 180
BV 합 = 1
현재 batch 처리시간 = max(9,10) = 10
```

---

## 22. Phase 2 Step 2: LTH hard mask가 실제 적용되는 순간

현재 open LTH는 `25,000`이다.

```text
WO1 추가 -> 25,000 + 20,000 = 45,000 -> feasible
WO2 추가 -> 25,000 + 25,000 = 50,000 -> feasible
WO3 추가 -> 25,000 + 35,000 = 60,000 -> hard violation
```

WO3는 점수가 나빠지는 것이 아니라 후보 목록에서 제거된다.

| local index | W/O | TACT | 선택 후 LTH |
|---:|---|---:|---:|
| 0 | WO1 | 50 | 45,000 |
| 1 | WO2 | 25 | 50,000 |

SPT는 현재 feasible 후보 중 TACT가 작은 WO2를 고른다.

```text
SPT 선택 W/O = WO2
selected_action_index = 1
```

환경 전이:

```text
PLS51 open batch = {WO5, WO4, WO2}
W/O 수 = 3
LTH 합 = 50,000
CUT 합 = 480
BV 합 = 3
batch 처리시간 = max(9,10,25) = 25
목표 크기 3 도달 -> batch 자동 close
PLS51 예약 시간 = [0,25]
```

---

## 23. Phase 2 Step 3

PLS51은 `t=25`까지 점유되어 있지만 PLS52는 `t=0`에 유휴다. 따라서 전역
시각은 이동하지 않고 환경이 PLS52를 선택한다.

```text
현재 시각 = 0
남은 W/O = WO1, WO3
selected Machine = PLS52
목표 batch 크기 = min(3, ceil(2/1), feasible 수) = 2
```

| local index | W/O | TACT | 선택 후 LTH | feasible |
|---:|---|---:|---:|---|
| 0 | WO1 | 50 | 20,000 | O |
| 1 | WO3 | 20 | 35,000 | O |

```text
SPT 선택 W/O = WO3
selected_action_index = 1
```

환경 전이:

```text
PLS52 open batch = {WO3}
LTH 합 = 35,000
CUT 합 = 450
BV 합 = 4
현재 batch 처리시간 = 20
```

---

## 24. Phase 2 Step 4

남은 W/O는 WO1뿐이다.

```text
현재 open LTH 35,000 + WO1 LTH 20,000 = 55,000
55,000 <= 55,000이므로 feasible
```

| local index | W/O | TACT | 선택 후 LTH |
|---:|---|---:|---:|
| 0 | WO1 | 50 | 55,000 |

```text
SPT 선택 W/O = WO1
selected_action_index = 0
```

환경 전이:

```text
PLS52 open batch = {WO3, WO1}
W/O 수 = 2
LTH 합 = 55,000
CUT 합 = 950
BV 합 = 9
batch 처리시간 = max(20,50) = 50
목표 크기 2 도달 -> batch 자동 close
PLS52 예약 시간 = [0,50]
```

모든 W/O가 배정됐으므로 남은 완료 event를 drain한다.

```text
최종 current_time = makespan = max(25,50) = 50
```

---

## 25. Phase 2 pointer trace와 실제 해

### 25.1 CE target

```text
SPT pointer trace = [4, 3, 1, 1, 0]
```

| Step | selected Machine | logit 수 | CE target | 실제 W/O |
|---:|---|---:|---:|---|
| 0 | PLS51 | 5 | 4 | WO5 |
| 1 | PLS51 | 4 | 3 | WO4 |
| 2 | PLS51 | 2 | 1 | WO2 |
| 3 | PLS52 | 2 | 1 | WO3 |
| 4 | PLS52 | 1 | 0 | WO1 |

Machine은 state/query context지만 CE target은 아니다.

### 25.2 W/O token sequence

```text
[WO5, WO4, WO2, WO3, WO1]
```

발표에서 `WO`를 생략하면 다음처럼 쓸 수 있다.

```text
[5, 4, 2, 3, 1]
```

이 숫자는 W/O ID를 줄여 쓴 표시일 뿐, CE target `[4,3,1,1,0]`과 다르다.

### 25.3 최종 설비별 batch sequence

```text
PLS51: Batch1={WO5, WO4, WO2}, [0,25]
PLS52: Batch1={WO3, WO1},      [0,50]
```

### 25.4 최종 Phase 2 score

```text
hard violation = 0
makespan = 50
PLS51 CUT=480, PLS52 CUT=950 -> CUT gap=470
PLS51 W/O=3,   PLS52 W/O=2   -> W/O gap=1
PLS51 BV=3,    PLS52 BV=9    -> BV gap=6
PLS51 점유=25, PLS52 점유=50 -> 점유시간 gap=25

score = (0, 50, 470, 1, 6, 25)
```

Phase 2 teacher는 Bay별로 다음 순서의 완성 해 score를 비교한다.

```text
1. hard violation
2. makespan
3. Bay 내부 Machine별 CUT_LTH gap
4. Bay 내부 Machine별 W/O 수 gap
5. Bay 내부 Machine별 BV_QTY gap
6. Bay 내부 Machine별 점유시간 gap
```

---

## 26. Phase 2 SPT가 teacher가 되었을 때 CE 계산

예를 들어 현재 모델이 SPT target에 다음 확률을 부여했다고 가정한다.

| Step | target index | target 확률 |
|---:|---:|---:|
| 0 | 4 | 0.10 |
| 1 | 3 | 0.20 |
| 2 | 1 | 0.35 |
| 3 | 1 | 0.40 |
| 4 | 0 | 1.00 |

평균 CE는 다음과 같다.

\[
L=-\frac{
\log(0.10)+\log(0.20)+\log(0.35)+\log(0.40)+\log(1.00)
}{5}
\approx 1.176
\]

backpropagation은 다음 parameter를 모두 갱신한다.

```text
Bay context encoder
Machine encoder
W/O encoder
Open-batch encoder
Projected-action encoder
Context encoder
Query encoder
Pointer scorer
```

Machine을 CE class로 선택하지 않더라도 selected Machine embedding이 query에
들어가므로 Machine encoder에도 gradient가 전달된다.

### 26.1 Phase 2 휴리스틱 동률 후보의 tie-aware CE

휴리스틱은 마지막 동률 해소를 위해 `job_id`를 사용할 수 있지만 정책 state에는
`job_id`가 없다. 따라서 휴리스틱 우선순위와 정책 관측 feature가 모두 같은 후보를
하나의 index로 강제하면, 신경망이 관측할 수 없는 문자열 차이를 외우라는 잘못된
target이 된다.

현재 Phase 2는 이런 후보들의 local index 집합을 모두 정답으로 인정한다.

```text
현재 후보 확률 = [0.10, 0.25, 0.05, 0.40, 0.20]
동률 정답 index 집합 T = {1, 3}
정답에 배분된 전체 확률 = 0.25 + 0.40 = 0.65
```

단일 target CE 대신 다음 loss를 사용한다.

\[
L_{\text{tie}}
=-\log\left(\sum_{i \in T} P(a_i \mid s)\right)
=-\log(0.65)
\]

즉 특정 동률 index 하나만 높이는 것이 아니라, **동률 정답 후보에 배분된 전체
확률 합을 높이도록 학습한다.** 동률 집합이 하나뿐이면 기존 단일 target CE와
정확히 같다.

복수 target은 다음 조건을 모두 만족하는 휴리스틱 teacher에만 적용한다.

```text
1. job_id 같은 identity tie-break를 제외한 휴리스틱 우선순위가 같다.
2. 정책이 실제로 보는 action feature가 같다.
```

우선순위가 같더라도 정책 feature가 다르면 구분 가능한 후보이므로 단일 target을
유지한다. Agent greedy/sample이 teacher인 경우에도 실제 agent가 선택한 index
하나만 target으로 사용한다.

---

## 27. Agent와 SPT/LPT가 함께 경쟁하는 정확한 형태

한 Bay subproblem에서 다음 complete solution들이 생성될 수 있다.

```text
Agent greedy
Agent sample 1
...
Agent sample K
SPT
LPT
기타 Phase 2 dispatching heuristic
```

각 candidate는 다음 세 표현을 모두 가진다.

```text
1. pointer trace
2. 실제 W/O token sequence
3. 최종 Machine/Batch schedule
```

예:

```text
Agent pointer trace = [1, 3, 0, 1, 0]
Agent W/O sequence  = [WO2, WO5, WO1, WO4, WO3]

SPT pointer trace   = [4, 3, 1, 1, 0]
SPT W/O sequence    = [WO5, WO4, WO2, WO3, WO1]
```

두 candidate의 최종 schedule을 Phase 2 사전식 score로 비교한다. SPT가 더
좋다면 SPT의 `[4,3,1,1,0]`이 해당 Bay update의 CE target이 된다.

다음 표현은 틀렸다.

```text
LPT라는 class를 학습한다.
WO5는 항상 class 4다.
Machine PLS51을 class로 함께 학습한다.
```

정확한 표현은 다음이다.

```text
현재 state와 현재 후보 목록에서 teacher가 선택한 local index의 확률을 높인다.
```

---

## 28. Phase 1과 Phase 2 비교

| 항목 | Phase 1 | Phase 2 |
|---|---|---|
| 의미 action | `(block-series, Bay)` | `W/O` |
| 후보 한 행 | feasible block-series/Bay pair | selected Machine에 추가 가능한 W/O |
| local index가 가리키는 것 | pair action | W/O action |
| Machine 선택 | 해당 없음 | 환경이 결정 |
| hard mask | 계열 Bay, NP 장척·광폭·CNT | Machine/Bay/계열, batch 3개, LTH 55,000 등 |
| complete solution | block-series -> Bay assignment | Machine별 batch sequence |
| teacher 단위 | 자원군 `NP_NC`, `FN_FL`별 | Bay별 |
| CE update | 자원군별 transition 평균 | Bay별 transition 평균 |

---

## 29. 최종 해 저장 시 index만 저장하면 안 되는 이유

다음 pointer trace만 보면 실제 해를 알 수 없다.

```text
[1, 1, 0]
```

복원하려면 step별 후보 목록이 필요하다.

```text
Step 0 candidates = [A, B, C], target=1 -> B
Step 1 candidates = [A, C],    target=1 -> C
Step 2 candidates = [A],       target=0 -> A
```

따라서 audit와 재현을 위해 다음을 함께 저장해야 한다.

### Phase 1

```text
candidate_ids
selected_action_index
selected_action_id
selected_block_set_id
selected_bay
```

### Phase 2

```text
selected_machine_id
action_ids 또는 candidate W/O IDs
selected_action_index
selected_job_ids
open batch와 target batch size
```

사람이 읽는 최종 산출물은 index trace가 아니라 다음이다.

```text
Phase 1: block-series별 assigned Bay CSV/JSON
Phase 2: W/O별 Machine assignment, batch CSV, timeline CSV/JSON
```

---

## 30. 자주 혼동하는 질문

### Q1. Agent가 출력하는 해는 index인가?

한 step의 신경망 직접 출력은 후보별 logit이고, 선택 결과는 local index다.
여러 step의 index를 모으면 pointer trace가 된다. 최종 해는 이를 실제 ID로
decode한 assignment/schedule이다.

### Q2. 휴리스틱도 index를 출력하는가?

휴리스틱의 본래 선택은 실제 pair 또는 W/O다. 학습에 사용할 때 현재 후보
목록에서 그 action의 위치를 찾아 local index로 변환한다.

### Q3. 같은 index가 반복돼도 되는가?

된다. 후보 목록이 매 step 다시 만들어지므로 같은 index가 다른 action을
가리킨다.

### Q4. Phase 1의 pair는 `(1,22)` 같은 숫자 입력인가?

아니다. `(G1,22)`는 사람이 읽는 의미 action이다. 모델 입력은 해당 pair의
20차원 feature row이며, CE target은 그 row의 local index다.

### Q5. Phase 2에서 Machine은 왜 target에 없는가?

환경이 event clock과 idle 상태에 따라 Machine을 결정하기 때문이다. 선택
Machine은 state/query와 최종 schedule에는 들어가지만 별도 CE class가 아니다.

### Q6. Batch 안에서 W/O 순서가 중요한가?

현재 처리시간이 `max(TACT_TIME)`이고 setup time이 없으므로 같은 batch 안의
순열은 운영 결과가 같다. 다만 policy는 batch를 한 개씩 구성하므로 학습
transition에는 대표 construction order 하나가 남는다.

### Q7. SPT인데 왜 큰 local index를 고르는가?

후보 index 순서는 TACT 내림차순으로 고정돼 있지만, SPT는 후보 전체에서 TACT
최솟값을 찾기 때문이다. local index의 크기에는 우선순위 의미가 없다.

---

## 31. 코드 기준 위치

| 내용 | 코드 |
|---|---|
| Phase 1 pair policy | `Phase1/pointer_policy.py` |
| Phase 1 rollout와 CE | `Phase1/pair_self_labeling.py` |
| Phase 1 휴리스틱 | `Phase1/heuristics.py` |
| Phase 1 hard mask | `Utils/phase1/multi_series_rules.py` |
| Phase 1 block/Bay score | `Utils/phase1/phase1_bay_balancer.py` |
| Phase 2 W/O-only state | `Phase2/state.py` |
| Phase 2 Set-Pointer policy | `Phase2/set_pointer_policy.py` |
| Phase 2 candidate/heuristic/CE | `Phase2/merged.py` |
| Phase 2 Machine dispatch와 batch 전이 | `Environment/hierarchical.py` |
| Phase 2 constraint profile | `Environment/constraints/profiles.py` |

---

## 32. CE target 핵심 정리

> Self-labeling은 가장 좋은 complete solution을 선택한 뒤, 그 해의 실제
> pair/W/O sequence를 매 step의 현재 후보 목록에 다시 대응시켜 얻은 local
> pointer index 또는 정책이 구분할 수 없는 동률 index 집합을 CE target으로
> 사용하고, 최종 결과에서는 index를 다시 실제 block-series/Bay 또는
> W/O/Machine/Batch ID로 복원하는 학습 방식이다.

---

## 33. 에이전트가 직접 선택하는 전체 흐름

앞의 Phase 1/2 예제는 선택된 해를 CE target으로 바꾸는 과정에 초점을 맞췄다.
이번 절은 **에이전트가 complete solution을 직접 생성하는 순간**을 설명한다.

에이전트 rollout과 CE update는 같은 과정이 아니다.

```text
[에이전트 rollout]
현재 state 생성
-> hard mask를 통과한 후보 생성
-> 후보별 logit 계산
-> greedy 또는 sampling으로 local index 선택
-> index를 실제 action으로 decode
-> 환경에 action 적용
-> complete solution이 끝날 때까지 반복

[self-labeling]
Agent complete solution들과 휴리스틱 complete solution들을 모두 평가
-> 가장 좋은 complete solution을 teacher로 선정

[CE update]
teacher의 각 step state와 후보 목록에서 모델을 다시 forward
-> teacher local index를 CE target으로 사용
-> step loss 평균
-> backward
-> optimizer update
```

따라서 에이전트가 rollout 중 선택한 `argmax` 또는 sampling 연산 자체를
미분하는 것이 아니다. rollout에서는 선택 결과를 저장하고, 최종 teacher가
정해진 뒤 teacher forcing으로 모델을 다시 실행해 CE gradient를 계산한다.

### 33.1 Greedy와 sampling

후보 logit이 다음과 같다고 가정한다.

```text
후보        C0    C1    C2
logit      0.4   1.7   0.9
```

Greedy는 가장 큰 logit의 index를 선택한다.

```text
argmax([0.4, 1.7, 0.9]) = 1
선택 action = C1
```

Sampling은 temperature `tau`를 적용한 확률에서 하나를 뽑는다.

\[
p_i =
\operatorname{softmax}\left(\frac{z_i}{\tau}\right)
\]

```text
확률 예시 = [0.16, 0.59, 0.25]
```

이 경우 C1이 가장 자주 선택되지만 C0 또는 C2도 선택될 수 있다. 이 탐색으로
현재 greedy보다 좋은 complete solution을 발견할 수 있다.

중요한 점:

```text
index 1이 항상 같은 action이라는 뜻이 아니다.
다음 step에서 후보 목록이 바뀌면 index 1이 가리키는 action도 바뀐다.
```

---

## 34. Phase 1 Agent가 pair를 선택하는 신경망 계산

Phase 1의 현재 후보 수를 `K`라고 하자.

```text
pair candidate feature X: (K, 20)
environment feature e:    (8,)
```

현재 구현의 계산은 다음과 같다.

\[
h_i = PairEncoder(x_i)
\]

\[
h_e = EnvEncoder(e)
\]

\[
q = Query\left(
  \operatorname{concat}
  \left(
    \frac{1}{K}\sum_{i=1}^{K}h_i,\,
    h_e
  \right)
\right)
\]

\[
z_i = Pointer\left(\tanh(h_i+q)\right)
\]

각 기호의 뜻은 다음과 같다.

| 기호 | 뜻 |
|---|---|
| `x_i` | 후보 `(block-series, Bay)` pair 하나의 20차원 feature |
| `h_i` | 해당 pair의 embedding |
| `e` | 현재 Bay 부하와 진행률을 담은 8차원 환경 feature |
| `q` | 전체 후보 요약과 현재 환경을 합친 query |
| `z_i` | 후보 `i`의 최종 logit |

Agent는 문자열 `G1@22`를 신경망에 직접 넣지 않는다. `G1@22`에서 계산한
20차원 feature를 넣고, 선택 후에는 local index를 다시 `G1@22`로 decode한다.

---

## 35. Phase 1 Agent Greedy 5-Step 선택 예제

이 절은 7~12절의 동일한 Phase 1 문제와 후보 목록을 사용한다.

아래 logit은 **선택 계산을 설명하기 위한 예시값**이다. 특정 checkpoint에서
실제로 출력된 수치가 아니다. 후보 순서와 index 변환은 실제 코드 계약과 같다.

### 35.1 Step 0

현재 후보는 8절의 12개 pair다.

```text
index  = [   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11]
action = [G1@22,G1@23,G2@22,G2@23,G2@24,G3@22,G3@23,G4@22,G4@23,G5@22,G5@23,G5@24]
logit  = [2.40,1.10,0.50,0.60,0.40,0.80,0.90,1.20,1.00,0.70,0.65,0.55]
```

```text
greedy selected index = argmax(logit) = 0
decode                 = candidates[0] = G1@22
실제 환경 action       = (P1::NP::B1, Bay22)
```

환경 변화:

```text
G1의 W/O=3, CUT=1200, BV=7이 Bay22에 누적
G1은 remaining set에서 제거
다음 후보 수는 12개에서 10개로 감소
```

### 35.2 Step 1

후보 목록은 9절처럼 다시 index 0부터 시작한다.

```text
index  = [   0,   1,   2,   3,   4,   5,   6,   7,   8,   9]
action = [G2@22,G2@23,G2@24,G3@22,G3@23,G4@22,G4@23,G5@22,G5@23,G5@24]
logit  = [0.50,0.70,0.80,1.10,1.00,0.90,1.20,0.60,0.75,2.30]
```

```text
greedy selected index = 9
decode                 = G5@24
```

환경 변화:

```text
G5의 W/O=3, CUT=900, BV=5가 Bay24에 누적
G5 제거
남은 그룹 = G2, G3, G4
```

### 35.3 Step 2

```text
index  = [   0,   1,   2,   3,   4,   5,   6]
action = [G2@22,G2@23,G2@24,G3@22,G3@23,G4@22,G4@23]
logit  = [0.70,0.60,0.40,1.00,2.10,0.90,0.80]
```

```text
greedy selected index = 4
decode                 = G3@23
```

환경 변화:

```text
G3의 W/O=1, CUT=500, BV=1이 Bay23에 누적
G3 제거
남은 그룹 = G2, G4
```

### 35.4 Step 3

```text
index  = [   0,   1,   2,   3,   4]
action = [G2@22,G2@23,G2@24,G4@22,G4@23]
logit  = [0.60,0.70,0.50,2.00,0.90]
```

```text
greedy selected index = 3
decode                 = G4@22
```

환경 변화:

```text
G4의 W/O=2, CUT=700, BV=4가 Bay22에 누적
G4 제거
남은 그룹 = G2
```

### 35.5 Step 4

```text
index  = [   0,   1,   2]
action = [G2@22,G2@23,G2@24]
logit  = [0.60,1.80,0.70]
```

```text
greedy selected index = 1
decode                 = G2@23
```

### 35.6 Phase 1 Agent가 만든 해

```text
Agent greedy local pointer trace
= [0, 9, 4, 3, 1]

decoded pair sequence
= [(G1,22), (G5,24), (G3,23), (G4,22), (G2,23)]

final assignment
= {
    G1: 22,
    G2: 23,
    G3: 23,
    G4: 22,
    G5: 24
  }
```

이 예제에서는 설명을 단순하게 하기 위해 Agent greedy가 앞의 teacher 예제와
같은 해를 만들었다. 실제 학습에서는 초기 모델의 logit이 다르므로 greedy와
각 sample의 trace가 서로 다르게 나온다.

Phase 1의 `rollout_samples=64`는 한 자원군에서 총 64개의 Agent rollout을
만든다.

```text
sample_index=1  -> agent_greedy
sample_index=2  -> agent_sample_2
...
sample_index=64 -> agent_sample_64
```

각 rollout은 hard mask를 통과한 후보만 선택하므로 NP 장척·광폭·CNT와 계열별
Bay eligibility를 위반할 수 없다.

---

## 36. Phase 2 Agent가 W/O를 선택하는 신경망 계산

Phase 2에서 Machine은 환경이 먼저 확정한다. Agent가 선택하는 것은 현재
Machine의 open batch에 추가할 W/O 하나다.

현재 구현은 다음 정보를 각각 MLP로 encoding한다.

```text
Machine node features
W/O node features
Bay context features
Open-batch features
후보 W/O 선택 후 projected features
```

계산 흐름은 다음과 같다.

\[
m_j=MachineEncoder(machine_j)
\]

\[
w_i=WOEncoder(wo_i)
\]

\[
b=BayEncoder(bay)
\]

\[
o=OpenBatchEncoder(openBatch)
\]

\[
p_i=ProjectedEncoder(projectedAction_i)
\]

\[
c=ContextEncoder(
  b,\,
  o,\,
  mean(m),\,
  max(m),\,
  mean(w),\,
  max(w)
)
\]

\[
q=QueryEncoder(c,m_{\text{selected machine}})
\]

\[
z_i=Pointer\left(\tanh(w_i+p_i+q)\right)
\]

즉 후보 W/O 점수에는 다음이 동시에 반영된다.

```text
W/O 자체 특성
현재 선택된 Machine 특성
현재 Bay의 전체 상태
현재 open batch 상태
해당 W/O를 추가했을 때의 예상 부하 변화
```

Machine은 CE target이 아니지만 `m_selected machine`이 query에 들어가므로
Machine encoder도 CE gradient를 받는다.

---

## 37. Phase 2 Agent Greedy 5-Step 선택 예제

이 절은 19~24절의 동일한 W/O 5개 문제를 사용한다.

아래 logit도 계산 과정을 설명하기 위한 예시이며 실제 checkpoint 출력값은
아니다.

### 37.1 Step 0

환경 상태:

```text
current_time = 0
idle machines = PLS51, PLS52
환경 selected Machine = PLS51
target batch size = 3
open batch = {}
```

Agent 후보와 logit:

| local index | W/O | TACT | LTH | Agent logit |
|---:|---|---:|---:|---:|
| 0 | WO1 | 50 | 20,000 | 0.80 |
| 1 | WO2 | 25 | 25,000 | 1.10 |
| 2 | WO3 | 20 | 35,000 | 0.50 |
| 3 | WO4 | 10 | 15,000 | 0.70 |
| 4 | WO5 | 9 | 10,000 | 2.00 |

```text
greedy selected index = 4
decode                 = WO5
환경 action            = add_to_batch(PLS51 open batch, WO5)
```

환경은 즉시 open batch를 갱신한다.

```text
W/O count = 1
LTH sum = 10,000
CUT sum = 80
BV sum = 0
max TACT = 9
```

### 37.2 Step 1

같은 PLS51 open batch에 두 번째 W/O를 고른다.

| local index | W/O | 선택 후 LTH | Agent logit |
|---:|---|---:|---:|
| 0 | WO1 | 30,000 | 0.60 |
| 1 | WO2 | 35,000 | 0.90 |
| 2 | WO3 | 45,000 | 0.40 |
| 3 | WO4 | 25,000 | 1.70 |

```text
greedy selected index = 3
decode                 = WO4
환경 action            = add_to_batch(PLS51 open batch, WO4)
```

환경 변화:

```text
open batch = {WO5, WO4}
W/O count = 2
LTH sum = 25,000
CUT sum = 180
BV sum = 1
max TACT = 10
```

### 37.3 Step 2

현재 LTH 합은 25,000이다.

```text
WO3 추가 후 LTH = 60,000
-> batch_length_sum_limit 위반
-> WO3는 후보 목록에서 제거
```

| local index | W/O | 선택 후 LTH | Agent logit |
|---:|---|---:|---:|
| 0 | WO1 | 45,000 | 0.70 |
| 1 | WO2 | 50,000 | 1.50 |

```text
greedy selected index = 1
decode                 = WO2
```

환경 변화:

```text
open batch = {WO5, WO4, WO2}
W/O count = 3
LTH sum = 50,000
CUT sum = 480
BV sum = 3
batch duration = max(9,10,25) = 25
target size 3 도달 -> close batch
PLS51 occupied interval = [0,25]
```

### 37.4 Step 3

PLS51은 사용 중이고 PLS52는 `t=0`에 유휴다. 환경은 event clock을 이동하지
않고 PLS52를 선택한다.

```text
current_time = 0
환경 selected Machine = PLS52
remaining W/O = WO1, WO3
target batch size = 2
```

| local index | W/O | LTH | Agent logit |
|---:|---|---:|---:|
| 0 | WO1 | 20,000 | 0.50 |
| 1 | WO3 | 35,000 | 1.40 |

```text
greedy selected index = 1
decode                 = WO3
환경 action            = add_to_batch(PLS52 open batch, WO3)
```

### 37.5 Step 4

남은 W/O는 WO1뿐이다.

```text
현재 open LTH 35,000 + WO1 LTH 20,000 = 55,000
55,000은 제한값과 같으므로 feasible
```

| local index | W/O | Agent logit |
|---:|---|---:|
| 0 | WO1 | 0.20 |

```text
greedy selected index = 0
decode                 = WO1
```

환경 변화:

```text
open batch = {WO3, WO1}
LTH sum = 55,000
CUT sum = 950
BV sum = 9
batch duration = max(20,50) = 50
target size 2 도달 -> close batch
PLS52 occupied interval = [0,50]
```

모든 W/O가 배정됐으므로 환경이 남은 완료 event를 drain한다.

```text
final current_time = makespan = 50
```

### 37.6 Phase 2 Agent가 만든 해

```text
Agent greedy local pointer trace
= [4, 3, 1, 1, 0]

decoded W/O construction sequence
= [WO5, WO4, WO2, WO3, WO1]

final schedule
= {
    PLS51: [{WO5, WO4, WO2}],
    PLS52: [{WO3, WO1}]
  }
```

local pointer trace와 W/O ID sequence를 혼동하면 안 된다.

```text
[4,3,1,1,0]             = 매 step 후보 목록의 위치
[WO5,WO4,WO2,WO3,WO1]   = 실제 선택된 W/O identity
```

Phase 2의 `rollout_samples=64`는 Bay subproblem마다 다음 Agent 후보를 만든다.

```text
agent_greedy 1개
agent_sample_1 ... agent_sample_64
```

따라서 Phase 2에서는 휴리스틱 수를 제외하고 Agent complete solution이 총
65개다.

---

## 38. Agent rollout이 CE 학습으로 바뀌는 정확한 순간

한 subproblem에서 다음 complete solution들이 있다고 가정한다.

```text
Agent greedy score   = (0, 55, ...)
Agent sample_1 score = (0, 52, ...)
Agent sample_2 score = (0, 49, ...)
SPT score            = (0, 50, ...)
LPT score            = (0, 47, ...)
```

사전식 score가 가장 작은 LPT가 teacher가 된다.

```text
teacher = LPT complete solution
```

이때 Agent greedy가 rollout에서 선택했던 index를 정답으로 쓰지 않는다.
LPT의 실제 action sequence를 현재 후보 목록에 다시 대응시켜 만든 local index를
정답으로 사용한다.

```text
LPT teacher trace = [0, 0, 1, 0, 0]  # 설명용 예시
```

각 teacher step에서 모델을 gradient가 연결된 상태로 다시 forward한다.

\[
L_t=CE(z_t,i_t^*)
\]

\[
L=\frac{1}{T}\sum_{t=0}^{T-1}L_t
\]

```text
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

반대로 Agent sample_2가 가장 좋았다면 그 Agent rollout에 저장된 state,
candidate list, selected local index가 그대로 teacher transition이 된다.

### 38.1 어떤 parameter가 학습되는가?

Phase 1:

```text
Pair encoder
Environment encoder
Query MLP
Pointer scorer
```

Phase 2:

```text
Machine encoder
W/O encoder
Bay encoder
Open-batch encoder
Projected-action encoder
Context encoder
Query encoder
Pointer scorer
```

teacher target index의 확률은 높아지고, 경쟁 후보의 확률은 상대적으로
낮아진다. 다음 episode에서는 같은 형태의 state에서 teacher와 비슷한 선택을
할 가능성이 커진다.

---

## 39. Agent 선택 과정 최종 정리

```text
Phase 1
20차원 pair 후보 + 8차원 환경
-> 후보별 logit
-> local index 선택
-> (block-series, Bay)로 decode

Phase 2
Machine/W/O/Bay/open-batch/projected state
-> 환경이 선택한 Machine을 query에 반영
-> W/O 후보별 logit
-> local index 선택
-> 실제 W/O ID로 decode

공통 학습
Agent와 휴리스틱이 complete solution 생성
-> 사전식 score 최소 해를 teacher로 선택
-> teacher local index trace로 CE
-> 모든 encoder와 pointer parameter update
```
