# Phase 1 MDP: 블록-계열별 Bay 배정

> Phase 1은 같은 `PROJ_NO + GYEL + BLK_NO`의 W/O를 하나의 블록-계열 단위로 묶고, 생산 제약과 Bay별 설비 수를 고려해 Bay를 결정하는 단계이다.

## 1. 입력과 출력

| 구분 | 내용 |
|---|---|
| 입력 | NP/FN/FL/NC W/O와 블록 특성 |
| 의사결정 단위 | `PROJ_NO::GYEL::BLK_NO` |
| 후보 Bay | 22, 23, 24, 25, trans |
| 출력 | 블록-계열별 Bay 배정 |

같은 물리 블록이라도 NP와 NC는 서로 다른 의사결정 단위다. 단, 같은 블록-계열의 모든 W/O는 같은 Bay로 간다.

## 2. MDP 정의

| 구성 | Phase 1에서의 의미 |
|---|---|
| State | 현재 자원군의 남은 블록-계열, Bay별 누적 부하, 현재 부하 편차 |
| Action | `(block-series, Bay)` pair 하나 선택 |
| Transition | 선택 블록의 부하를 Bay에 더하고 남은 후보에서 제거 |
| Objective | 자원군 내 계열별 W/O 수, CUT_LTH, BV_QTY 부하평준화 |
| Termination | 현재 자원군의 모든 블록-계열이 배정됨 |

한 물리 episode는 설비를 공유하는 두 서브문제로 나뉜다.

| 서브문제 | 계열 | Bay |
|---|---|---|
| `NP_NC` | NP, NC | 22, 23, 24 |
| `FN_FL` | FN, FL | 25, trans |

같은 policy parameter를 사용하지만 후보 bank, teacher, CE update는 서브문제별로
독립한다. 한쪽 자원군이 episode에 없으면 그 서브문제는 `0`으로 채우지
않고 아예 생성하지 않는다.

## 3. Action

```text
a_t = (선택할 블록-계열, 배정할 Bay)
```

예를 들어 NP 블록 `B1`이 Bay 22와 23에만 갈 수 있다면 후보는 다음 두 개다.

```text
(B1, Bay22)
(B1, Bay23)
```

에이전트는 이 pair 중 하나를 선택하므로 **블록 선택 순서와 Bay 배정을 동시에 결정**한다.

## 4. Hard Mask

| 계열/조건 | 선택 가능 Bay |
|---|---|
| NP 기본 | 22, 23, 24 |
| NC 기본 | 22, 23, 24 |
| FN, FL | 25, trans |
| NP block-series `CUT_LTH` 합 `>= 1000` | 22, 23 |
| NP `BTH > 4500` | 22, 23 |
| NP `CNT_BLK` | 22, 23 |

제약을 위반하는 pair는 점수를 낮게 주는 것이 아니라 action 후보에서 제거한다.

장척 판정값은 동일 `PROJ_NO+GYEL+BLK_NO`에 속한 W/O 절단장의 합이다.

```text
block-series CUT_LTH = sum(W/O CUT_LTH)
```

개별 W/O의 최댓값을 사용하지 않는다. 합계가 1,000 이상이면 해당 block-series의
모든 W/O가 함께 상속할 수 있는 Bay 후보를 22/23으로 제한한다.

## 5. State Feature

| 구분 | 차원 | 내용 |
|---|---:|---|
| Pair feature | 20 | 블록 부하, NP/NC/FN/FL flag, 제약 여부, Bay 현재 부하, 배정 후 예상 gap |
| Environment feature | 8 | 진행률, 남은 블록 비율, 현재 W/O·CUT·BV gap |

현재 정책은 graph의 feasible edge를 20차원 pair로 표현하는 **MLP 기반 Pointer
Policy**다. GNN message passing은 사용하지 않는다. 현재 서브문제와 관계없는
Bay edge는 hard mask 다음 후보 matrix에 들어오지 않는다.

## 6. 설비 수 반영

| Bay | 설비 수 |
|---|---:|
| 22 | 4 |
| 23 | 3 |
| 24 | 4 |
| 25 | 2 |
| trans | 2 |

평준화는 Bay 총량이 아니라 `Bay 부하 / 설비 수`를 기준으로 계산한다. 따라서 Bay 22/23/24의 W/O가 `40/30/40`이면 설비당 W/O는 모두 10으로 gap은 0이다.

## 7. 자원군별 사전식 목적함수

```text
NP_NC score = (
  NP W/O gap + NC W/O gap,
  NP CUT_LTH gap + NC CUT_LTH gap,
  NP BV_QTY gap + NC BV_QTY gap
)

FN_FL score = (
  FN W/O gap + FL W/O gap,
  FN CUT_LTH gap + FL CUT_LTH gap,
  FN BV_QTY gap + FL BV_QTY gap
)
```

각 gap은 해당 계열의 Bay별 부하를 Bay 설비 수로 나눈 뒤 `max-min`으로
계산한다. 예를 들어 `NP_NC` teacher 비교에 FN/FL gap은 사용하지 않는다.
두 score의 합은 parent report에 기록할 수 있지만 teacher 선정에는 사용하지 않는다.

`shared_and_series`를 선택하면 각 서브문제 안에서만 공유 자원군 전체 gap을
계열별 gap 앞에 추가한다. 기본 권장 학습은 `--objective-scope series_only`다.

## 8. Self-labeling 학습

```text
부모 합성 episode 생성
→ NP_NC / FN_FL로 분할
→ NP_NC Agent greedy/sampling + 휴리스틱 비교
→ NP_NC teacher sequence로 CE update
→ FN_FL Agent greedy/sampling + 휴리스틱 비교
→ FN_FL teacher sequence로 CE update
→ 두 assignment를 하나의 parent plan으로 병합
```

`--rollout-samples 64`는 **각 서브문제당** `greedy 1개 + sampling 63개`를
의미한다. 여기에 W/O-first, CUT-first, Bevel-first 휴리스틱 3개를 각각
추가한다. 하나의 자원군만 있는 episode는 update 1번, 두 자원군이 모두 있는
episode는 같은 policy에 update 2번을 수행한다.

## 9. Validation

하나의 holdout parent를 `NP`, `NC`, `NP+NC`, `FN`, `FL`, `FN+FL` view로
나눠 평가한다. 없는 계열의 gap은 `0`으로 저장하지 않고 CSV에서 빈값/N/A로
남긴다. 계열별 gap과 자원군 전체 gap은 진단용으로 모두 저장하지만,
`series_only` teacher/rank는 위 3개 계열별 gap 합만 사용한다.

## 10. Phase 2 연결

Phase 1이 두 자원군의 assignment를 합쳐 `block-series -> Bay`를 확정하면 해당
블록-계열의 모든 W/O가 같은 Bay를 상속한다. Phase 2는 이 W/O들을 Bay
내부 설비와 batch에 배정한다. frozen Phase 1 checkpoint도 각 자원군에서
best-of-K를 별도로 선정한 뒤 병합한다.
