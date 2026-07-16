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

- 최댓값: `LTH`, `BTH`, `THK`, `MARK_LTH`, `TACT_TIME`
- 합계: `CUT_LTH`, `BVL_LTH`, `PTLST_QTY`, `STL_QTY`, `BV_QTY`, `CURVE_QTY`
- `WO_QTY`: 해당 block-series의 W/O 행 수
- `STL_QTY`: W/O `STL_QTY` 합
- `WO_QTY`와 `STL_QTY`는 서로 대체하지 않는다.

필수 컬럼 누락, block-W/O FK 불일치, 집계 identity 불일치는 원인을 출력하고
실패한다. `RET_QTY`는 모델 입력, 목적, 제약에서 사용하지 않는다.

## 4. MIXED 합성데이터

### 4.1 물리 블록마다 생성되는 계열 기준

모든 물리 블록에 NP/FN/FL/NC가 무조건 생기지 않는다. 기준 모집단은 신규 실적의
물리 블록 `(PROJ_NO, BLK_NO)` 1,044개다. 먼저 기존 발표자료 블록 수식으로 물리
블록의 전체 `WO_QTY`를 생성한다. 그 수보다 계열 수가 많거나 계열별 생성 지원범위로
배분할 수 없는 조합을 제외한 뒤, 아래 실적 조합 확률을 재정규화해 계열 조합 하나를
표본화한다. 계열별 독립 Bernoulli 추출이나 W/O별 무작위 계열 부여는 하지 않는다.

| 실적 계열 조합 | 물리 블록 수 | 생성 확률 |
|---|---:|---:|
| NP | 362 | 34.6743% |
| FL | 130 | 12.4521% |
| FN | 21 | 2.0115% |
| NC | 33 | 3.1609% |
| FL+NP | 274 | 26.2452% |
| FN+NP | 64 | 6.1303% |
| NC+NP | 55 | 5.2682% |
| FL+FN | 5 | 0.4789% |
| FN+NC | 1 | 0.0958% |
| FL+FN+NP | 35 | 3.3525% |
| FL+NC+NP | 25 | 2.3946% |
| FN+NC+NP | 38 | 3.6398% |
| FL+FN+NC+NP | 1 | 0.0958% |
| **합계** | **1,044** | **100.0000%** |

| 한 물리 블록의 계열 수 | 물리 블록 수 | 비율 |
|---:|---:|---:|
| 1계열 | 546 | 52.2989% |
| 2계열 | 399 | 38.2184% |
| 3계열 | 98 | 9.3870% |
| 4계열 | 1 | 0.0958% |

예를 들어 전체 `WO_QTY=10`인 물리 블록에서 `NP+FL`이 선택되면 실적의 조건부
계열 count vector를 기준으로 `N_NP+N_FL=10`, `N_NP>=1`, `N_FL>=1`이 되도록
정수 배분한다. NP W/O는 전부 `GYEL=NP`, FL W/O는 전부 `GYEL=FL`이다. 같은
block-series의 모든 W/O는 Phase 1에서 반드시 같은 Bay로 가지만, NP와 FL은 서로
다른 Bay를 선택할 수 있다.

### 4.2 한 episode의 생성 순서

1. `min_blocks..max_blocks` 범위에서 물리 블록 수를 뽑는다.
2. 각 물리 블록에 대해 기존 발표자료 블록 수식으로 `LTH/THK/CUT_LTH` 목표와 전체
   `WO_QTY`를 한 번만 생성한다. 이 `WO_QTY`는 이후 단계에서 다른 지원값으로 바꾸지
   않는다.
3. 전체 `WO_QTY`로 배분 가능한 계열 조합만 남기고, 실적 조합 확률을 재정규화해
   하나를 표본화한다. 따라서 한 episode에 FN, FL 또는 NC가 없을 수 있다.
4. 선택 조합 안에서 생성 물리 특성 `LTH/THK/CUT_LTH/WO_QTY`의 percentile과 가까운
   실적 물리 블록 8개를 찾고, 거리 역수 확률로 조건부 count vector profile 하나를
   선택한다. 8-neighbor는 숨은 fallback이 아니라 현재 고정된 조건부 표본화 계약이다.
5. 실적 count vector의 비율을 유지하는 최소제곱 정수 배분으로 계열별 `WO_QTY`를
   계산한다. 선택 계열은 각각 1개 이상이고, 합은 원 수식 전체 `WO_QTY`와 정확히
   같아야 한다. FN/FL/NC는 해당 계열의 관측 최대 W/O 수를 넘기지 않으며 NP가 포함된
   조합에서는 NP가 남은 합을 수용할 수 있다.
6. 계열별 생성기에 확정된 `WO_QTY`를 입력해 W/O 특성을 생성한다. 같은 물리 블록의
   계열들은 공통 물리 seed에서 파생하되 서로 다른 결정론적 난수 stream을 사용하므로,
   동일 난수열 때문에 가짜 완전상관이 생기지 않는다.
7. 생성 순서 그대로 새 `PROJ_NO/BLK_NO/WK_ORD_NO`를 부여한다. 생성 이후 block swap,
   Hungarian 매칭, 계열 교환, W/O 수 재조정은 하지 않는다.
8. W/O를 `(PROJ_NO, BLK_NO, GYEL)`로 역집계해 block-series를 만든다. validator는
   `원 수식 WO_QTY = 계열별 배분 합 = 실제 W/O 행 수`, 계열 조합, W/O identity와
   모든 block-series 집계값을 검사하고 불일치 시 실패한다.
9. Phase 1과 Phase 2는 이 공용 episode를 같은 seed로 호출하므로 동일한 DataFrame과
   W/O identity를 받는다. episode는 파일로 미리 쌓지 않고 매 episode 메모리에서
   생성한다.

실적 profile의 W/O 값을 합성 정답으로 복사하지 않는다. profile은 계열 조합과 count
vector의 조건부 구조만 제공하고, W/O 물리값은 계열별 생성식으로 새로 만든다.

### 4.3 계열별 수식 계약

“모든 계열이 NP 수식에서 절편만 바꾼다”는 설명은 정확하지 않다. NP는 발표자료
고정식과 고정 오차분포를 보존하고, FN/FL/NC 세 계열은 서로 같은 empirical 생성
코드와 식 종류를 공유하되 계열별 실적에서 계수·절편·잔차분포·지원범위를 각각
적합한다.

| 항목 | NP | FN/FL/NC | 계열 간 동일·상이 기준 |
|---|---|---|---|
| 블록 LTH | 발표자료 절단정규분포 고정값 | 계열별 정규분포 적합 | 분포 형태는 유사, 평균·표준편차·범위는 다름 |
| 블록 MARK/CUT/W/O 수 | 발표자료 고정 회귀·감마오차 | 같은 선형 사슬에 계열별 기울기·절편·정규잔차 | 식의 계층 순서는 같지만 오차분포까지 같지는 않음 |
| 블록 THK | 발표자료 `ln(W/O 수)` 고정식 후 규격 스냅 | 같은 `ln(W/O 수)` 형태에 계열별 계수·잔차·규격 | 형태는 같고 값과 지원범위가 다름 |
| W/O LTH | 발표자료 고정 멱감쇠식 | 공통 멱감쇠·인접 병합 구조를 계열별 적합 | 구조는 같고 파라미터·병합확률이 다름 |
| W/O THK | 발표자료 고정 조건부 skew 분포 | 길이비 조건부 skew 분포·copula를 계열별 적합 | 구조와 fitting 절차는 같고 분포 파라미터가 다름 |
| W/O MARK/CUT | 발표자료 고정 선형식 후 블록 합계 보정 | 계열별 log-log 식 후 신규 block 집계 계약 보정 | NP와 비NP의 세부 식은 동일하지 않음 |
| W/O BVL/BV/PTLST | 발표자료 고정 선형식 | 공통 hurdle·log-log 식에 계열별 계수·잔차 | 생성 관계는 같지만 세부 식·분포가 다름 |
| W/O BTH | 전 계열 공통 로그선형식 | 전 계열 공통 로그선형식 | 절편·계수·잔차·관측 규격만 계열별 적합 |
| W/O STL_QTY | 계열별 조건부 범주분포 | 계열별 조건부 범주분포 | 회귀 절편 치환이 아니라 조건부 이웃확률 75%+주변분포 25% |
| W/O TACT_TIME | Case 6 고정식 | 같은 Case 6 고정식 | 식과 계수가 전 계열 완전히 동일 |

`BTH`의 정확한 공통식은 다음과 같다. 입력 특성은
`LTH/THK/MARK_LTH/CUT_LTH/BVL_LTH/BV_QTY/PTLST_QTY`이며, 각 계열 실적에서
`b0`, 각 `bk`, 잔차 표준편차와 허용 폭 규격을 별도로 적합한다.

```text
ln(BTH) = b0 + sum(bk * ln(1 + xk)) + epsilon
epsilon ~ Normal(0, series_residual_std^2)
```

생성 연속값은 해당 계열에서 실제 관측된 BTH 규격 중 가장 가까운 값으로 스냅한다.
실적 적합 설명력은 NP `R²=0.665831`, FN `0.617106`, FL `0.252072`, NC
`0.356277`이며, 낮은 설명력을 숨기기 위해 잔차를 줄이거나 값을 조작하지 않는다.

`STL_QTY`는 `WO_QTY`와 별개다. 같은 계열 실적에서 핵심 가공특성이 가까운 이웃의
범주확률 75%와 계열 전체 주변확률 25%를 결합해 정수값을 생성한다. 블록
`STL_QTY`는 W/O 값의 합이고, `WO_QTY`는 W/O 행 수다.

### 4.4 TACT_TIME 공통식

전 계열 W/O `TACT_TIME`은 현재 확정된 Case 6 식과 동일 계수를 사용한다.

```text
TACT_TIME = 0.3037 * CUT_LTH
          + 0.1325 * MARK_LTH
          + 0.4790 * THK
          + 0.3840 * PTLST_QTY
```

`BV_QTY`를 TACT 식에 임의로 추가하지 않는다. actual start-end elapsed time도
TACT_TIME 대체값으로 사용하지 않는다.

### 4.5 실적·생성 상관관계 검증 산출물

다음 명령은 같은 MIXED 생성 결과 하나를 기준으로 히트맵 16개와 CSV 2개를 만든다.

```bash
python scripts/plot_multi_series_generator_heatmaps.py \
  --seed 20260716 \
  --output-dir output/analysis/mixed_causal_formula_total_20260716
```

- 계열별 W/O/블록 실적·생성 비교: 8개
- 전체 물리 블록 24×24 계열 간 공동분포 비교: 1개
- NP-FN, NP-FL, NP-NC, FN-FL, FN-NC, FL-NC 확대 비교: 6개
- 전체 계열 간 상관계수 절대오차: 1개
- 관계별 원시값: `physical_block_cross_series_correlations.csv`
- 범위별 MAE 요약: `correlation_error_summary.csv`

seed `20260716` 검증은 물리 블록 1,044개에서 series-block 1,610개와 W/O
9,221개를 생성했다. 실적 계열 조합분포와 생성 조합분포의 총변동거리는 `0.034483`,
최대 조합 비율 오차는 `1.8199%p`다. 계열 간 216개 상관관계의 전체 MAE는
`0.278174`다. 생성 W/O 총수는 실적 행 수를 복제한 값이 아니라 1,044개 물리 블록의
발표자료 원 수식 `WO_QTY` 합이므로 실적 11,646행과 같을 필요가 없다.

| 계열 쌍 | 실적 동일 블록 수 | 생성 동일 블록 수 | 상관 MAE |
|---|---:|---:|---:|
| NP-FN | 138 | 121 | 0.266234 |
| NP-FL | 335 | 331 | 0.190452 |
| NP-NC | 119 | 107 | 0.233150 |
| FN-FL | 41 | 32 | 0.339274 |
| FN-NC | 40 | 32 | 0.349373 |
| FL-NC | 26 | 22 | 0.290560 |

NP-FL은 현재 계열 쌍 중 오차가 가장 작고, FN-NC는 가장 크다. FL-NC 26개,
FN-NC 40개처럼 희소한 실적 관계는 표본 불확실성도 크다. 사후 Hungarian 재배치를
제거했으므로 과거 비인과 구현의 낮은 MAE와 직접 비교해 성능 향상으로 주장하지 않는다.
현재 생성기는 인과 순서, 조합분포, 전체/계열별 W/O 수 identity를 우선 보장하며 모든
희소 계열 간 상관을 완전 재현한다고 주장하지 않는다.

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
