# 다계열 학습데이터 생성: 물리 블록부터 W/O까지

> 이 문서는 현재 MIXED Phase 1/2 학습에서 실제로 사용하는 합성데이터 생성 흐름과 수식을 Notion 발표용으로 정리한 것이다. NP는 발표자료 고정식을 사용하고, FN/FL/NC는 같은 생성 구조에 계열별 고정 계수와 잔차분포를 사용한다. 이 값은 실적 Excel에서 오프라인으로 한 번 산출한 JSON profile에 고정하며, 학습 중에는 Excel을 다시 적합하지 않는다.

## 1. 한 줄 요약

```text
물리 블록 특성 생성
-> 물리 블록 전체 W/O 수 생성
-> 실적 기반 계열 조합 선택
-> 계열별 W/O 수 배분
-> NP/FN/FL/NC별 W/O 특성 생성
-> W/O를 block-series 단위로 역집계
-> Phase 1/2가 같은 episode를 사용
```

중요한 순서는 다음과 같다.

```text
물리 블록 -> 전체 WO_QTY -> 계열 조합 -> 계열별 WO_QTY -> W/O 특성 -> block-series
```

- W/O를 먼저 계열별로 따로 만든 뒤 합쳐 전체 W/O 수를 정하지 않는다.
- W/O별 계열을 독립 확률로 무작위 부여하지 않는다.
- 생성 후 상관관계를 맞추기 위한 block swap, W/O swap, Hungarian 재배치를 하지 않는다.
- `WO_QTY`는 W/O 행 수이고 `STL_QTY`는 강재수량이므로 서로 대체하지 않는다.
- episode는 파일로 미리 쌓지 않고 학습 episode마다 메모리에서 생성한다.

## 2. 사용 코드 위치

| 역할 | 코드 위치 | 핵심 함수 |
|---|---|---|
| 다계열 전체 생성 제어 | `Utils/data/multi_series_formula_data_generator.py` | `generate_multi_series_formula_data()` |
| 고정 profile 로드·검증 | 같은 파일 | `load_multi_series_generation_profile()` |
| 물리 블록별 계열 조합·W/O 수 배분 | 같은 파일 | `select_physical_block_series_allocations()` |
| 계열별 생성기 호출 | 같은 파일 | `_generate_one_series()` |
| block-series 역집계 | 같은 파일 | `_aggregate_multi_series_blocks()` |
| 전체 identity·집계 검증 | 같은 파일 | `validate_multi_series_formula_data()` |
| NP 발표자료 고정 수식 | `Utils/data/report_formula_data_generator.py` | `_generate_block_seed_values()`, `_generate_work_order_rows()` |
| NP BTH·STL_QTY 생성 | 같은 파일 | `fit_bth_formula()`, `sample_bth_formula()`, `sample_stl_quantity()` |
| FN/FL/NC 계열별 profile 직렬화·복원 | `데이터분석/shipyard_data_generator.py` | `ShipyardGenerator.to_generation_profile()`, `ShipyardGenerator.from_generation_profile()` |
| FN/FL/NC 계열별 W/O 생성 | 같은 파일 | `ShipyardGenerator._gen_one()` |
| profile 오프라인 재생성 | `scripts/build_multi_series_generation_profile.py` | `write_multi_series_generation_profile()` |
| 학습용 고정 profile | `Utils/data/multi_series_generation_profile.json` | MIXED Phase 1/2의 유일한 적합 파라미터 입력 |
| Phase 1/2 공용 episode 진입점 | `Utils/phase1/phase1_episode_dataset.py` | `build_phase1_episode_jobs()` |
| NP/MIXED 생성기 분기 | `Utils/data/report_formula_data_generator.py` | `build_report_formula_episode_jobs()` |
| Phase 1 on-the-fly 호출 | `main.py` | `command_phase1_train_pair_self_labeling()` 내부 `episode_payload()` |
| Phase 2 on-the-fly 호출 | `main.py` | `command_phase2_train_batch_machine_self_labeling()` 내부 `mixed_jobs()` |

실행 호출 관계는 다음과 같다.

```text
main.py
-> build_phase1_episode_jobs()
-> build_report_formula_episode_jobs(gyel="MIXED")
-> build_multi_series_formula_episode_jobs()
-> load_multi_series_generation_profile()
-> generate_multi_series_formula_data()
```

## 3. 실적 원천과 데이터 규모

| 구분 | 파일 |
|---|---|
| W/O 실적 | `변경사항/절단WO_데이터.xlsx` |
| 블록 실적 | `변경사항/절단블록_데이터.xlsx` |
| 학습용 고정 profile | `Utils/data/multi_series_generation_profile.json` |

Excel 두 파일은 profile을 명시적으로 재생성하고 검증할 때만 사용한다. 일반 Phase 1/2 학습은 고정 JSON만 읽으므로 Excel을 매 episode 다시 읽거나 계수를 재적합하지 않는다.

| 계열 | 실적 W/O 수 | 실적 block-series 수 |
|---|---:|---:|
| NP | 7,355 | 854 |
| FN | 333 | 165 |
| FL | 3,520 | 470 |
| NC | 438 | 153 |
| **합계** | **11,646** | **1,642** |

- 실적 물리 블록 `(PROJ_NO, BLK_NO)`은 1,044개다.
- block-series identity는 `(PROJ_NO, GYEL, BLK_NO)`다.
- W/O identity는 `WK_ORD_NO`다.
- 같은 물리 블록에 여러 계열이 존재할 수 있다.
- 같은 block-series의 모든 W/O는 Phase 1에서 같은 Bay로 가야 한다.
- 같은 물리 블록이라도 계열이 다르면 서로 다른 Bay로 갈 수 있다.

## 4. 전체 생성 Flow

### Step 1. episode의 물리 블록 수 결정

학습 명령의 `--min-blocks`, `--max-blocks` 범위에서 episode의 물리 블록 수를 하나 뽑는다.

```text
n_physical_blocks ~ DiscreteUniform(min_blocks, max_blocks)
```

여기서 block 수는 block-series 수가 아니라 `(PROJ_NO, BLK_NO)` 기준 물리 블록 수다. 한 물리 블록에 여러 계열이 선택되면 최종 Phase 1 의사결정 block-series 수는 더 많아진다.

### Step 2. 물리 블록 특성과 전체 W/O 수 생성

각 물리 블록마다 NP 발표자료 블록 수식을 한 번 실행해 다음 seed를 생성한다.

```text
LTH -> MARK_LTH -> CUT_LTH -> 전체 WO_QTY -> THK
```

이 단계의 `WO_QTY`가 해당 물리 블록의 최종 전체 W/O 수다. 이후 계열 배분 단계에서 다른 값으로 대체하지 않는다.

### Step 3. 계열 조합 선택

전체 `WO_QTY`보다 계열 수가 많은 조합과 계열별 관측 최대 W/O 수로 배분할 수 없는 조합을 제외한다. 남은 조합의 실적 확률을 다시 합이 1이 되도록 정규화한 뒤 하나를 선택한다.

| 계열 조합 | 실적 물리 블록 수 | 선택 확률 |
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

따라서 한 물리 블록에 모든 계열이 무조건 생기지 않으며, 한 episode에 FN·FL·NC가 하나도 없을 수도 있다.

| 물리 블록 내 계열 수 | 실적 비율 |
|---:|---:|
| 1계열 | 52.2989% |
| 2계열 | 38.2184% |
| 3계열 | 9.3870% |
| 4계열 | 0.0958% |

### Step 4. 조건부 계열별 W/O 수 배분

선택한 계열 조합이 같은 실적 물리 블록만 후보로 둔다. 생성 물리 블록과 실적 후보를 다음 4개 percentile로 비교한다.

```text
LTH percentile
THK percentile
CUT_LTH percentile
전체 WO_QTY percentile
```

가까운 실적 후보 8개를 고른 뒤 거리 역수 확률로 실적 count vector 하나를 선택한다. 정확히 같은 후보가 있으면 동일 거리 후보끼리 균등 선택한다.

실적 count vector가 `(NP=6, FL=2)`이고 생성 전체 `WO_QTY=10`이면 원 비율 `6:2`를 유지하도록 양의 정수로 확대한다.

```text
ideal_g = actual_count_g * generated_total / actual_total

minimize sum_g (allocated_g - ideal_g)^2
subject to
  allocated_g >= 1
  sum_g allocated_g = generated_total
  FN/FL/NC는 각 계열의 실적 관측 최대값 이하
```

결과 예시는 `(NP=8, FL=2)`다. 생성 후 W/O 수를 다시 조정하지 않는다.

### Step 5. 계열별 W/O 특성 생성

- NP는 발표자료 고정 수식과 고정 오차분포를 사용한다.
- FN/FL/NC는 같은 생성 구조를 사용하지만 고정 JSON에 저장된 계열별 계수·잔차·입력 지원 규격을 사용한다.
- 각 계열은 확정된 `SERIES_WO_QTY`만큼 정확히 W/O를 생성한다.
- 같은 물리 블록의 계열들은 공통 물리 seed에서 파생된 서로 다른 난수 stream을 사용한다.

Step 2의 물리 블록 `LTH/THK/CUT_LTH` seed는 계열 조합과 count vector profile을 조건부 선택하는 기준이다. 계열 조합이 정해진 뒤 최종 NP/FN/FL/NC W/O 물리값은 각 계열 생성식으로 새로 생성한다. 실적 profile의 W/O 값을 복사하지 않는다.

### Step 6. identity 부여

```text
PROJ_NO    = SYNTH_MULTI_PROJ_...
BLK_NO     = SYNTH_MULTI_BLK_...
WK_ORD_NO  = SYNTH_MULTI_WO_<physical>_<series>_<index>
GYEL       = NP / FN / FL / NC
```

### Step 7. block-series 역집계

생성 W/O를 `(PROJ_NO, BLK_NO, GYEL)`로 묶어 최종 block-series 데이터를 만든다.

| 블록 특성 | W/O에서 집계하는 방법 |
|---|---|
| `WO_QTY` | W/O 행 수 |
| `LTH` | 최댓값 |
| `BTH` | 최댓값 |
| `THK` | 최댓값 |
| `MARK_LTH` | 최댓값 |
| `TACT_TIME` | 최댓값 |
| `CUT_LTH` | 합계 |
| `BVL_LTH` | 합계 |
| `STL_QTY` | 합계 |
| `BV_QTY` | 합계 |
| `PTLST_QTY` | 합계 |

### Step 8. 검증 후 학습 환경 전달

validator는 다음을 모두 검사한다.

```text
원 수식 전체 WO_QTY
= 계열별 SERIES_WO_QTY 합
= 실제 생성 W/O 행 수

W/O 역집계값
= 최종 block-series 값
```

NaN, 음수, 중복 W/O identity, 계열 조합 불일치, 집계 불일치가 있으면 원인을 출력하고 즉시 실패한다. 다른 값으로 조용히 대체하지 않는다.

## 5. NP 물리 블록 고정 수식

NP 발표자료 고정식은 물리 블록 seed와 NP W/O 생성에 사용한다.

### 5.1 블록 LTH

```text
LTH_raw ~ Normal(13,380.9, 5,376.4^2)
LTH = clip(LTH_raw, 515, 21,995)
```

### 5.2 블록 MARK_LTH

```text
P(MARK_LTH = 0) = 0.0127

그 외:
MARK_LTH = max(0,
  0.03425 * LTH
  - 178.9
  + Gamma(shape=3.049, scale=150.8)
  - 459.9)
```

### 5.3 블록 CUT_LTH

```text
CUT_LTH = max(0,
  (1.7056 * MARK_LTH + 59.21)
  * Gamma(shape=3.391, scale=0.292))
```

### 5.4 물리 블록 전체 WO_QTY

```text
WO_QTY = max(1, round(
  (0.01203 * CUT_LTH + 2.014)
  * Gamma(shape=7.616, scale=0.128)))
```

이 값은 `STL_QTY`가 아니다. 다계열 생성에서는 이 값이 먼저 확정된 후 계열별 W/O 수로 분배된다.

### 5.5 블록 THK

```text
THK_raw = 3.7816 * ln(WO_QTY) + 13.31 + Normal(0, 4.797^2)
THK = nearest_observed_thickness_spec(THK_raw)
```

NP 두께 스냅은 코드에 선언된 실제 관측 규격 6.0~36.0mm, 총 45개를 사용한다.

## 6. NP W/O 고정 수식

### 6.1 W/O LTH

W/O index를 `i=0..N-1`, `u_i=i/N`으로 둔다.

```text
factor_i = 1 - (0.95 - 0.86/N) * u_i^0.89
LTH_i = max(1, block_LTH * factor_i)
```

첫 W/O의 길이는 블록 LTH와 같고 이후 W/O는 index에 따라 감소한다.

### 6.2 W/O THK

```text
r_i = LTH_i / block_LTH

THK_raw_i = 12.4
          + 6.7 * exp(-2.85 * r_i)
          + SkewNormal(shape=7.09, loc=-1.15, scale=1.53)

THK_i = min(block_THK, nearest_observed_spec(THK_raw_i))
```

최종적으로 적어도 한 W/O의 THK가 블록 THK와 같도록 보정한다.

### 6.3 W/O CUT_LTH

```text
CUT_raw_i ~ Normal(0.0058 * LTH_i + 12.649, 42.072^2)
```

음이 아닌 표본을 만든 뒤 순서를 유지하면서 다음 조건을 정확히 맞춘다.

```text
sum_i CUT_LTH_i = block CUT_LTH seed
```

### 6.4 W/O MARK_LTH

```text
MARK_raw_i ~ Normal(0.0047 * LTH_i - 8.293, 17.452^2)
```

NP 고정 생성기 안에서는 다음 합계 조건으로 W/O 상대값을 연결한다.

```text
sum_i MARK_LTH_i = block MARK_LTH seed
```

최종 MIXED block-series 파일의 `MARK_LTH`는 신규 실적 계약에 따라 W/O 최댓값으로 다시 집계한다.

### 6.5 W/O BVL_LTH

```text
BVL_LTH = max(0,
  -12.496
  + 0.827  * THK
  + 0.0004 * LTH
  + 0.0649 * MARK_LTH
  + Normal(0, 7.602^2))
```

### 6.6 W/O BV_QTY

```text
if BVL_LTH = 0:
  BV_QTY = 0
else:
  BV_QTY = max(1, round(
    1.4136
    + 0.1447  * BVL_LTH
    + 0.01236 * CUT_LTH
    + Normal(0, 2.812^2)))
```

### 6.7 W/O PTLST_QTY

```text
PTLST_QTY = max(1, round(
  5.441
  + 0.2156 * CUT_LTH
  - 0.0005 * LTH
  - 0.1079 * MARK_LTH
  + Normal(0, 7.602^2)))
```

### 6.8 W/O TACT_TIME

```text
TACT_TIME = 0.3037 * CUT_LTH
          + 0.1325 * MARK_LTH
          + 0.4790 * THK
          + 0.3840 * PTLST_QTY
```

- 단위는 분이다.
- NP/FN/FL/NC 전 계열이 같은 Case 6 식을 사용한다.
- `BV_QTY`는 현재 TACT_TIME 식에 포함하지 않는다.
- actual 착수·종료 elapsed time을 TACT_TIME으로 대체하지 않는다.

## 7. BTH 공통 수식과 계열별 값

전 계열은 같은 로그선형식 구조를 사용한다.

```text
ln(BTH) = b0
        + b1 * ln(1 + LTH)
        + b2 * ln(1 + THK)
        + b3 * ln(1 + MARK_LTH)
        + b4 * ln(1 + CUT_LTH)
        + b5 * ln(1 + BVL_LTH)
        + b6 * ln(1 + BV_QTY)
        + b7 * ln(1 + PTLST_QTY)
        + Normal(0, residual_sd^2)
```

| 계열 | b0 | LTH | THK | MARK | CUT | BVL | BV_QTY | PTLST | 잔차 SD | R^2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NP | 6.677550 | -0.171541 | 0.239296 | 0.187978 | 0.424088 | -0.014660 | -0.013820 | -0.162696 | 0.289263 | 0.665831 |
| FN | 8.635531 | -0.214857 | -0.116671 | 0.177521 | 0.329709 | 0.038007 | 0.028260 | -0.354113 | 0.230223 | 0.617106 |
| FL | 9.421209 | -0.349194 | 0.084343 | 0.003569 | 0.548270 | -0.046789 | 0.080551 | -0.403524 | 0.151003 | 0.252072 |
| NC | 8.367374 | -0.274651 | 0.002204 | 0.162329 | 0.485548 | 0.014519 | -0.030201 | -0.306288 | 0.182775 | 0.356277 |

연속 예측값은 해당 계열에서 실제로 관측된 BTH 규격 중 가장 가까운 값으로 스냅한다.

| 계열 | 관측 BTH 최소 | 관측 BTH 최대 | 관측 규격 수 |
|---|---:|---:|---:|
| NP | 60 | 4,395 | 765 |
| FN | 1,000 | 4,400 | 146 |
| FL | 1,000 | 4,395 | 396 |
| NC | 1,000 | 4,375 | 233 |

## 8. STL_QTY 생성

`STL_QTY`는 회귀식으로 만들지 않는다. 다음 7개 특성을 표준화해 같은 계열 실적에서 가장 가까운 W/O 32개를 찾는다.

```text
LTH, THK, MARK_LTH, CUT_LTH, BVL_LTH, BV_QTY, PTLST_QTY
```

```text
P(STL_QTY = c)
= 0.75 * 근접 32개 안의 class c 비율
+ 0.25 * 해당 계열 전체의 class c 비율
```

| 계열 | STL_QTY class와 실적 주변확률 |
|---|---|
| NP | 0: 8.905506%, 1: 91.080897%, 2: 0.013596% |
| FN | 0: 1.201201%, 1: 98.798799% |
| FL | 0: 0.198864%, 1: 99.801136% |
| NC | 1: 76.484018%, 2: 23.515982% |

생성 전체 행에서 계열별 주변 class 수를 먼저 정수로 맞추고, 조건부 확률이 높은 W/O에 각 class를 배정한다. 따라서 `STL_QTY`와 `WO_QTY`는 독립된 의미를 유지한다.

### STL_QTY 조건부 거리 계산용 계열별 표준화 값

각 특성은 `(x-평균)/표준편차`로 표준화한 뒤 최근접 W/O를 찾는다.

| 계열 | 특성 | 평균 | 표준편차 |
|---|---|---:|---:|
| NP | LTH | 8,685.022434 | 5,222.845058 |
| NP | THK | 14.314004 | 4.247494 |
| NP | MARK_LTH | 32.764823 | 30.453510 |
| NP | CUT_LTH | 63.561027 | 53.071298 |
| NP | BVL_LTH | 1.715514 | 5.322331 |
| NP | BV_QTY | 0.801903 | 2.077439 |
| NP | PTLST_QTY | 9.638885 | 12.829578 |
| FN | LTH | 11,806.261261 | 5,650.470434 |
| FN | THK | 17.767267 | 5.623501 |
| FN | MARK_LTH | 72.355685 | 46.583972 |
| FN | CUT_LTH | 38.881613 | 19.293846 |
| FN | BVL_LTH | 5.774420 | 11.079340 |
| FN | BV_QTY | 0.723724 | 1.288180 |
| FN | PTLST_QTY | 1.930931 | 1.549398 |
| FL | LTH | 15,840.103693 | 4,531.041246 |
| FL | THK | 14.713494 | 3.506399 |
| FL | MARK_LTH | 23.135152 | 20.954410 |
| FL | CUT_LTH | 42.232540 | 12.162747 |
| FL | BVL_LTH | 2.117516 | 5.674319 |
| FL | BV_QTY | 0.292045 | 0.651436 |
| FL | PTLST_QTY | 1.139489 | 0.486322 |
| NC | LTH | 10,822.728311 | 4,107.438448 |
| NC | THK | 20.168950 | 5.016297 |
| NC | MARK_LTH | 115.610701 | 46.521754 |
| NC | CUT_LTH | 36.755452 | 14.625010 |
| NC | BVL_LTH | 3.157308 | 6.355587 |
| NC | BV_QTY | 0.894977 | 1.696874 |
| NC | PTLST_QTY | 2.691781 | 1.993935 |

## 9. FN/FL/NC 블록 수식과 계열별 값

FN/FL/NC는 같은 블록 사슬을 사용한다.

```text
block LTH ~ Normal(length_mean, length_sd^2), observed range로 clip
block MARK = max(mark_a * LTH + mark_b + Normal(0, mark_sd^2), mark_min)
block CUT  = max(cut_a * MARK + cut_b + Normal(0, cut_sd^2), cut_min)
block WO_QTY = round(wo_a * CUT + wo_b + Normal(0, wo_sd^2))
block THK = nearest_spec(thk_a * ln(WO_QTY) + thk_b + Normal(0, thk_sd^2))
```

MIXED 경로에서는 물리 블록 전체 W/O 수와 계열별 W/O 수가 앞 단계에서 이미 확정되므로, 아래 `wo_a/wo_b/wo_sd`는 standalone 계열 생성 분석에는 사용되지만 MIXED episode의 count를 다시 결정하지 않는다.

### 9.1 블록 LTH 분포

| 계열 | 평균 | 표준편차 | 최소 | 최대 |
|---|---:|---:|---:|---:|
| FN | 11,840.000 | 5,742.044 | 2,480 | 21,605 |
| FL | 17,237.606 | 3,283.961 | 6,950 | 22,025 |
| NC | 12,436.471 | 4,473.716 | 3,080 | 22,095 |

### 9.2 블록 MARK_LTH

| 계열 | mark_a | mark_b | 잔차 SD | 최솟값 |
|---|---:|---:|---:|---:|
| FN | 0.005728 | 9.266571 | 39.913516 | 0.000 |
| FL | 0.001042 | 24.025293 | 26.595654 | 1.280 |
| NC | 0.006686 | 60.163050 | 44.195407 | 18.959 |

### 9.3 블록 CUT_LTH

| 계열 | cut_a | cut_b | 잔차 SD | 최솟값 |
|---|---:|---:|---:|---:|
| FN | 0.652119 | 28.197934 | 45.121935 | 3.222 |
| FL | 0.800574 | 282.686815 | 154.938746 | 46.539 |
| NC | 0.555702 | 25.584278 | 63.715677 | 9.499 |

### 9.4 standalone 계열 WO_QTY 식

| 계열 | wo_a | wo_b | 잔차 SD | 실적 지원범위 |
|---|---:|---:|---:|---:|
| FN | 0.019954 | 0.452420 | 0.740439 | 1~8 |
| FL | 0.019124 | 1.440523 | 1.583001 | 1~19 |
| NC | 0.022763 | 0.467545 | 0.857656 | 1~12 |

### 9.5 블록 THK 식

| 계열 | thk_a | thk_b | 잔차 SD | 관측 규격 범위 | 규격 수 |
|---|---:|---:|---:|---:|---:|
| FN | 1.251693 | 18.024112 | 6.334998 | 10~37 | 36 |
| FL | 4.627933 | 7.766765 | 4.509093 | 10~35 | 33 |
| NC | 2.478202 | 18.363635 | 5.065698 | 12~36 | 30 |

## 10. FN/FL/NC W/O LTH 생성

계열별 W/O 길이는 단순 독립 정규분포가 아니라 블록 안에서 내림차순 멱감쇠 형태로 생성한다.

```text
sigma_A(N) = max(0.04, 0.476 / sqrt(N) - 0.050)

a_N = clip(
  a_inf - c/N + sigma_A(N) * Z_A,
  0.05,
  1.4)

relative_LTH(u) = 1 - a_N * u^b
W/O LTH = block_LTH * relative_LTH(u) * (1 + Normal(0, 0.0072^2))
```

- 인접 W/O 길이가 실적에서 상대 5% 이내로 비슷했던 확률을 위치 10구간으로 적합한다.
- 같은 그룹으로 병합된 W/O는 유사한 길이를 갖는다.
- 길이는 내림차순으로 정렬한다.
- 첫 W/O의 길이는 항상 블록 LTH와 같다.
- 최솟값은 계열별 실적 하한으로 제한한다.

| 계열 | a_inf | c | b | LTH 하한 | Z_A SkewNormal `(shape, loc, scale)` |
|---|---:|---:|---:|---:|---|
| FN | 0.940419 | 1.676258 | 2.240926 | 3,000 | `(-34417939.217062, 1.203207, 1.564473)` |
| FL | 0.573395 | 0.765114 | 2.795475 | 3,527 | `(-2.520037, 1.112229, 1.495678)` |
| NC | 0.787173 | 1.511223 | 1.409689 | 3,721 | `(3.028387, -1.155709, 1.528289)` |

FN의 매우 큰 음수 shape는 SciPy가 실적 잔차에 적합한 수치 파라미터이며 물리계수가 아니다. 값을 임의로 줄이지 않고 현재 적합 결과를 그대로 사용한다.

### 인접 길이 병합 확률

| 위치 구간 | FN | FL | NC |
|---:|---:|---:|---:|
| 0 | 0.656250 | 0.953271 | 0.424779 |
| 1 | 1.000000 | 0.865306 | 0.375000 |
| 2 | 0.400000 | 0.897143 | 0.470588 |
| 3 | 0.692308 | 0.849817 | 0.394737 |
| 4 | 0.666667 | 0.869863 | 0.500000 |
| 5 | 0.400000 | 0.784314 | 0.684211 |
| 6 | 0.733333 | 0.822102 | 0.560976 |
| 7 | 0.250000 | 0.742063 | 0.642857 |
| 8 | 0.800000 | 0.693878 | 0.545455 |
| 9 | 사용 안 함 | 0.602410 | 0.000000 |

FN은 관측 W/O 수 범위가 1~8이므로 위치 구간 9에 도달하지 않는다. 이는 결측값을 대체한 것이 아니라 생성 가능한 위치 자체가 없는 경우다.

## 11. FN/FL/NC W/O THK 생성

### 11.1 길이비 조건부 두께 분포

```text
r = W/O LTH / block LTH

mu(r) = m0 + m1 * exp(-m2 * r)
sigma(r) = s2 * r^2 + s1 * r + s0

THK_base = mu(r) + sigma(r) * Z_T
Z_T ~ SkewNormal(shape_T, loc_T, scale_T)
```

| 계열 | `(m0, m1, m2)` | `(s2, s1, s0)` | Z_T `(shape, loc, scale)` |
|---|---|---|---|
| FN | `(-4488.518471, 4509.235831, 0.000556)` | `(-23.506423, 28.988680, -1.975992)` | `(3.448126, -1.736963, 2.052635)` |
| FL | `(-3631.933606, 3648.181948, 0.000311)` | `(-12.032228, 12.301577, 1.581089)` | `(4.600493, -2.125902, 2.398305)` |
| NC | `(-5728.020549, 5751.114004, 0.000462)` | `(-22.979827, 30.559637, -4.473920)` | `(2.834054, -1.586742, 1.818353)` |

### 11.2 블록 내부 LTH-THK 상관 구조

```text
logLr = ln(max W/O LTH / min W/O LTH)

rho_mean  = c0 + c1*N + c2*logLr
rho_scale = max(0.12, d0 + d1*N + d2*logLr)
rho       = clip(rho_mean + rho_scale*Z_rho, -0.95, 0.70)
```

| 계열 | `(c0, c1, c2)` | `(d0, d1, d2)` | Z_rho `(shape, loc, scale)` |
|---|---|---|---|
| FN | `(-2.198741, 0.405028, 0.006300)` | `(0.039993, 0.132920, -0.215877)` | `(6059402.684444, -1.459546, 1.731630)` |
| FL | `(-0.015969, 0.010061, -0.391653)` | `(1.061675, -0.040050, -0.215534)` | `(1.316510, -0.812229, 1.257378)` |
| NC | `(-0.386660, 0.113011, -0.831435)` | `(0.580592, 0.027170, -0.428076)` | `(34247782.035645, -1.273457, 1.582607)` |

생성한 연속 두께는 계열별 실적 두께 규격으로 빈도가중 스냅하고, 적어도 한 W/O의 THK가 block THK와 같도록 보정한다.

| 계열 | W/O THK 규격 범위 | 규격 수 |
|---|---:|---:|
| FN | 10~37 | 33 |
| FL | 10~35 | 36 |
| NC | 12~36 | 34 |

## 12. FN/FL/NC W/O MARK_LTH와 CUT_LTH

기본 모드는 Spearman 구조를 보존하는 log-log 식이다.

```text
MARK_raw = exp(mark_a * ln(LTH) + mark_b + Normal(0, mark_sd^2))
CUT_raw  = exp(cut_a  * ln(LTH) + cut_b  + Normal(0, cut_sd^2))
```

| 계열 | MARK a | MARK b | MARK log 잔차 SD | MARK 양수 하한 |
|---|---:|---:|---:|---:|
| FN | 1.338244 | -8.438838 | 0.825145 | 0.260 |
| FL | 0.505372 | -2.179584 | 1.058902 | 0.121 |
| NC | 0.568624 | -0.571036 | 0.344562 | 18.959 |

| 계열 | CUT a | CUT b | CUT log 잔차 SD | CUT 양수 하한 |
|---|---:|---:|---:|---:|
| FN | 0.835168 | -4.206296 | 0.395146 | 1.831 |
| FL | 0.768790 | -3.696039 | 0.166113 | 4.093 |
| NC | 0.514191 | -1.207116 | 0.320920 | 8.137 |

생성한 상대값은 다음 block-series 계약에 맞게 조정한다.

```text
sum_i MARK_LTH_i = 생성 block MARK_LTH
sum_i CUT_LTH_i  = 생성 block CUT_LTH
```

값의 순서와 양수 하한을 보존하면서 맞추며, 실패 시 임의 균등분배로 대체하지 않는다.

## 13. FN/FL/NC 베벨 발생과 BVL_LTH

### 13.1 베벨 발생 여부

```text
z = (THK - thickness_mean) / thickness_sd
p(bevel) = sigmoid(h0 + h1*z)
has_bevel ~ Bernoulli(p(bevel))
```

| 계열 | h0 | h1 | 두께 평균 | 두께 SD |
|---|---:|---:|---:|---:|
| FN | -0.623649 | 1.732600 | 17.767267 | 5.623501 |
| FL | -1.603171 | 1.560465 | 14.713494 | 3.506399 |
| NC | -0.929970 | 1.663930 | 20.168950 | 5.016297 |

### 13.2 BVL_LTH

베벨이 발생한 W/O만 다음 식을 사용하고, 발생하지 않으면 `BVL_LTH=0`이다.

```text
BVL_LTH = exp(
  c0
  + c_thk  * ln(THK)
  + c_lth  * ln(LTH)
  + c_mark * ln(max(MARK_LTH, 1e-6))
  + Normal(0, residual_sd^2))
```

| 계열 | c0 | THK | LTH | MARK | log 잔차 SD |
|---|---:|---:|---:|---:|---:|
| FN | -8.133596 | 1.211961 | 0.751596 | 0.019585 | 0.716415 |
| FL | -2.996598 | 1.821283 | -0.112279 | 0.224007 | 1.040872 |
| NC | -1.788208 | 1.342136 | -0.259109 | 0.403592 | 0.732923 |

## 14. FN/FL/NC BV_QTY

```text
if BVL_LTH = 0:
  BV_QTY = 0
else:
  BV_QTY = max(1, round(exp(
    q0
    + q_bvl * ln(max(BVL_LTH, 0.1))
    + q_cut * ln(CUT_LTH)
    + Normal(0, residual_sd^2))))
```

| 계열 | q0 | BVL | CUT | log 잔차 SD |
|---|---:|---:|---:|---:|
| FN | 0.912259 | 0.085542 | -0.184712 | 0.553606 |
| FL | -0.058736 | 0.104506 | 0.013899 | 0.337995 |
| NC | -2.372378 | 0.266336 | 0.751306 | 0.433183 |

## 15. FN/FL/NC PTLST_QTY

```text
PTLST_QTY = max(1, round(exp(
  p0
  + p_cut  * ln(CUT_LTH)
  + p_lth  * ln(LTH)
  + p_mark * ln(max(MARK_LTH, 1e-6))
  + Normal(0, residual_sd^2))))
```

| 계열 | p0 | CUT | LTH | MARK | log 잔차 SD |
|---|---:|---:|---:|---:|---:|
| FN | 5.737890 | 1.137075 | -0.895994 | -0.252232 | 0.358046 |
| FL | 4.365288 | 0.686823 | -0.708807 | -0.000892 | 0.220602 |
| NC | 3.844763 | 1.323386 | -0.828797 | -0.015569 | 0.378279 |

## 16. 특성별 최종 생성 기준

| 특성 | W/O 생성 기준 | block-series 생성 기준 |
|---|---|---|
| `PROJ_NO` | 합성 물리 블록 identity | W/O identity에서 유지 |
| `BLK_NO` | 합성 물리 블록 identity | W/O identity에서 유지 |
| `GYEL` | 선택된 NP/FN/FL/NC 계열 | 같은 계열끼리 집계 |
| `WK_ORD_NO` | W/O마다 유일하게 생성 | 블록에는 없음 |
| `LTH` | NP 고정 멱감쇠 또는 계열별 멱감쇠 | 최댓값 |
| `BTH` | 계열별 로그선형식 후 관측 규격 스냅 | 최댓값 |
| `THK` | 길이비 조건부 분포 후 관측 규격 스냅 | 최댓값 |
| `CUT_LTH` | 길이 조건부 식 후 block 목표에 연결 | 합계 |
| `MARK_LTH` | 길이 조건부 식 후 block 목표에 연결 | 최댓값 |
| `BVL_LTH` | 베벨 허들 + THK/LTH/MARK 조건부 식 | 합계 |
| `BV_QTY` | BVL/CUT 조건부 식 | 합계 |
| `PTLST_QTY` | CUT/LTH/MARK 조건부 식 | 합계 |
| `STL_QTY` | 계열별 조건부 범주확률 | 합계 |
| `TACT_TIME` | 전 계열 공통 Case 6 식 | 최댓값 |
| `WO_QTY` | 별도 W/O 특성이 아니라 W/O 행 수 | W/O 행 수 |

## 17. Phase 1과 Phase 2에서 사용하는 방식

### Phase 1

```text
episode seed로 MIXED 데이터 생성
-> W/O를 block-series로 집계
-> block-series별 WO_QTY/CUT_LTH/BV_QTY를 Phase 1 state에 입력
-> NP block-series의 W/O CUT_LTH 합이 1,000 이상이면 Bay 22/23으로 hard mask
-> NP/NC block-series를 NP_NC(22/23/24) 서브문제로 분리
-> FN/FL block-series를 FN_FL(25/trans) 서브문제로 분리
-> 각 서브문제에서 (block-series, Bay) pair를 순차 선택
-> 각 teacher sequence로 공유 policy를 별도 CE update
-> 두 assignment를 하나의 Phase 1 plan으로 병합
```

이 분할은 학습데이터를 다시 생성하거나 identity를 바꾸는 과정이 아니다. 하나의
부모 MIXED episode에서 서로 설비를 공유하지 않는 W/O만 학습 시점에 분할한다.
한 자원군이 없으면 해당 서브문제와 metric row는 생성하지 않는다.

### Phase 2

```text
같은 MIXED 데이터 생성
-> Phase 1 휴리스틱 또는 frozen agent로 Bay 결정
-> Bay별 W/O subproblem 생성
-> 환경이 다음 가용 설비와 목표 batch 크기 결정
-> SELECT_WO만 학습
```

Phase 2 teacher는 `hard violation -> makespan -> Bay 내부 CUT_LTH gap 합 -> W/O 수
gap 합 -> BV_QTY gap 합 -> 점유시간 gap 합`의 사전식 score로 선택한다. Bay별
self-labeling에서는 해당 Bay의 gap을 사용하고, combined validation/full-flow에서는
active Bay의 gap을 합산한다.

Phase 1과 Phase 2는 같은 생성 함수와 seed 계약을 사용한다. 따라서 같은 seed의 `PROJ_NO`, `BLK_NO`, `WK_ORD_NO`, 계열 조합과 W/O 특성이 서로 일치한다.

## 18. 재현성과 난수 구조

하나의 episode seed를 `SeedSequence`로 분리한다.

```text
block seed stream
series allocation stream
physical block stream
NP stream
FN stream
FL stream
NC stream
```

- 같은 seed를 다시 사용하면 같은 DataFrame과 identity가 생성된다.
- 계열마다 별도 stream을 사용하므로 같은 난수열 때문에 가짜 완전상관이 생기지 않는다.
- 물리 블록 수 또는 seed가 바뀌면 새로운 문제가 생성된다.

## 19. 고정 JSON profile 계약

### 19.1 profile의 의미

`multi_series_generation_profile.json`은 매 학습 실행 때 다시 계산하는 임시 cache가 아니다. 검증된 실적에서 산출한 생성 모형의 고정 스냅샷이다.

```text
실적 Excel
-> 오프라인 profile 산출·검증
-> multi_series_generation_profile.json 고정
-> Phase 1/2 학습은 JSON만 로드
-> episode마다 새 난수로 합성데이터 생성
```

profile에는 다음 정보가 들어 있다.

| 구분 | 저장 내용 |
|---|---|
| 계약 식별 | `schema=multi_series_generation_profile_v1` |
| 원천 추적 | 원천 Excel 상대경로, SHA256, 파일 크기 |
| NP 고정식 | 발표자료 계수, 오차항 범위, THK 규격, BTH/STL 난수 stream 계약 |
| 물리 블록 공동분포 | 1,044개 물리 블록의 계열 조합, 조합 확률, 조건부 계열별 W/O count vector |
| NP 보조 profile | BTH 로그선형식과 잔차, STL_QTY 조건부 입력·범주확률 |
| FN/FL/NC 입력분포 | block chain, W/O 수 범위, 규격 지원값, 조건부 표본 배열 |
| FN/FL/NC 생성식 | 계열별 계수·절편, merge 확률, 잔차분포 파라미터 |

NP 수식 실행 로직은 `report_formula_data_generator.py`에 유지한다. JSON에도 NP 고정 상수를 함께 기록하고, loader가 코드의 고정 계약과 정확히 같은지 검사한다. 둘이 다르면 학습을 시작하지 않는다.

### 19.2 profile 재생성

원천 Excel이 변경되거나 계열별 적합 방법을 의도적으로 바꾼 경우에만 다음 명령을 실행한다.

```bash
python scripts/build_multi_series_generation_profile.py
```

경로를 명시하려면 다음과 같이 실행한다.

```bash
python scripts/build_multi_series_generation_profile.py \
  --wo-xlsx "변경사항/절단WO_데이터.xlsx" \
  --block-xlsx "변경사항/절단블록_데이터.xlsx" \
  --output "Utils/data/multi_series_generation_profile.json"
```

이 명령만 Excel을 읽어 NP 보조 profile, 물리 블록 공동분포, FN/FL/NC 계수를 다시 산출한다. 생성된 JSON과 source SHA256 변경분은 검토 후 학습에 사용한다. 학습 명령이 원천 Excel 변경을 감지해 JSON을 몰래 다시 만드는 동작은 없다.

### 19.3 no-fallback 규칙

다음 조건에서는 기본값이나 Excel 재적합으로 대체하지 않고 원인을 출력한 뒤 즉시 실패한다.

- profile 파일 누락
- schema 불일치
- NP 고정 수식 계약 불일치
- NP/FN/FL/NC 필수 profile 누락
- source SHA256 metadata 누락 또는 형식 오류
- 직렬화 배열의 길이·shape 불일치

따라서 학습 실행의 생성 파라미터는 실행 시점이나 Excel 로딩 상태에 따라 달라지지 않는다.

## 20. 현재 모델 가정과 해석 범위

- NP 수식은 발표자료 고정값이다.
- FN/FL/NC 계수는 profile 생성 시점의 실적에서 적합한 고정값이다. 원천 Excel이 바뀌어도 profile을 명시적으로 재생성하기 전에는 학습값이 바뀌지 않는다.
- FN/FL/NC 실적에는 별도 확정 TACT_TIME 식이 없어 현재 NP Case 6 식을 공통 적용한다.
- FL의 BTH 설명력은 `R^2=0.252072`로 낮다. 이를 숨기기 위해 잔차를 축소하거나 값을 조작하지 않는다.
- 희소 계열 조합은 실적 표본이 적으므로 모든 계열 간 상관관계를 완전히 재현한다고 주장하지 않는다.
- 현재 우선 보장하는 것은 인과 생성 순서, 실적 계열 조합분포, W/O 수 identity, block-W/O 집계 identity다.

## 21. 실행 중 확인되는 핵심 로그

```text
[CHECK][multi_series_formula_data_generator._load_multi_series_generation_profile]
[VALIDATION][multi_series_formula_data_generator.generate_multi_series_formula_data]
[VALIDATION][phase1_episode_dataset.build_phase1_episode_jobs]
```

정상 생성 완료 조건은 다음과 같다.

```text
physical_blocks = 요청한 물리 블록 수
series_blocks >= physical_blocks
wos = 모든 물리 블록의 FORMULA_WO_QTY 합
validation passed=true
```

학습 로그에 `ShipyardGenerator.fit`이나 Excel 로딩 로그가 반복되면 현재 고정 profile 계약을 우회한 것이므로 정상 경로가 아니다.

## 22. 핵심 구분

| 혼동하기 쉬운 항목 | 정확한 의미 |
|---|---|
| `WO_QTY` | block-series를 구성하는 W/O 행 수 |
| `STL_QTY` | 각 W/O의 강재수량, block에서는 합계 |
| `BVL_LTH` | 베벨 길이 |
| `BV_QTY` | 베벨 수량 |
| `TACT_TIME` | W/O의 절단 처리시간, 단위 분 |
| 물리 블록 | `(PROJ_NO, BLK_NO)` |
| block-series | `(PROJ_NO, GYEL, BLK_NO)` |
| 생성 block 수 | 물리 블록 수 |
| Phase 1 action 수의 block | block-series 수 |
