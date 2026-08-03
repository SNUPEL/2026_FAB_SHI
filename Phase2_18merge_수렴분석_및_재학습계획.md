# Phase 2 (18th-merge) 수렴 분석 및 재학습 계획 후보

> 작성 시점 기준 실험: `output/phase2_18merge_tempanneal_w8` (episodes=20000 완료).
> 최종 채택 모델: `phase2_best.pt` = **ep15000** (parent best-rate 50%, mean_rank 1.5, `selection_key=[25,-25,1.5]`).
> 이 문서는 "수렴 실패 원인 진단"과 "재학습 계획 후보(단계별)"를 기록한다. 현재는 **여기서 학습 종료** 후 결과 분석 단계이며, 아래 계획은 다음 실험 착수 시 참조용이다.

---

## 1. 18th-merge 실험 요약

| 항목 | 값 |
|---|---|
| 학습 경로 | MIXED, self-labeling teacher-best CE (tie-aware) |
| episodes / lr | 20000 / **0.001 고정 (decay 없음)** |
| temperature | 1.0 → 0.3, **anneal_episodes=20000 (전 구간)** |
| rollout_samples | 64 (greedy 1 + sample 63), candidate_workers 8 |
| 학습 문제 크기 | **`--min-blocks 10 --max-blocks 30`** (job_count 34~334, 평균 141) |
| validation grid | **`--validation-min/max-blocks 10..100` gap 10** × distribution_type 5 = 50문제 (job_count 78~785) |
| heuristics(teacher 경쟁) | lpt_batch, long_cut_batch, best_fit_lth, long_bevel_batch |

**holdout best-rate 추이 (parent, proposed=best-of-64):**
`8,18,22,22,22,26,28,30,38,20,44,24,46,34,50,30,46,44,26,22` (ep1k~20k)
→ 상승 후 **44~50% 대역에서 큰 진폭으로 진동, 마지막 ep20000은 저점(22%)에서 종료. 단조 수렴 없음.**

---

## 2. 수렴 실패 원인 진단 (데이터 근거)

### 진단 A — 【지배적】 train/validation 문제 크기 불일치 (OOD)
학습은 **10~30 블록(job≤334)**만 보는데, validation grid는 **10~100 블록(job≤785)**까지 평가한다.
best-rate가 문제 크기에 따라 단조 붕괴 (후반 9개 checkpoint ep12k~20k 평균):

| block_count | job_count | best-rate | 비고 |
|---:|---:|---:|---|
| 10 | ~78 | **100%** | in-dist |
| 20 | ~148 | **87%** | in-dist (학습 중앙값 부근) |
| 30 | ~186 | 42% | in-dist 경계 |
| 40 | ~288 | 24% | 학습 최대 부근 |
| 50 | ~351 | 47% | edge |
| 60 | ~408 | 38% | OOD |
| 70 | ~500 | 20% | OOD |
| 80 | ~563 | **0%** | far OOD |
| 90 | ~618 | **0%** | far OOD |
| 100 | ~713 | **0%** | far OOD |

**집계:** in-dist(blocks 10-30) **76%** vs OOD(blocks 60-100) **12%** (blocks 80-100 = 0% 일관).
→ 전체 best-rate(~36~50%)는 **학습에서 본 적 없는 대형 문제가 절반을 끌어내린 결과**. 대형 문제에서는 `lpt_batch`가 5/5 승리(makespan 지표에서 구조적으로 강함).

#### ✅ 확인 실험 — 학습분포 전용 validation (target-matching 없이 blocks 10~30 균등, held-out seed 84문제, `phase2_best.pt` 가중치 동결)
`output/phase2_indist_eval` (driver: `scratchpad/indist_eval_driver.py`, grid builder monkeypatch + `lr=1e-30` 동결). job_count 57~251(평균 140) = 학습분포와 일치.

| 지표 | 일반화 grid (blocks 10-100) | **In-distribution (blocks 10-30)** |
|---|---:|---:|
| 전체 best-rate | 50% | **88.1%** |
| mean_rank | 1.50 | **1.12** |
| NP_NC pool | 53% | 88% |
| FN_FL pool | 38% | **94%** |
| Bay25 (최약이었던) | 28% | **94%** |

→ **모델은 학습분포에서 88% 승률로 강함.** 일반화 grid의 낮은 수치(특히 Bay25 28%, OOD 0%)는 **전적으로 OOD 크기·분포에서의 측정 아티팩트**였다. Bay25의 "약점"도 OOD 아티팩트였을 뿐, in-dist에서는 오히려 최강(94%). blocks별로 0%가 전혀 없고 대부분 75~100%.

### 진단 B — subproblem(pool) 성능 불균형
per-Bay best-rate (ep15000, bay_history):

| pool | Bay | best-rate | mean_rank |
|---|---|---:|---:|
| NP_NC | 22 | 50% | 1.50 |
| NP_NC | 23 | 56% | 1.46 |
| NP_NC | 24 | 52% | 1.48 |
| **FN_FL** | **25** | **28%** | **1.72** |
| FN_FL | trans | 48% | 1.52 |
| **pool 집계** | NP_NC | **53%** | 1.48 |
| **pool 집계** | FN_FL | **38%** | 1.62 |

→ **FN_FL(특히 Bay25)가 약함.** 학습 로그에서도 Bay25/trans는 `lpt_batch`가 teacher-best인 빈도가 높다.

### 진단 C — 최적화 불안정 (진동의 직접 원인)
- **LR 0.001 고정, decay 스케줄러 없음** → 좋은 영역 주변을 계속 튐(진폭이 끝까지 일정).
- **temperature anneal이 전 구간(20000ep)** → T가 하한 0.3에 **맨 끝에서야** 도달(ep15k T≈0.40, ep18k T≈0.34). **저온 consolidation 구간이 사실상 없어** greedy takeover(11%→최대 31%)가 자리잡지 못함.

---

## 3. 재학습 계획 후보

> 공통 전제: 새 실험은 **현재 학습 프로세스 완전 종료(GPU 해제) 후** 착수. baseline 비교는 `output/_baseline_260728/`(EXP-B) 및 본 실험 `phase2_best.pt`(ep15000) 유지.
> 명령 예시는 ep20000 실행의 기존 플래그를 상속하고 **바뀌는 플래그만** 표기한다.

### Plan A — 【최우선 권장】 train/eval 분포 정합 (진단 A 해소)
문제 크기 불일치가 지배적이므로 **가장 먼저** 해결한다. 두 방향 중 택1은 **실제 배포 대상 문제 크기**에 달려 있다(→ §4 의사결정 필요).

- **A1. 학습 범위를 validation에 맞춰 확장** — 대형 문제까지 성능을 원할 때
  - 단계
    1. `--max-blocks 100`(필요시 `--max-wo-count` 상향)로 학습 문제를 10~100 블록으로 확대.
    2. 대형 문제는 episode당 연산량↑ → episodes 또는 candidate_workers 재산정(소요시간 재추정).
    3. validation grid는 10~100 유지(정합됨).
  - 기대: OOD 0% 구간 해소. 단, 난이도↑로 전체 best-rate 상승은 완만할 수 있음.
  - 비용: 높음(대형 episode 느림).
  ```
  ... --min-blocks 10 --max-blocks 100 --max-wo-count <상향> \
      --validation-min-blocks 10 --validation-max-blocks 100 --validation-block-gap 10
  ```

- **A2. validation을 학습 범위로 축소** — 배포 대상이 소형(≤30~40블록)일 때
  - 단계
    1. `--validation-max-blocks 40`(또는 30)으로 grid를 학습 분포에 맞춤.
    2. best-rate가 in-dist 실성능(76~100%)을 반영하게 되어 checkpoint 선택·비교가 정직해짐.
    3. 필요시 distribution_type 수(`--validation-episodes`)를 늘려 소형 구간 표본↑.
  - 기대: best-rate 지표가 실제 성능을 반영(과소평가 제거). 능력 자체가 느는 건 아님.
  - 비용: 낮음(측정 정합만).
  ```
  ... --validation-min-blocks 10 --validation-max-blocks 40 --validation-block-gap 10
  ```

> 권장: **A1과 A2는 배타적이지 않다.** 배포 대상이 넓으면 A1(학습 확장)+grid 유지, 소형 위주면 A2(측정 정합). 우선 **A2로 지표를 바로잡아** 현재 모델의 실제 in-dist 성능(76~100%)을 확정한 뒤, 대형 문제 수요가 있으면 A1을 추가하는 2단계가 안전하다.

### Plan B — 최적화 안정화 (진단 C 해소)
진동을 줄이고 저온 consolidation을 확보한다. Plan A와 **병행 적용 권장**.

- **B1. LR decay 도입** (소규모 코드 수정 필요 — 전용 스케줄러 플래그 없음)
  1. optimizer에 스케줄러 추가(cosine 또는 step): 0.001 → ~1e-4.
  2. 대안(코드 무수정): 2단계 학습 — 1차 통상 학습 후 `--resume-checkpoint`로 `--lr 0.0001` 재개.
- **B2. temperature anneal 재설계**
  1. `--temperature-anneal-episodes`를 전체의 **50~60%**(예: 20000ep이면 12000)로 단축 → 뒤 8000ep는 T=0.3 고정(저온 consolidation).
  2. 필요시 `--temperature-min 0.2`로 하한을 더 낮춰 greedy takeover 촉진.
  ```
  ... --temperature 1.0 --temperature-min 0.3 --temperature-anneal-episodes 12000
  ```
- **B3.(선택) weight EMA / checkpoint averaging** — 잔여 진동 완화(코드 수정 필요).

### Plan C — subproblem 타깃 개선 (진단 B 해소)
FN_FL(특히 Bay25) 약점 규명 후 대응.
1. Bay25가 왜 약한지 분해: 배치 제약(LTH합≤55000, WO 1~3개), family_eligibility, job_count(대형)·machine=2 구조 중 무엇이 lpt_batch 우위를 만드는지.
2. 대응 후보: pool별 loss 가중, FN_FL 표본 비중↑, 또는 pool별 정책 분리 검토(현재는 공유 정책 2회 순차 CE).
3. 먼저 **진단만** 수행(저비용), 개선은 Plan A/B 이후.

### Plan D — 저비용 fine-tune (빠른 검증용)
근본 수정 없이 consolidation 가설만 싸게 확인.
1. `phase2_best.pt`(ep15000)에서 `--resume-checkpoint`.
2. `--lr 0.0001`, `--temperature 0.3 --temperature-min 0.3`(고정), `--episodes 5000`.
3. in-dist(A2) 지표로 best-rate 안정화 여부 확인. 개선되면 Plan B 정식화, 아니면 Plan A로.
```
... --resume-checkpoint output/phase2_18merge_tempanneal_w8/phase2_best.pt \
    --lr 0.0001 --temperature 0.3 --temperature-min 0.3 --episodes 5000 \
    --validation-max-blocks 40
```

---

## 4. 권장 우선순위 & 의사결정 필요 사항

**권장 실행 순서**
1. **(측정 정합) A2** — validation을 학습 범위(≤40블록)로 축소해 현재 모델 실성능(≈76~100%)을 지표로 확정. *코드 무수정, 저비용.*
2. **(빠른 검증) D** — best 모델 fine-tune로 진동 완화 여부 확인.
3. **(정식 재학습) B1+B2 (+ 필요시 A1)** — LR decay + anneal 단축을 넣어 안정 수렴 목표. 대형 문제 수요가 있으면 A1로 학습 범위 확장.
4. **(심화) C** — FN_FL/Bay25 약점 개선.

**사용자 결정이 필요한 항목**
- ❓ **배포/평가 대상 문제 크기**: 실제 절단 shop이 다루는 블록 규모가 소형(≤30~40)인가, 대형(~100)까지 포함인가? → A2 단독 vs A1 병행을 가른다.
- ❓ **재학습 비용 허용치**: 코드 수정(B1 스케줄러, B3 EMA)까지 갈지, 무수정 범위(A2/D/B2)로 제한할지.
- ❓ best-rate 목표치(예: in-dist ≥90%) 및 heuristic 대비 "무승부(mean_rank≈1.5)"를 승리로 볼지 여부.

---

## 6. 구현 완료 — main/generalization validation 상시화 (2026-08-03)

이제 학습이 **두 개의 고정 validation**을 매 checkpoint 동시 수행한다.

| 구분 | 역할 | 문제 세트 | 출력 | best 선택 |
|---|---|---|---|---|
| **MAIN validation** | 정식 지표 (in-distribution) | 학습분포(blocks `min..max`, target-matching 없음, 자연표본) | `<output>/validation/` | **✅ 이것으로 best checkpoint 선택** |
| generalization test | 일반화 성능 측정 | 기존 grid(blocks 10-100, target-matched) | `<output>/validation_generalization/` | ❌ 리포팅 전용 |

- 새 CLI 플래그: `--main-validation-min-blocks`(기본=`--min-blocks`), `--main-validation-max-blocks`(기본=`--max-blocks`), `--main-validation-block-gap`(기본 5), `--main-validation-episodes`(기본=`--validation-episodes`). 기존 `--validation-*`는 그대로 generalization grid를 구성.
- **학습/validation 겹침 없음(검증됨)**: MAIN held-out seed = `seed + 500_000 + i·7919`, 불변식 `offset+(N-1)·7919 < 1_000_003(EPISODE_SEED_STRIDE)` 로 모든 MAIN seed가 `(seed, seed+1_000_003)` 구간 → 학습 episode seed(`seed+k·1_000_003`, k≥1)·generalization seed(`≥seed+10_000_000`)와 **증명적으로 disjoint**. 실제 jobs 해시도 상이. main.py가 학습 시작 시 overlap을 assert하고 `validation_overlap_check: passed`를 출력(겹치면 raise).
- 코드: `Phase2/validation_grid.py`(`build_phase2_main_validation_grid` + 상수), `Phase2/merged.py`(`secondary_validation_problems` + `_run_phase2_validation_cycle`), `main.py`(플래그·빌드·overlap print). 검증: py_compile clean, unit 218 tests OK, smoke run에서 두 validation 디렉토리 생성 및 `best_checkpoint.json`은 `validation/`에만.
- 대시보드: `validation/`(MAIN)이 자동으로 표시되며, 본 run은 분리 이전이라 별도 eval한 ep15000 in-dist 88.1%를 참조선으로 오버레이.

## 5. 분석 재현 커맨드 (참고)
```bash
PY=~/anaconda3/envs/accord_env/bin/python
RUN=output/phase2_18merge_tempanneal_w8
# parent best-rate by block_count / distribution_type:  validation/validation_parent_history.csv
# per-Bay best-rate:                                     validation/validation_bay_history.csv
# grid 정의(문제별 target/actual 분포):                 validation/grid_contract.json
# best checkpoint 선택 결과:                             validation/best_checkpoint.json  (train_episode=15000)
```
