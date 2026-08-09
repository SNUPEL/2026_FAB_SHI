# P1-BASE best checkpoint 재선정 (2026-08-07)

run 자체 `best.pt`를 쓰지 않고(절대규칙 A7) 표준 조건으로 다시 골랐다.

## 채점 조건 (전 checkpoint 동일)

```
--eval-only --hidden-dim 128 --objective-scope series_only --heuristic-algorithms all
--rollout-samples 1 --rollout-samples-validation 1        # K=1 → 채점 결정적 (A11)
--seed 0 --min-blocks 12 --max-blocks 80 --validation-episodes 200
```

`candidate_count=4`(greedy 1 + 휴리스틱 3)로 K=1 적용을 실측 확인했다.
지표는 pool view(`NP_NC`+`FN_FL`) best-rate, n=400. `overall` 행은 view 중복 집계라 쓰지 않는다(A13).

## 절차

1. 학습 궤적 **22개 지점**을 100문제로 스크리닝 (ep300~ep20000)
2. 상위 6개를 **200문제**로 재확인

## 최종 결과 (200문제, pool n=400)

| ep | best-rate | ±CI | mean_rank | NP_NC | FN_FL |
|---|---:|---:|---:|---:|---:|
| **15000** | **46.0%** | ±4.9 | 2.022 | 42.5% | 49.5% |
| 16000 | 44.8% | ±4.9 | 2.020 | 43.5% | 46.0% |
| 17000 | 44.2% | ±4.9 | 2.095 | 40.0% | 48.5% |
| 3400 | 43.5% | ±4.9 | 2.033 | 42.5% | 44.5% |
| 20000 | 41.2% | ±4.8 | 2.147 | 35.5% | 47.0% |
| 6800 | 41.2% | ±4.8 | 2.152 | 37.0% | 45.5% |

## 선정: **ep15000** — 단 통계적 우위는 없다

**6개 전부 CI가 겹친다.** 1위와 6위 차이가 4.8%p인데 CI 합이 9.7%p다.
"ep15000이 더 낫다"고 말할 근거는 없으며, 문서화된 tie-break(best-rate → mean_rank → 이른 ep)로
정한 것이다. 대안은 ep16000(mean_rank가 2.020으로 근소하게 낫다).

보조 근거: **ep15000–17000이 연속 고원**이다(100문제 스크리닝에서 47.0/43.5/43.5로 상위 3개가 인접).
ep3400은 이웃이 낮은 단독 봉우리(35.5 → 45.0 → 41.0)라 표본 노이즈일 가능성이 있다.

## 확실한 배제 (CI 겹치지 않음)

- **자체 `best.pt` = ep300, 22.5%** (100문제 기준). `agent_mean_score`로 뽑혀 최하위다. 쓰면 안 된다.
- **`reselected_ep10600_best.pt` = 33.5%.** `checkpoints/phase1_pair_pointer_ep10600.pt`와 SHA256
  동일한 사본이며(`25086c3d…`), run 자체 holdout 20문제에서 97.5%로 보였던 것은 표본 노이즈였다.
  선정 기준·문제집합 기록이 없어 근거를 재현할 수 없다.

## 관찰: 수렴하지 않았다

ep1800에 이미 42%에 도달한 뒤 20,000 ep 내내 30~47%를 진동한다. 후반이 전반보다 뚜렷하게 낫지 않다.
P2-BASE에서 진단한 것과 같은 패턴이다(LR 0.001 고정, decay 없음 — RUNS.md §4-1 진단 C).

## 한계

- Phase 1은 고정 grid·contract가 없어 **결론 강도는 "방향성"까지만**이다(절대규칙 §6.7).
- 이 run은 `git dirty=예`, `run_manifest.json` 없음 → **코드 동일성 미증명**(A1 위반).
- 절대값은 run 자체 수치(K=64 best-of-64)보다 훨씬 낮다. K=1 greedy 단독이라서다. **run 자체
  수치와 나란히 놓지 않는다**(A6).

## 파일

- 선정·대안·비교용 3개만 이 폴더에 둔다: `P1-BASE_ep15000.pt`, `P1-BASE_ep16000.pt`, `P1-BASE_ep03400.pt`
- 나머지 200개와 `_selfbest_*`는 `output/_compare/models/P1-BASE/`(gitignore)
- 재채점 산출물: `output/_compare/eval/P1-BASE__ep*_n100|n200/`
