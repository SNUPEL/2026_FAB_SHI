# 학습 run 인덱스

`output/`에 남아 있는 학습 run과 그 결과를 보는 대시보드(Artifact)를 연결한 기록이다.
`/output/` 자체는 `.gitignore` 대상이지만 이 인덱스는 저장소에 남겨야 하므로 리포 루트에 둔다.
따라서 아래 표의 디렉터리는 모두 `output/` 아래 경로다(예: `phase2_h128/` → `output/phase2_h128/`).

**이 문서는 "무엇이 실제로 돌았고 결과가 얼마였나"만 기록한다.** 학습 실행·제출·비교 규칙은
[모델_학습_비교_절대규칙.md](모델_학습_비교_절대규칙.md)가 유일한 기준이며, 여기에 규칙을 다시 쓰지 않는다.
공유 판정 증빙은 `experiments/` 아래에 있다.

- 최종 갱신: 2026-08-07 (별칭 체계 도입, `_compare` 산출물 폐기 반영)
- 전체 용량: 4.9G + `_compare` 8.3G (`_compare`는 절대규칙 §2.4 전이 조항으로 **폐기 대상**)

---

## 1. 모델 현황 마스터 표

**별칭(ALIAS)이 run의 유일 ID다.** 별칭 하나가 디렉터리명·`--arm-label`·제출 경로·비교표 label을 전부
결정한다. 규칙은 [모델_학습_비교_절대규칙.md](모델_학습_비교_절대규칙.md), 이 표는 현황 기록이다.
(기준 2026-08-07)

### 1-1. Phase 1

| 별칭 | 담당 | 학습 디렉터리 | 위치 | 학습 상태 | 비교 가능 | 대시보드 |
| --- | :---: | --- | --- | --- | --- | --- |
| **P1-BASE** | **YC** | `phase1_mixed_resource_pool_v2` | 타 머신 (Windows) | ✅ **20,000 ep 완주** (2026-08-07 확인) | ⚠️ 방향성만 · **Phase 1 기준선** | [c854ba09](https://claude.ai/code/artifact/c854ba09-4c02-4d77-80c4-ef1e5c6d95f3) |
| **P1-ANNEAL** | **YJ** | `phase1_anneal_v1` | 타 머신 (Windows) | ✅ **20,000 ep 완주** | ⚠️ 방향성만 | [99e89a95](https://claude.ai/code/artifact/99e89a95-6fe1-4efb-b8a4-07a5fa45d535) |
| **P1-H16** | **SY** | ❓ 확인 필요 | 타 머신 | 🔄 **약 15,000 / 20,000 ep 진행 중** | ⏳ 완주 후 | [d0fbcf65](https://claude.ai/code/artifact/d0fbcf65-07e7-4d13-8ad9-c6864f9b7127) |
| ~~**P1-OLD**~~ | (로컬) | `phase1_pair_20k` | 로컬 `output/` | ⛔ 중단 ep4,627 | ❌ **비교 불가 — 구 데이터** | [83c887f0](https://claude.ai/code/artifact/83c887f0-d48c-46ab-93cd-d85105b21440) |

| 별칭 | hidden | blocks | temperature | rollout | 휴리스틱 | objective | 크기 |
| --- | ---: | --- | --- | ---: | --- | --- | --- |
| **P1-BASE** | **128** (86,401) | 12–80 | **1.0 고정** (당시 Phase 1 annealing 미지원) | 64 / val 64 | 3종 wo/cut/bevel_first_balanced | `series_only`, val-episodes 20, seed 0, device **cuda** | 224M(제출본) |
| **P1-ANNEAL** | **128** | 12–80 | 1.0→0.3 (20k 전구간) | 64 / val 64 | 3종 wo/cut/bevel_first_balanced | `series_only`, val-episodes 20, device **cpu** | — |
| **P1-H16** | **16** | 12–80 | P1-BASE와 동일 | P1-BASE와 동일 | P1-BASE와 동일 | `series_only` | (타 머신) |
| ~~P1-OLD~~ | 128 | 12–80 | — | 64 | 3종 (동일) | `shared_and_series`, cuda, checkpoint 46개 | 3.6G + 4.9M |

- **P1-H16은 P1-BASE에서 `--hidden-dim`만 128 → 16으로 바꾼 arm이다.** 캠페인 `p1_capacity` / arm_group `hidden_dim`.
- **P1-ANNEAL은 P1-BASE 대비 temperature만 다르다**(1.0 고정 → 1.0→0.3 anneal). 나머지(blocks 12–80,
  rollout 64/64, 휴리스틱 3종, hidden 128, val-episodes 20, `series_only`)가 같아 캠페인 `p1_temperature`가
  성립한다. 단 device가 cuda(P1-BASE) vs cpu(P1-ANNEAL)로 달라 비결정성 요인이 하나 남는다.
- **P1-BASE는 dirty tree에서 학습됐다**(`git commit 0c4960d4`, dirty=예). 절대규칙 A1 위반이라
  코드 동일성이 증명되지 않는다 — 비교표에 마커를 단다.
- Phase 1은 고정 validation grid와 contract가 없어 **결론 강도가 "방향성"으로 제한된다**(§5-2).
- P1-BASE·P1-ANNEAL 상세는 로컬 staging 사본(`output/_compare/models/p1_baseline_v2/`,
  `output/_compare/models/p1_tempanneal/`)의 checkpoint·`summary.json`에서 직접 읽은 값이다.

### 1-2. Phase 2

| 별칭 | 담당 | 학습 디렉터리 | 위치 | 학습 상태 | 비교 가능 | 대시보드 |
| --- | :---: | --- | --- | --- | --- | --- |
| **P2-BASE** | (로컬) | `phase2_18merge_tempanneal_w8` | 로컬 `output/` | ✅ **20,000 ep 완주** | ✅ **Phase 2 기준선** | [bdfc74b1](https://claude.ai/code/artifact/bdfc74b1-53f3-426f-8447-ec15ddf50934) |
| **P2-H128** | (로컬) | `phase2_h128` | 로컬 `output/` | ✅ **20,000 ep 완주** (08-03 21:13 → 08-06) | ✅ 가능 | [f5786278](https://claude.ai/code/artifact/f5786278-3240-4ad8-bf20-0c3ec6379b10) |
| **P2-B1080** | **HJ** | (타 HW 재시작본) | 타 HW | 🔄 **약 4,000 / 20,000 ep 진행 중** | ⏳ 완주 후 | 없음 (담당자 Claude 미사용) |
| **P2-FIXT** | **NY** | `phase2_fixedT10` | 타 HW | 🔄 **약 2,000~3,000 / 20,000 ep 진행 중** | ⏳ 완주 후 | [cee09a50](https://claude.ai/code/artifact/cee09a50-081b-4435-95b3-ebf9669e0564) |
| ~~**P2-OLD**~~ | (로컬) | `phase2_upstream_wo` | 로컬 `output/` | ⛔ 중단 ep7,500 | ❌ **비교 불가 — 구 데이터** | [83c887f0](https://claude.ai/code/artifact/83c887f0-d48c-46ab-93cd-d85105b21440) |

| 별칭 | hidden | blocks | temperature | rollout | 휴리스틱 | validation grid | 크기 |
| --- | ---: | --- | --- | ---: | --- | --- | --- |
| **P2-BASE** | **16** (4,801) | 10–30 | 1.0→0.3 (20k 전구간) | 64 | 4종 lpt/long_cut/best_fit_lth/long_bevel | generalization 10–100 gap10 ×5 = 50문제로 best 선택. MAIN은 사후 백필 | 973M + 26M |
| **P2-H128** | **128** (253,441) | 10–30 | 1.0→0.3 | 64 | 4종 (동일) | MAIN 10–30 gap5 ×5 = 25문제 실시간. generalization 10–100 ×5 = 50문제 | 1.5G + 26M |
| **P2-B1080** | 16 | **10–80** | 1.0→0.3 | 64 | 4종 (동일) | MAIN 10–80 gap10 ×5 = 40문제. generalization 10–100 ×5 | (타 HW) |
| **P2-FIXT** | 16 | 10–30 | **1.0 고정 (annealing 없음)** | 64 | 4종 (동일) | P2-BASE와 동일 | (타 HW) |
| ~~P2-OLD~~ | 128 | 12–80 | — | — | **6종** | 구 validation(`--validation-episodes 20`, 100 ep 주기), checkpoint 74개 | 257M + 12M |

- **P2-H128의 arm identity만 기록돼 있다**: `phase2_capacity / hidden_dim / h128`. 나머지 run은
  identity 플래그 없이 시작해 manifest identity가 비어 있다(절대규칙 A2 소급 대상).
- 부속 산출물(모델 아님): `phase2_indist_maincurve/`(1.8M) = **P2-BASE의 in-dist MAIN 곡선 백필**
  (ep1000~20000 × 25문제 = 500행). P2-BASE 대시보드의 "정식 지표" 차트가 이 CSV를 쓰므로 **삭제하면 재현 불가**.
- 로컬 `phase2_blocks10_80/`(ep1000) = **P2-B1080 로컬 폐기본**, 재개하지 않는다.

### 1-3. 공통

담당자(`--owner`)는 run마다 다르다. 로컬 run은 이 머신에서 직접 돌린 것이라 별도 담당자가 없다.
**P1-H16의 디렉터리명만 아직 미확인**이다 — 제출 요청 시 채운다.

run과 무관한 대시보드 — 학습 네트워크 구조도(Two Pointer Scoring Networks), 데이터 의존 없음:
https://claude.ai/code/artifact/7c5b9cb3-9b4f-425e-883d-d76aecedfa41

### 1-4. 왜 P1-OLD·P2-OLD는 비교에서 뺐는가

둘 다 **데이터 산출식 수정 이전**의 합성데이터로 학습됐다. 문제 분포 자체가 달라
어떤 재채점으로도 공정 비교가 성립하지 않으므로 §4 비교 대장에서 제외한다. 기록은 이 표에만 남긴다.

P2-OLD는 그에 더해 **휴리스틱 6종**이라, `heuristic_algorithms`가 `phase2_run_spec_v4_event_clock`의
필드인 이상 loader가 애초에 재채점을 거부한다(절대규칙 A13).

### 1-5. hidden_dim 주의사항

CLI 기본값은 Phase 1/2 모두 128인데, **기준선 두 개가 서로 다른 값으로 돌았다 — P1-BASE(128)와
P2-BASE(16)를 같은 조건으로 보면 안 된다.**

파라미터 수는 pointer policy 기준 `15·h² + 60·h + 1`이라 h에 제곱으로 늘어난다:
h=16 → 4,801, h=64 → 65,281, h=128 → 253,441. Phase 1 pair policy는 계수가 달라 h=128에서 86,401이다.

**Phase 1 CLI는 로그 헤더에 `hidden_dim`을 찍지 않는다**(Phase 2는 찍는다). Phase 1 run의 hidden_dim은
checkpoint를 직접 열어야 알 수 있다:

```bash
python -c "import torch;ck=torch.load('<run>/phase1_pair_pointer_best.pt',map_location='cpu',weights_only=False);print(ck['hidden_dim'])"
```

P1-BASE·P2-BASE 모두 `run_manifest.json` 도입(2026-08-03) 이전이라 **manifest가 없다.**
manifest를 남기는 첫 run은 P2-B1080(로컬 폐기본)이다.

---

## 2. 어떤 파일이 대시보드를 그리는가

| 지표 | 파일 |
| --- | --- |
| 학습 loss / hard violation / best-source | `<run>/metrics.csv`, `<run>/subproblem_metrics.csv` |
| Phase 1 holdout best-rate | `<run>/validation_summary.csv` (UTF-8 BOM — `utf-8-sig`로 읽을 것) |
| Phase 2 validation(문제별) | `<run>/validation/validation_parent_history.csv`, `validation_bay_history.csv` |
| Phase 2 rank·winner 집계 | `<run>/validation/validation_rank_history.csv`, `validation/evaluations/ep_*/aggregate/*.csv` |
| 문제집합 지문 | `<run>/validation/grid_contract.json` (두 run이 같은 문제로 채점됐는지 판별하는 기준) |
| 실행 조건 전체 | `<run>/run_manifest.json` (2026-08-03 이후 run만), `<run>/summary.json` (학습 완주 시에만 기록) |

용량 대부분은 그래프에 쓰이지 않는 감사 로그다: **P1-OLD**의 `candidate_summary.csv`(1.9G)·
`validation_candidate_summary.csv`(1.2G)·`best_action_table.jsonl`(529M),
**P2-BASE**의 `validation/evaluations/`(932M). 이 파일들은 제출 금지 대상이다(절대규칙 §4.4).

---

## 3. 2026-08-03 정리 기록

아래 32개 항목을 삭제했다(15G → 4.9G). 재생성이 필요하면 각 run의 CLI를 다시 실행해야 한다.

| 그룹 | 삭제 대상 |
| --- | --- |
| 260728 계열 중단 본학습 | `phase1_run_260728`(9.0G, 9,695 ep)·`.log`·`.pid`, `phase2_run_260728`(232M, 6,781 ep)·`.log`·`.pid`, `_baseline_260728`(16M, 위 두 run CSV 사본), `phase2_run_expC`(36M, 1,096 ep)·`.log` |
| smoke·벤치·중단 조각 | `bench`(171M), `phase1_smoke`, `phase2_smoke`, `phase2_smoke_par8`(각 +log), `phase2_parallel_smoke_w4/w8/w8_ep2/w8_solo/w16/w16_solo`, `phase2_profile_before`, `phase2_upstream_wo_seq_interrupted_ep12_20260724`(+log), `phase2_upstream_wo_sequential_interrupted_ep7_20260722`, `phase1_anneal_smoke`(1.8M, Phase 1 annealing 4-ep 스모크)·`.log` |
| 학습 아닌 산출물 | `actual_replay_np_100`, `playback_np_100` (해당 CLI는 cleanup으로 제거되어 재생성 불가) |
| 파생·대체됨 | `phase2_indist_eval`(12M, lr 동결 우회 백필 — 결과는 `phase2_indist_maincurve`에 반영, 방식은 `--eval-only`로 대체), `dashboard`(524K, 실행 시마다 덮어써지는 스냅샷) |

당시 실행 중이던 P2-B1080 로컬본(`phase2_blocks10_80`)은 삭제 전 가드로 제외했고, 삭제 후 `metrics.csv`의
episode가 결번 없이 연속임을 확인했다.

---

## 4. Run별 best model (비교 대장)

추후 **Phase 1끼리 / Phase 2끼리 best model 비교**용. best checkpoint 선택 지표
`selection_key = [non_win, -win, mean_rank]`(사전식 최소가 best) → **best-rate = win / 문제수**.

> ⚠️ **아래 수치는 각 run이 자기 grid에서 낸 값이라 run 간 비교에 쓸 수 없다**(절대규칙 A6).
> P2-BASE는 generalization grid, P2-H128·P2-B1080은 main in-dist grid로 best를 골랐다.
> 나란히 비교하려면 [모델_학습_비교_절대규칙.md](모델_학습_비교_절대규칙.md) §5의 중앙 재채점을 거쳐야 한다.

**best model을 ID / OOD 두 기준으로 각각 선정한다:**
- **ID best** = in-distribution(main) validation, 즉 **학습분포와 같은 블록범위**의 고정 grid에서 best-rate 최고 checkpoint.
- **OOD best** = generalization validation(blocks 10-100, distribution-type target-matched)에서 best-rate 최고 checkpoint.
- 선정 규칙: parent 문제 `proposed_best_rank==1`(win) 최다, 동률 시 mean_rank 최소, 그래도 동률 시 이른 ep.
  best-rate = win / 문제수. (best_checkpoint.json의 `selection_key = [non_win, -win, mean_rank]`와 동일 기준)

### Phase 1

Phase 1은 validation을 계열(NP/NC/FN/FL) view로 나눌 뿐 ID/OOD 블록 grid 분리가 없어 단일 best만 있다.

표준 조건 재채점: `--eval-only`, seed 0, blocks 12–80, **200문제**, **K=1**(greedy 단독 → 결정적),
`series_only`, 휴리스틱 3종. 지표는 pool view(`NP_NC`+`FN_FL`) best-rate, n=400.

| run | hidden / blocks | 상태 | **재선정 best** | 지표 | 자체 best.pt (쓰지 않음) |
|---|---|---|---|---|---|
| **P1-BASE** | 128 / 12–80 | ✅ 완주 20k | **ep15000** | 46.0% ±4.9, mr 2.022 | ep300 → 22.5% |
| **P1-ANNEAL** | 128 / 12–80 | ✅ 완주 20k | 미측정 (재채점 대기) | — | ep200 |
| **P1-H16** | 16 / 12–80 | 🔄 ~15k | — | — | — |

> ⚠️ **P1-BASE 상위 6개(ep15000·16000·17000·3400·20000·6800)는 전부 CI가 겹친다.** ep15000은
> tie-break(best-rate → mean_rank → 이른 ep)로 정한 것이며 통계적 우위는 없다. 근거·전체 궤적·
> 배제한 checkpoint는 [`experiments/_baselines/P1-BASE/BEST_CHECKPOINT.md`](experiments/_baselines/P1-BASE/BEST_CHECKPOINT.md).

### Phase 2 — ID / OOD best 각각

checkpoint 경로 = `output/<run>/checkpoints/phase2_batch_machine_policy_ep<NNNNN>.pt`.

| run | hidden_dim / blocks | 상태 | **ID best** (in-dist) | **OOD best** (generalization 10-100) |
|---|---|---|---|---|
| **P2-BASE** `phase2_18merge_tempanneal_w8` | 16 / 10-30 | 완료 20000ep | **ep13000** → 23/25 = **92.0%**, mr 1.08 (in-dist 10-30, 사후 백필) | **ep15000** → 25/50 = **50.0%**, mr 1.5  *(= `phase2_best.pt`)* |
| **P2-H128** `phase2_h128` | **128** / 10-30 | **완주 20000ep** | **ep11000** → 23/25 = **92.0%**, mr 1.08  *(= `phase2_best.pt`)* | **ep19000** → 29/50 = **58.0%**, mr 1.42 |

> **P1-OLD·P2-OLD는 이 표에 올리지 않는다.** 데이터 산출식 수정 **이전**의 합성데이터로 학습돼 문제 분포
> 자체가 다르므로 어떤 재채점으로도 공정 비교가 성립하지 않는다. 기록은 §1과 §6에 남긴다.
> (P2-OLD는 휴리스틱도 6종이라 `heuristic_algorithms`가 `run_spec` 필드인 이상 loader가 애초에 거부한다.)

**타 하드웨어에서 진행 중 (여기 output/ 에 데이터 없음 — best 도착 시 재기록):**

| run (라벨) | hidden / blocks / temp | 상태 | 대시보드 | best |
|---|---|---|---|---|
| **P2-FIXT** `phase2_fixedT10` (temperature ablation) | 16 / 10-30 / **T=1.0 고정(annealing 없음)** | 타 HW 진행 중 | https://claude.ai/code/artifact/cee09a50-081b-4435-95b3-ebf9669e0564 | 미정(진행 중) |
| **P2-B1080** (blocks10-80 재시작본) | 16 / **10-80** / annealed 1.0→0.3 | 타 HW에서 **fresh 재시작·진행 중** | 없음(담당자 Claude 미사용) | 미정(진행 중) |

> 로컬 `phase2_blocks10_80`은 ep1000에서 중단된 뒤 **로컬 재개하지 않음** — 동일 설정으로 **타 하드웨어에서 처음부터 재시작**했다.
> 위 "재시작본"이 실제 진행 run이고, 로컬 `output/phase2_blocks10_80/`(ep1000 checkpoint)은 폐기 대상.

**관찰:** ID best와 OOD best는 **서로 다른 checkpoint**가 될 수 있다.
예) P2-BASE는 ID best=ep13000·OOD best=ep15000, P2-H128은 ID best=ep11000·OOD best=ep19000
(ID·OOD 정점 위치가 서로 다르다). 각 run의 공식 `phase2_best.pt`는 그 run이 학습 중 **선택에 쓴 grid**의 best다
(P2-BASE=OOD 기준, P2-H128/P2-B1080=ID 기준).

- 코드가 ID·OOD best 두 checkpoint를 **학습 중 자동 저장**하게 하려면 `Phase2/merged.py`(`select_best`를
  secondary에도 적용)를 수정해야 하는데, 과거에는 run_manifest fingerprint 때문에 로컬 진행 중 run이 깨졌다.
  → **P2-H128이 2026-08-06 완주**하여 로컬에 진행/재개대기 run이 없다. **이제 tracked .py 수정 가능**(fingerprint 제약 해제). 그 전까지는 위처럼 사후 산출했다.
- 갱신 시점: `phase2_h128` 종료(예상 08/06) 시, 그리고 타 하드웨어에서 학습 중인 run(blocks10_80 재시작본·fixedT10 등)의
  best model이 도착할 때 ID/OOD best 재기록. (로컬 `phase2_blocks10_80`은 재개하지 않으므로 그 dir 기준으로는 갱신하지 않는다.)

### 4-1. P2-BASE 수렴 진단 (2026-08-03)

> 이후 실험 설계(MAIN/generalization 분리, `blocks10_80`, `fixedT10`, `h128` arm)의 근거가 된 분석이라
> 요약을 남긴다. 원본 `Phase2_18merge_수렴분석_및_재학습계획.md`는 이 절로 흡수하고 삭제했다(2026-08-07).

**증상**: 20,000 ep 완주했으나 holdout parent best-rate(generalization grid 50문제)가
`8,18,22,22,22,26,28,30,38,20,44,24,46,34,50,30,46,44,26,22`(ep1k~20k)로 **44~50% 대역에서 큰 진폭으로
진동**하고 마지막 ep20000은 저점(22%). 단조 수렴 없음.

**진단 A 【지배적】 — train/validation 문제 크기 불일치(OOD)**: 학습은 blocks 10–30(job≤334)만 보는데
validation grid는 10–100(job≤785)까지 평가. 후반 9개 checkpoint(ep12k~20k) 평균 best-rate가 크기에 따라
단조 붕괴 — blocks 10: **100%**, 20: **87%**, 30: 42%, 40: 24%, 50: 47%, 60~100: 낮음.
집계로 in-dist(10–30) **76%** vs OOD(60–100) **12%**(80–100은 0% 일관). 대형 문제에서는 `lpt_batch`가
makespan 지표상 구조적으로 강해 5/5 승리. → 전체 best-rate(~36~50%)는 **학습에서 본 적 없는 대형 문제가
절반을 끌어내린 결과**.

**확인 실험** (`phase2_best.pt` 가중치 동결, blocks 10–30 균등·held-out seed 84문제):

| 지표 | 일반화 grid (10–100) | **in-distribution (10–30)** |
|---|---:|---:|
| 전체 best-rate | 50% | **88.1%** |
| mean_rank | 1.50 | **1.12** |
| NP_NC pool | 53% | 88% |
| FN_FL pool | 38% | **94%** |
| Bay25 | 28% | **94%** |

**진단 B — subproblem 불균형은 OOD 아티팩트였다**: generalization grid에서는 FN_FL(38%), 특히 Bay25(28%,
mean_rank 1.72)가 약해 보였으나, in-dist에서는 FN_FL 94% / Bay25 94%로 **오히려 최강**. "Bay25 약점"은
측정 아티팩트로 판정.

**진단 C — 최적화 불안정(진동의 직접 원인)**: ① LR 0.001 고정, decay 스케줄러 없음 → 좋은 영역 주변을
계속 튐(진폭이 끝까지 일정). ② temperature anneal이 전 구간(20,000 ep)이라 하한 0.3에 맨 끝에서야 도달
(ep15k T≈0.40, ep18k T≈0.34) → **저온 consolidation 구간이 사실상 없어** greedy takeover(11%→최대 31%)가
자리잡지 못함.

**후속 조치와 현황**

| 진단 | 조치 | 상태 |
|---|---|---|
| A (측정 정합) | MAIN(in-dist) validation을 정식 지표로 분리, best 선택 기준 이관 | ✅ 구현 완료 → 체크리스트 §18, `Phase2_Validation_구조.md` §2 |
| A (학습 확장) | 학습 블록범위 10–80으로 확대한 arm | `phase2_blocks10_80` (타 HW 재시작본 진행 중) |
| C (temperature) | anneal 설계 검증용 ablation | `phase2_fixedT10` (T=1.0 고정, 타 HW 진행 중) |
| C (LR decay) | 스케줄러 도입 또는 2단계 저-LR 재개 | ❗ 미착수 |
| capacity | `--hidden-dim` 16 → 128 | **P2-H128** ✅ 완주 (OOD best 58.0% vs baseline 50.0%. **차이 8.0%p < 95% CI ±13.9%p → 유의하지 않다**) |

미해결 질문: 실제 절단 shop이 다루는 블록 규모가 소형(≤30~40)인지 대형(~100)까지인지에 따라
"학습 확장(A1)"과 "측정 정합(A2)" 중 무엇을 정식 기준으로 둘지가 갈린다.

---

## 5. best model 공정 비교 — 규칙은 다른 문서에 있다

비교 규칙·제출 계약·재채점 절차·무효 판정·결론 강도는 전부
[모델_학습_비교_절대규칙.md](모델_학습_비교_절대규칙.md)가 유일한 기준이다. 여기에 복제하지 않는다.
캠페인 산출물은 `experiments/<ARM-GROUP>/comparison/`에 커밋한다.

### 5-0. 1차 시도(P1-ANNEAL vs P1-BASE)에서 실제로 드러난 함정 — 근거 기록

규칙의 근거가 된 실측이라 대장에 남긴다.

- **run 자체 validation 수치는 비교 불가**: seed·grid가 제각각. 같은 `ep200` 모델이 담당자 grid **86%** vs 여기 seed-0 grid **43.6%**.
- **best.pt 선정 기준이 run마다 다름**: Phase 1=`mean_score`, Phase 2=ID/OOD grid → 자체 best.pt가 실제 최고성능이 아닐 수 있음
  (P1-ANNEAL best.pt=**ep200**인데 best-rate 최고는 **ep19800**).
- **소표본 노이즈**: val 20문제 → view별 95% CI ±15~22%p. 웬만한 차이는 통계적으로 무의미.
- **objective_scope 불일치**: `series_only` vs `shared_and_series` → 채점 기준이 다름.
- **문제 크기 편향**: blocks 12–80이라 뽑히는 문제 크기(8~48)가 seed마다 크게 달라 난이도가 요동.

### 5-1. 기존 `_compare` 산출물 폐기 (2026-08-07)

절대규칙 §2.4 전이 조항에 따라 아래를 폐기하고 재생성한다. **이 수치들을 인용하지 않는다.**

| 폐기 대상 | 사유 (실측) |
| --- | --- |
| `output/_compare/eval/p1cmp_*` | 1,191행 중 고유 `(train_episode, problem_id)` 150개 — **1,041행 중복**(같은 dir에 `--eval-only` 반복 실행) |
| `output/_compare/phase1_comparison.csv` | 위 중복으로 `overall_n=1191`이 허수(실제 150) → CI 약 2.8배 과소 추정. 또 `candidate_count=7`·`agent_sample_*` 존재 → K=4로 채점(규칙은 K=1). 재채점 run 간 `--validation-episodes`도 20 vs 200으로 상이 |
| `output/_compare/phase2_comparison.csv` | `ID_grid` 지문 3종 불일치(`(no_contract)`/`4af6dbe1…`/`77a1473a…`) → 서로 다른 ID 문제로 채점. h128 OOD 값도 완주 전(ep6000) |
| `output/_compare/models/` | allowlist 없이 통째 복사해 `p1_tempanneal/`이 **8.0 GB** |

### 5-2. Phase 1 parity 미결

Phase 1은 고정·층화 grid와 `contract_sha256`이 없어 결론 강도가 **"방향성"**으로 제한된다.
해제 조건 U1~U5는 절대규칙 §6.8, 진행 상태는 [현재과제_진행체크리스트.md](현재과제_진행체크리스트.md) §21.
**U1(Phase 1 `--eval-only` 커밋)이 최우선** — 현재 미커밋이라 Phase 1 재채점 결과는 코드가 증명되지 않는다.
