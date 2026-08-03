# 학습 run 인덱스

`output/`에 남아 있는 학습 run과 그 결과를 보는 대시보드(Artifact)를 연결한 기록이다.
`/output/` 자체는 `.gitignore` 대상이지만 이 인덱스는 저장소에 남겨야 하므로 리포 루트에 둔다.
따라서 아래 표의 디렉터리는 모두 `output/` 아래 경로다(예: `phase2_h128/` → `output/phase2_h128/`).

- 최종 갱신: 2026-08-03
- 전체 용량: 4.9G (정리 전 15G)

---

## 1. 디렉터리 ↔ 대시보드 매칭

| 디렉터리 | 크기 | 대시보드 | 설명 |
| --- | --- | --- | --- |
| `phase1_pair_20k/` + `.log` | 3.6G + 4.9M | (OLD) Phase 1, 2<br>https://claude.ai/code/artifact/83c887f0-d48c-46ab-93cd-d85105b21440 | **첫 20k 본학습의 Phase 1** (데이터 생성 수정 전). 목표 20,000 ep 중 **4,627 ep에서 중단**. blocks 12–80, rollout 64, 휴리스틱 3종(wo/cut/bevel_first_balanced), objective `shared_and_series`, cuda. checkpoint 46개 |
| `phase2_upstream_wo/` + `.log` | 257M + 12M | (OLD) Phase 1, 2<br>https://claude.ai/code/artifact/83c887f0-d48c-46ab-93cd-d85105b21440 | **첫 20k 본학습의 Phase 2**. 목표 20,000 ep 중 **7,500 ep에서 중단**. blocks 12–80, hidden 128, 휴리스틱 6종, 구(舊) validation 방식(`--validation-episodes 20`, 100 ep 주기). checkpoint 74개 |
| (로컬에 없음)<br>`phase1_mixed_resource_pool_v2` | — | (Baseline) Phase 1<br>https://claude.ai/code/artifact/c854ba09-4c02-4d77-80c4-ef1e5c6d95f3 | **데이터 산출식 수정 후 Phase 1 Baseline.** 대시보드의 `run.dir`이 Windows 경로(`output\phase1_mixed_resource_pool_v2`)이고 2026-08-03 15:12 기준 3,880/20,000 ep 진행 중. **hidden 128**(파라미터 86,401 — 2026-08-03 해당 머신에서 checkpoint 직접 확인). **이 머신의 output/에는 없다** — 다른 머신에서 실행 중 |
| `phase2_18merge_tempanneal_w8/` + `.log` | 973M + 26M | (Baseline) Phase 2<br>https://claude.ai/code/artifact/bdfc74b1-53f3-426f-8447-ec15ddf50934 | **데이터 산출식 수정 후 Phase 2 Baseline, 20,000 ep 완주.** 18th merge(tie-aware CE + 고정 validation grid)에 greedy 통일·temperature annealing(1.0→0.3, 20,000 ep)·params 데이터 생성 결합. 학습 blocks **10–30**, hidden 16, rollout 64, 휴리스틱 4종(lpt/long_cut/best_fit_lth/long_bevel), validation·checkpoint 1,000 ep 주기, grid 10–100 gap 10 × Type 5 = **50문제**. best checkpoint = **ep 15000**(`phase2_best.pt`, 50문제 중 25승 25패, 평균 rank 1.5) |
| `phase2_indist_maincurve/` | 1.8M | (Baseline) Phase 2의 in-dist MAIN 차트 전용<br>https://claude.ai/code/artifact/bdfc74b1-53f3-426f-8447-ec15ddf50934 | 위 run은 main-validation 도입 이전이라, 저장 checkpoint를 사후 재평가해 백필한 곡선(ep 1000~20000 × 25문제 = 500행). 대시보드의 "정식 지표" 차트가 이 CSV를 쓴다. **삭제하면 그 차트 재현 불가** |
| `phase2_blocks10_80/` + `.log` | 5.1M + 460K | (블록 개수 확장 10-80) Phase 2<br>https://claude.ai/code/artifact/a53f5799-e45d-43af-a730-4ad3b861d279 | **실행 중**(2026-08-03 14:59 시작). 학습·validation 블록 범위 불일치를 없애려고 학습 blocks를 **10–80**으로 확장한 20,000 ep run. hidden 16, rollout 64, T 1.0→0.3, candidate_workers 32, cuda, seed 0. MAIN grid 10–80 gap 10 × 5 = 40문제(실시간), generalization grid 10–100 gap 10 × 5. `run_manifest.json`을 남기는 첫 run |

run과 무관한 대시보드 — 학습 네트워크 구조도(Two Pointer Scoring Networks), 데이터 의존 없음:
https://claude.ai/code/artifact/7c5b9cb3-9b4f-425e-883d-d76aecedfa41

### 1-1. run별 hidden_dim (2026-08-03 실물 확인)

CLI 기본값은 Phase 1/2 모두 128인데, Baseline은 두 phase가 서로 다른 값으로 돌았다.
**Baseline Phase 1(128)과 Phase 2(16)를 같은 조건으로 보면 안 된다.**

| run | phase | hidden_dim | 파라미터 | 확인 방법 |
| --- | --- | --- | --- | --- |
| `phase1_pair_20k` (OLD) | 1 | 128 | 86,401 | checkpoint `hidden_dim` |
| `phase1_mixed_resource_pool_v2` (**Baseline P1**) | 1 | **128** | 86,401 | 실행 머신에서 checkpoint 직접 확인 |
| `phase2_upstream_wo` (OLD) | 2 | 128 | — | 로그 헤더 |
| `phase2_18merge_tempanneal_w8` (**Baseline P2**) | 2 | **16** | 4,801 | 로그 헤더 + checkpoint |
| `phase2_blocks10_80` (실행 중) | 2 | **16** | 4,801 | 로그 헤더 |

파라미터 수는 `15·h² + 60·h + 1`(pointer policy 기준)이라 h에 제곱으로 늘어난다: h=16 → 4,801,
h=64 → 65,281, h=128 → 253,441. Phase 1 pair policy는 계수가 달라 h=128에서 86,401이다.

**Phase 1 CLI는 로그 헤더에 `hidden_dim`을 찍지 않는다**(Phase 2는 찍는다). Phase 1 run의 hidden_dim은
checkpoint를 직접 열어야 알 수 있다:

```bash
python -c "import torch;ck=torch.load('<run>/phase1_pair_pointer_best.pt',map_location='cpu',weights_only=False);print(ck['hidden_dim'])"
```

Baseline 두 run 모두 `run_manifest.json` 도입(2026-08-03) 이전이라 manifest가 없다. manifest를 남기는
첫 run은 `phase2_blocks10_80`이다.

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

용량 대부분은 그래프에 쓰이지 않는 감사 로그다: `phase1_pair_20k`의 `candidate_summary.csv`(1.9G)·
`validation_candidate_summary.csv`(1.2G)·`best_action_table.jsonl`(529M),
`phase2_18merge_tempanneal_w8/validation/evaluations/`(932M).

---

## 3. 2026-08-03 정리 기록

아래 32개 항목을 삭제했다(15G → 4.9G). 재생성이 필요하면 각 run의 CLI를 다시 실행해야 한다.

| 그룹 | 삭제 대상 |
| --- | --- |
| 260728 계열 중단 본학습 | `phase1_run_260728`(9.0G, 9,695 ep)·`.log`·`.pid`, `phase2_run_260728`(232M, 6,781 ep)·`.log`·`.pid`, `_baseline_260728`(16M, 위 두 run CSV 사본), `phase2_run_expC`(36M, 1,096 ep)·`.log` |
| smoke·벤치·중단 조각 | `bench`(171M), `phase1_smoke`, `phase2_smoke`, `phase2_smoke_par8`(각 +log), `phase2_parallel_smoke_w4/w8/w8_ep2/w8_solo/w16/w16_solo`, `phase2_profile_before`, `phase2_upstream_wo_seq_interrupted_ep12_20260724`(+log), `phase2_upstream_wo_sequential_interrupted_ep7_20260722`, `phase1_anneal_smoke`(1.8M, Phase 1 annealing 4-ep 스모크)·`.log` |
| 학습 아닌 산출물 | `actual_replay_np_100`, `playback_np_100` (해당 CLI는 cleanup으로 제거되어 재생성 불가) |
| 파생·대체됨 | `phase2_indist_eval`(12M, lr 동결 우회 백필 — 결과는 `phase2_indist_maincurve`에 반영, 방식은 `--eval-only`로 대체), `dashboard`(524K, 실행 시마다 덮어써지는 스냅샷) |

실행 중이던 `phase2_blocks10_80`은 삭제 전 가드로 제외했고, 삭제 후 `metrics.csv`의 episode가
결번 없이 연속임을 확인했다.
