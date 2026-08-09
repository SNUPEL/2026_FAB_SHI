# P2-FIXT — 제출 폴더

**여기에 파일을 붙여넣으면 된다.** 규칙 전문은 [모델_학습_비교_절대규칙.md](../../../모델_학습_비교_절대규칙.md) §4.

| | |
| --- | --- |
| 별칭 | **P2-FIXT** |
| Phase | 2 |
| 담당 (`--owner`) | **NY** |
| 원 학습 경로 | `output/phase2_fixedT10` |
| 캠페인 / 비교 축 | `temperature`: P2-BASE(1.0→0.3) 대비 **T=1.0 고정** |
| 학습 상태 | 🔄 약 2,000~3,000 / 20,000 ep 진행 중 — **완주 후 제출** |

대시보드: https://claude.ai/code/artifact/cee09a50-081b-4435-95b3-ebf9669e0564

---

## 1차 제출 — 여기에 넣을 파일 (Phase 2)

| 파일 | 필수 | 비고 |
| --- | :---: | --- |
| `run_manifest.json` | ✅ | 코드·데이터·grid·identity 증명 |
| `summary.json` | ✅ | run_spec·validation_contract·온도 계열 |
| `metrics.csv` | ✅ | 학습 곡선. 약 6 MB |
| `validation/grid_contract.json` | ✅ | **문제 동일성 증명** |
| `validation/validation_parent_history.csv` | ✅ | **checkpoint별 판정 근거 — 여기서 best를 고른다** |
| `validation/validation_rank_history.csv` | ✅ | rank·winner 집계 |
| `validation_generalization/` 의 같은 3종 | ✅ | MAIN grid를 쓴 run이면 함께 |
| `run.log.tail` (뒤 2000줄) | 권장 | 헤더의 실제 플래그 |
| `checkpoints/*.pt` | ⏳ | **지금 넣지 않는다.** 아래 "2차 제출" 참조 |

## 넣지 말 것

`best_action_table.jsonl`(529 MB) · `candidate_summary.csv`(1.9 GB) ·
`validation_candidate_summary.csv`(1.2 GB) · `evaluations/**` · `problems/**` · PNG 전량 ·
`checkpoints/` 전량. 앞의 것들은 `.gitignore`가 막으므로 실수로 `git add`해도 들어가지 않는다.

## 2차 제출 — checkpoint

**1차 제출을 먼저 한다.** 중앙에서 `validation/validation_parent_history.csv`를 읽어
best-rate 상위 후보 3개를 지목하면, 그 `.pt`만 `checkpoints/P2-FIXT_ep<NNNNN>.pt`로 넣는다.
개당 약 3.0 MB(h128) / 128 KB(h16).

**run 자체 `best.pt`는 보내지 않아도 된다.** 그 파일은 이 캠페인과 다른 기준으로 뽑힌 것이라
쓰지 않는다(규칙 A7). 궁금하면 참고용으로 함께 넣되 파일명에 `_selfbest`를 붙인다.

---

## 붙여넣은 뒤 확인

```bash
git status --porcelain experiments    # allowlist 밖 파일이 뜨면 중단
du -sh experiments/p2_temperature/P2-FIXT          # 50 MB 이하여야 한다
```
