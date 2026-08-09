# experiments/ — 공유 실험 증빙

여러 머신에서 돌린 학습 결과를 **판정 가능한 형태로 모으는 곳**이다. 규칙은 전부
[모델_학습_비교_절대규칙.md](../모델_학습_비교_절대규칙.md)에 있다. 이 파일은 요약이다.

`output/`은 `.gitignore` 대상이라 이 머신에만 존재한다. **판정에 인용하는 숫자의 근거는 반드시
여기에 있어야 한다.**

---

## 구조

```
experiments/_baselines/<ALIAS>/           # 여러 캠페인이 공유하는 기준 arm (P1-BASE, P2-BASE)
experiments/<EXPERIMENT-ID>/<ALIAS>/      # 그 캠페인의 비교 arm. 예: p2_capacity/P2-H128/
  run_manifest.json  summary.json  metrics.csv  run.log.tail
  validation/{grid_contract.json,validation_parent_history.csv,validation_rank_history.csv}   # Phase 2만
  validation_summary.csv                                                                      # Phase 1만 (루트)
  checkpoints/<ALIAS>_ep<NNNNN>.pt        # 2차 제출 — 중앙이 지목한 것만
experiments/<EXPERIMENT-ID>/comparison/   # 중앙 재채점 결과 CSV + 판정문
```

**Phase 1과 Phase 2는 낼 파일이 다르다.** Phase 1에는 `validation/` 디렉터리도 `grid_contract.json`도
없고 전부 run 루트에 있다 — 목록은 절대규칙 §4.2(Phase 2) / **§4.2b(Phase 1)**를 따로 본다.
`run_manifest.json`이 없는 구 run(2026-08-03 이전 시작)은 **§4.7의 대체 증빙 4종**을 대신 낸다.

`<ALIAS>`는 별칭이다(`P2-BASE`, `P2-H128`, `P1-BASE` …). 별칭 하나가 디렉터리명·`--arm-label`·
이 경로·비교표 label을 **전부** 결정한다. 현재 별칭표는 절대규칙 §1.1.

## 폴더는 이미 만들어져 있다

모델별 폴더와 **그 모델 전용 안내(`README.md`)**가 미리 생성돼 있다. 자기 별칭 폴더를 열고
그 안의 `README.md`가 시키는 대로 파일을 넣으면 된다.

| 별칭 | 담당 | 붙여넣을 위치 | 지금 제출 가능? |
| --- | :---: | --- | --- |
| **P1-BASE** | YC | `_baselines/P1-BASE/` | ✅ 학습 종료 — 지금 |
| **P1-ANNEAL** | YJ | `p1_temperature/P1-ANNEAL/` | ✅ 완주 — 지금 |
| **P2-BASE** | (로컬) | `_baselines/P2-BASE/` | ✅ 완주 |
| **P2-H128** | (로컬) | `p2_capacity/P2-H128/` | ✅ 완주 |
| **P1-H16** | SY | `p1_capacity/P1-H16/` | ⏳ ~15k/20k 진행 중 |
| **P2-FIXT** | NY | `p2_temperature/P2-FIXT/` | ⏳ ~2~3k/20k 진행 중 |
| **P2-B1080** | HJ | `p2_blockrange/P2-B1080/` | ⏳ ~4k/20k 진행 중 |

P1-BASE·P2-BASE는 여러 캠페인의 공통 기준 arm이라 `_baselines/`에 둔다.
P1-OLD·P2-OLD는 **구 데이터로 학습돼 비교 대상이 아니므로 폴더를 만들지 않는다.**

## 제출은 2단계

1. **1차 (필수, arm당 ~6.5 MB)** — `.pt` 없이 위 목록. 중앙이 `validation_parent_history.csv`로
   best-rate 상위 checkpoint 후보를 지목한다.
2. **2차** — 지목된 `.pt`만. (Phase 2 h128 3.0 MB / h16 128 KB, Phase 1 1.1 MB)

Phase 1은 `grid_contract.json`이 없고 `validation/` 대신 루트의 `validation_summary.csv`(2.6 MB)를 낸다.

## 절대 넣지 않는 것

`best_action_table.jsonl`(529 MB) · `candidate_summary.csv`(1.9 GB) ·
`validation_candidate_summary.csv`(1.2 GB) · Phase 2 `validation*/evaluations/**`(run 용량의 97%) ·
`problems/**/jobs.csv`. `.gitignore`가 이 경로들을 차단하므로 실수로 `git add`해도 들어가지 않는다.

## 제출 전 자가 점검

```bash
git status --porcelain   # 비어 있어야 한다 — dirty tree run은 비교 불가
python -c "import json;m=json.load(open('experiments/<EXPERIMENT-ID>/<ALIAS>/run_manifest.json'));\
print(m['identity']['arm_label'], m['code']['git']['dirty'], m['code']['code_fingerprint']['digest'][:12])"
```

`arm_label`이 별칭과 다르거나 `dirty`가 `True`면 그 run은 비교에 쓸 수 없다. 즉시 알린다.
