# P2-BASE — 제출 폴더

**여기에 파일을 붙여넣으면 된다.** 규칙 전문은 [모델_학습_비교_절대규칙.md](../../../모델_학습_비교_절대규칙.md) §4.

| | |
| --- | --- |
| 별칭 | **P2-BASE** |
| Phase | 2 |
| 담당 (`--owner`) | **(로컬)** |
| 원 학습 경로 | `output/phase2_18merge_tempanneal_w8` |
| 캠페인 / 비교 축 | p2_capacity·p2_temperature·p2_blockrange 공통 기준 arm |
| 학습 상태 | ✅ 20,000 ep 완주 |

이 머신 `output/`에 원본이 있다. 로컬에서 복사하면 된다.

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
best-rate 상위 후보 3개를 지목하면, 그 `.pt`만 `checkpoints/P2-BASE_ep<NNNNN>.pt`로 넣는다.
개당 약 3.0 MB(h128) / 128 KB(h16).

**run 자체 `best.pt`는 보내지 않아도 된다.** 그 파일은 이 캠페인과 다른 기준으로 뽑힌 것이라
쓰지 않는다(규칙 A7). 궁금하면 참고용으로 함께 넣되 파일명에 `_selfbest`를 붙인다.

## ⚠️ 이 run은 `run_manifest.json`이 없을 것이다

manifest 도입(2026-08-03) 이전에 시작된 run이다. 그러면 절대규칙 §4.7의 **대체 증빙**을 대신 넣는다.
`summary.json` + `run.log.head`(앞 200줄) + checkpoint 내부 메타 + 아래 텍스트 4종이 **모두** 있어야
비교에 쓸 수 있다. 하나라도 없으면 비교 무효다.

`INFO.txt`를 이 폴더에 만들어 아래를 채운다.

```
별칭        :
학습 경로   :              (원본 그대로. Windows면 역슬래시 그대로)
최종 episode:              완주 여부: 예/아니오
git commit  :              dirty: 예/아니오        ← 모르면 "미상"
실행 CLI    :              (전체 명령 한 줄)
```

**`summary.json`에는 seed·블록범위·`--validation-episodes`가 들어 있지 않다.** 그 값은 로그 헤더에만
있으므로 `run.log.head`를 반드시 넣어야 한다. 없으면 문제집합 동일성을 확인할 수 없어 비교 무효다.

---

## 붙여넣은 뒤 확인

```bash
git status --porcelain experiments    # allowlist 밖 파일이 뜨면 중단
du -sh experiments/_baselines/P2-BASE          # 50 MB 이하여야 한다
```
