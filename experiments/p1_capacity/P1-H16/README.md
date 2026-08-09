# P1-H16 — 제출 폴더

**여기에 파일을 붙여넣으면 된다.** 규칙 전문은 [모델_학습_비교_절대규칙.md](../../../모델_학습_비교_절대규칙.md) §4.

| | |
| --- | --- |
| 별칭 | **P1-H16** |
| Phase | 1 |
| 담당 (`--owner`) | **SY** |
| 원 학습 경로 | `❓ 확인 필요 — 담당자가 `INFO.txt`에 적을 것` |
| 캠페인 / 비교 축 | `hidden_dim`: P1-BASE(h128) 대비 **h16** |
| 학습 상태 | 🔄 약 15,000 / 20,000 ep 진행 중 — **완주 후 제출** |

P1-BASE에서 `--hidden-dim`만 128 → 16으로 바꾼 arm. 나머지 설정은 P1-BASE와 같아야 한다.

---

## 1차 제출 — 여기에 넣을 파일 (Phase 1)

| 파일 | 필수 | 비고 |
| --- | :---: | --- |
| `validation_summary.csv` | ✅ | **가장 중요.** Phase 1의 유일한 비교 등급 산출물. UTF-8 BOM. 2.6~8.7 MB |
| `metrics.csv` | ✅ | 학습 곡선. 1.4~4.6 MB |
| `summary.json` | ✅ | 완주 run만 생성된다 |
| `run_manifest.json` | ✅* | **2026-08-03 이후 시작한 run만 존재.** 없으면 아래 "manifest 없을 때" |
| `subproblem_metrics.csv` | 권장 | `NP_NC`/`FN_FL` 분해 |
| `training_quick_status.json` | 권장 | 1 KB |
| `run.log.head` (앞 200줄) | ✅ | **시작 헤더의 실제 CLI 플래그.** manifest가 없으면 유일한 근거다 |
| `run.log.tail` (뒤 2000줄) | 권장 | 종료 상태 |
| `checkpoints/*.pt` | ⏳ | **지금 넣지 않는다.** 아래 "2차 제출" 참조 |

`.csv`는 gzip으로 넣어도 된다(`validation_summary.csv.gz`). 압축하면 합계 3 MB 안팎.

## 넣지 말 것

`best_action_table.jsonl`(529 MB) · `candidate_summary.csv`(1.9 GB) ·
`validation_candidate_summary.csv`(1.2 GB) · `evaluations/**` · `problems/**` · PNG 전량 ·
`checkpoints/` 전량. 앞의 것들은 `.gitignore`가 막으므로 실수로 `git add`해도 들어가지 않는다.

## 2차 제출 — checkpoint

**1차 제출을 먼저 한다.** 중앙에서 `validation_summary.csv`를 읽어
best-rate 상위 후보 3개를 지목하면, 그 `.pt`만 `checkpoints/P1-H16_ep<NNNNN>.pt`로 넣는다.
개당 약 1.1 MB.

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
du -sh experiments/p1_capacity/P1-H16          # 50 MB 이하여야 한다
```
