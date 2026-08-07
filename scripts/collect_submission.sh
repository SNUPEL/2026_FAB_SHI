#!/usr/bin/env bash
# collect_submission.sh — 학습 결과 제출물 수집기 (Linux / macOS)
#
# 타 하드웨어에서 돌린 학습 run 중 "제출해야 할 파일만" 골라 한 폴더에 모으고 압축한다.
# GB급 감사 로그는 자동으로 제외한다. git 없이 파일만 직접 전달할 때 쓴다.
#
#   ./scripts/collect_submission.sh <RUN_DIR> <ALIAS> <PHASE:1|2> [--with-checkpoints]
#
# 예)
#   ./scripts/collect_submission.sh output/phase1_anneal_v1 P1-ANNEAL 1
#   ./scripts/collect_submission.sh output/phase2_fixedT10  P2-FIXT   2 --with-checkpoints
#
# 결과: _submit/<ALIAS>/ 와 _submit/<ALIAS>.tar.gz
# 규칙 전문: 모델_학습_비교_절대규칙.md §4

set -euo pipefail

if [ $# -lt 3 ]; then
  sed -n '2,16p' "$0"; exit 1
fi

RUN="${1%/}"; ALIAS="$2"; PHASE="$3"; WITH_CK="${4:-}"
HEAD_LINES=200; TAIL_LINES=2000

[ -d "$RUN" ] || { echo "[ERROR] 학습 폴더를 찾을 수 없습니다: $RUN" >&2; exit 1; }
case "$PHASE" in 1|2) ;; *) echo "[ERROR] PHASE 는 1 또는 2" >&2; exit 1;; esac

OUT="_submit/$ALIAS"
mkdir -p "$OUT"
echo "[수집] $RUN  ->  $OUT"

copy_if() {  # copy_if <상대경로>
  local rel="$1" src="$RUN/$1"
  if [ -f "$src" ]; then
    mkdir -p "$OUT/$(dirname "$rel")"
    cp "$src" "$OUT/$rel"
    printf '  + %-38s %6s\n' "$rel" "$(du -h "$src" | cut -f1)"
  else
    printf '  - %-38s 없음\n' "$rel"
  fi
}

# ---- 1. 지표·증빙 파일 -------------------------------------------------
if [ "$PHASE" = "1" ]; then
  # Phase 1: 전부 run 루트에 있다. validation/ 디렉터리가 없다.
  for f in validation_summary.csv metrics.csv summary.json run_manifest.json \
           subproblem_metrics.csv training_quick_status.json; do
    copy_if "$f"
  done
else
  for f in run_manifest.json summary.json metrics.csv; do copy_if "$f"; done
  for d in validation validation_generalization; do
    for f in grid_contract.json validation_parent_history.csv validation_rank_history.csv; do
      copy_if "$d/$f"
    done
  done
fi

# ---- 2. 로그 앞/뒤 ------------------------------------------------------
# manifest 없는 run은 로그 헤더가 seed/블록범위/hidden_dim 의 유일한 근거다.
found_log=0
for lg in "$RUN"/launch_stdout.log "$RUN"/launch_stderr.log "$RUN"/*.log "$RUN.log"; do
  [ -f "$lg" ] || continue
  base="$(basename "${lg%.log}")"
  head -n "$HEAD_LINES" "$lg" > "$OUT/$base.head.log"
  tail -n "$TAIL_LINES" "$lg" > "$OUT/$base.tail.log"
  printf '  + %-38s head %s / tail %s 줄\n' "$base.head/.tail.log" "$HEAD_LINES" "$TAIL_LINES"
  found_log=1
done
if [ "$found_log" = "0" ]; then
  echo "  ! 로그를 찾지 못했습니다. 로그 앞 ${HEAD_LINES}줄이 없으면" >&2
  echo "    seed / --min-blocks / --max-blocks / hidden_dim 을 확인할 수 없어 비교가 무효입니다." >&2
fi

# ---- 3. checkpoint -----------------------------------------------------
if [ "$WITH_CK" = "--with-checkpoints" ] && [ -d "$RUN/checkpoints" ]; then
  mkdir -p "$OUT/checkpoints"
  cp "$RUN"/checkpoints/*.pt "$OUT/checkpoints/" 2>/dev/null || true
  printf '  + checkpoints/  %s개  %s\n' \
    "$(ls "$OUT"/checkpoints/*.pt 2>/dev/null | wc -l)" "$(du -sh "$OUT/checkpoints" | cut -f1)"
elif [ -d "$RUN/checkpoints" ]; then
  printf '  . checkpoints/ %s개 — 제외했습니다. 전부 보내려면 --with-checkpoints 를 붙이세요.\n' \
    "$(ls "$RUN"/checkpoints/*.pt 2>/dev/null | wc -l)"
fi
# 최종·best 모델은 항상 참고용으로 포함 (작다)
for m in phase1_pair_pointer.pt phase1_pair_pointer_best.pt phase2_best.pt phase2_batch_machine_policy.pt; do
  [ -f "$RUN/$m" ] && { cp "$RUN/$m" "$OUT/_selfbest_$m"; printf '  + _selfbest_%s\n' "$m"; }
done

# ---- 4. INFO.txt 양식 --------------------------------------------------
if [ ! -f "$OUT/INFO.txt" ]; then
  commit="$(git rev-parse HEAD 2>/dev/null || echo 미상)"
  dirty="$( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo 예 || echo 아니오 )"
  cat > "$OUT/INFO.txt" <<EOF
별칭        : $ALIAS
학습 경로   : $RUN
최종 episode:                       완주 여부: 예/아니오
git commit  : $commit
dirty       : $dirty
실행 CLI    :
              (실제로 실행한 명령 전체를 한 줄로. 기억나지 않으면 로그 헤더에서 옮겨 적으세요)
비고        :
EOF
  echo "  + INFO.txt  ← 최종 episode / 실행 CLI 를 직접 채워주세요"
fi

# ---- 5. 체크섬 + 압축 ---------------------------------------------------
( cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS.txt )
tar czf "_submit/$ALIAS.tar.gz" -C _submit "$ALIAS"

echo
echo "[완료] $OUT ($(du -sh "$OUT" | cut -f1))  ->  _submit/$ALIAS.tar.gz ($(du -h "_submit/$ALIAS.tar.gz" | cut -f1))"
echo "INFO.txt 의 '최종 episode' 와 '실행 CLI' 를 채운 뒤 전달하세요."
echo
echo "제외된 것(의도적): validation_candidate_summary.csv, candidate_summary.csv,"
echo "  best_action_table.jsonl, *.png, validation*/problems, validation*/evaluations"
