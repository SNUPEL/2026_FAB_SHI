#!/usr/bin/env bash
# Phase 1 / Phase 2 학습 속도 캘리브레이션.
# 이전 20k 학습(output/phase1_pair_20k, output/phase2_upstream_wo)의 설정을 그대로 재현하되
# 에피소드 수만 줄여 현재 브랜치의 sec/ep 를 잰다. 두 phase 는 순차 실행한다(단독 속도 측정).
set -u

PY="${PY:-/home/temp_id/anaconda3/envs/accord_env/bin/python}"
EPISODES="${EPISODES:-3}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"
OUT_ROOT="${OUT_ROOT:-output/bench}"
RESULT="${OUT_ROOT}/calibration_result.txt"

mkdir -p "${OUT_ROOT}"
: > "${RESULT}"

record() {  # name, seconds, episodes
  local name="$1" secs="$2" eps="$3"
  awk -v n="$name" -v s="$secs" -v e="$eps" \
    'BEGIN{printf "%s\ttotal=%.1fs\tepisodes=%d\tsec_per_ep=%.1f\tep_per_min=%.2f\n", n, s, e, s/e, e/(s/60)}' \
    | tee -a "${RESULT}"
}

echo "=== Phase 1 calibration (device=${DEVICE}, episodes=${EPISODES}) ==="
p1_dir="${OUT_ROOT}/calib_p1_${DEVICE}"
rm -rf "${p1_dir}"
start=$(date +%s)
"${PY}" main.py phase1-train-pair-self-labeling \
  --config config_mixed.yaml \
  --episodes "${EPISODES}" \
  --rollout-samples 64 \
  --heuristic-algorithms wo_first_balanced,cut_first_balanced,bevel_first_balanced \
  --objective-scope shared_and_series \
  --min-blocks 12 --max-blocks 80 \
  --device "${DEVICE}" \
  --seed 0 \
  --output-dir "${p1_dir}" \
  > "${p1_dir}.log" 2>&1
p1_rc=$?
end=$(date +%s)
p1_secs=$((end - start))
if [ ${p1_rc} -ne 0 ]; then
  echo "PHASE1 FAILED rc=${p1_rc} — tail of log:" | tee -a "${RESULT}"
  tail -20 "${p1_dir}.log" | tee -a "${RESULT}"
else
  record "phase1_${DEVICE}_solo" "${p1_secs}" "${EPISODES}"
fi

echo "=== Phase 2 calibration (device=${DEVICE}, workers=${WORKERS}, episodes=${EPISODES}) ==="
p2_dir="${OUT_ROOT}/calib_p2_${DEVICE}_w${WORKERS}"
rm -rf "${p2_dir}"
start=$(date +%s)
"${PY}" main.py phase2-train-batch-machine-self-labeling \
  --config config_mixed.yaml \
  --episodes "${EPISODES}" \
  --rollout-samples 64 \
  --candidate-workers "${WORKERS}" \
  --hidden-dim 128 --lr 0.001 \
  --min-blocks 12 --max-blocks 80 \
  --device "${DEVICE}" \
  --seed 0 \
  --output-dir "${p2_dir}" \
  > "${p2_dir}.log" 2>&1
p2_rc=$?
end=$(date +%s)
p2_secs=$((end - start))
if [ ${p2_rc} -ne 0 ]; then
  echo "PHASE2 FAILED rc=${p2_rc} — tail of log:" | tee -a "${RESULT}"
  tail -20 "${p2_dir}.log" | tee -a "${RESULT}"
else
  record "phase2_${DEVICE}_w${WORKERS}_solo" "${p2_secs}" "${EPISODES}"
fi

echo "=== done. result: ${RESULT} ==="
