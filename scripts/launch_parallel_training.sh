#!/usr/bin/env bash
# Phase 1 / Phase 2 동시 학습 기동.
# Phase 1 은 워커 없이 단일 프로세스, Phase 2 는 --candidate-workers 로 후보 생성을 병렬화한다.
# 두 학습은 서로 독립이다: Phase 2 는 에피소드마다 자체 휴리스틱으로 Bay 배정을 만들므로
# Phase 1 의 산출물을 기다리지 않는다.
#
# 세션이 끊겨도 살아남도록 setsid + nohup 으로 완전히 분리해 띄운다.
set -eu

PY="${PY:-/home/temp_id/anaconda3/envs/accord_env/bin/python}"
STAMP="${STAMP:-260728}"
EPISODES="${EPISODES:-20000}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"

P1_OUT="output/phase1_run_${STAMP}"
P2_OUT="output/phase2_run_${STAMP}"

for d in "${P1_OUT}" "${P2_OUT}"; do
  if [ -e "${d}" ]; then
    echo "[ERROR][launch_parallel_training] cause=output_dir_exists path=${d}"
    echo "기존 학습을 덮어쓰지 않는다. STAMP 를 바꾸거나 해당 디렉터리를 옮겨라."
    exit 1
  fi
done

echo "=== Phase 1 기동 (워커 없음) ==="
setsid nohup "${PY}" main.py phase1-train-pair-self-labeling \
  --config config_mixed.yaml \
  --episodes "${EPISODES}" \
  --rollout-samples 64 \
  --heuristic-algorithms wo_first_balanced,cut_first_balanced,bevel_first_balanced \
  --objective-scope shared_and_series \
  --min-blocks 12 --max-blocks 80 \
  --device "${DEVICE}" \
  --seed 0 \
  --checkpoint-every 100 \
  --validation-every 100 \
  --validation-episodes 20 \
  --output-dir "${P1_OUT}" \
  > "${P1_OUT}.log" 2>&1 < /dev/null &
P1_PID=$!
echo "  PID ${P1_PID} -> ${P1_OUT}"

echo "=== Phase 2 기동 (candidate-workers ${WORKERS}) ==="
setsid nohup "${PY}" main.py phase2-train-batch-machine-self-labeling \
  --config config_mixed.yaml \
  --episodes "${EPISODES}" \
  --rollout-samples 64 \
  --candidate-workers "${WORKERS}" \
  --hidden-dim 128 --lr 0.001 \
  --min-blocks 12 --max-blocks 80 \
  --device "${DEVICE}" \
  --seed 0 \
  --checkpoint-every 100 \
  --validation-every 100 \
  --validation-episodes 20 \
  --output-dir "${P2_OUT}" \
  > "${P2_OUT}.log" 2>&1 < /dev/null &
P2_PID=$!
echo "  PID ${P2_PID} -> ${P2_OUT}"

printf '%s\n' "${P1_PID}" > "${P1_OUT}.pid"
printf '%s\n' "${P2_PID}" > "${P2_OUT}.pid"
echo "=== 기동 완료. 중단하려면: kill \$(cat ${P1_OUT}.pid) \$(cat ${P2_OUT}.pid) ==="
