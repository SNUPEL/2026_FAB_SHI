#!/usr/bin/env bash
# Phase 1 append 전환의 속도 곡선을 측정한다.
# 기존 output/phase1_pair_20k 와 동일한 설정·시드로 300 에피소드를 돌리고,
# 100 에피소드 구간별 s/ep 가 증가하지 않는지 확인한다.
set -eu

PY="${PY:-/home/temp_id/anaconda3/envs/accord_env/bin/python}"
OUT="${OUT:-output/bench/phase1_append_300}"
EPISODES="${EPISODES:-300}"

rm -rf "${OUT}"
mkdir -p "$(dirname "${OUT}")"
start=$(date +%s)
"${PY}" main.py phase1-train-pair-self-labeling \
  --config config_mixed.yaml \
  --episodes "${EPISODES}" \
  --rollout-samples 64 \
  --heuristic-algorithms wo_first_balanced,cut_first_balanced,bevel_first_balanced \
  --objective-scope shared_and_series \
  --min-blocks 12 --max-blocks 80 \
  --device cuda \
  --seed 0 \
  --checkpoint-every 100 \
  --output-dir "${OUT}" \
  > "${OUT}.log" 2>&1
end=$(date +%s)

echo "총 소요: $(( end - start ))초 / ${EPISODES} 에피소드"
echo "구간별 s/ep (체크포인트 간격 기준):"
ls -l --time-style='+%s' "${OUT}"/checkpoints/*.pt 2>/dev/null | awk '{print $6, $7}' | sort -n | \
  awk '{ep=$2; gsub(/.*_ep0*/,"",ep); gsub(/\.pt/,"",ep);
        if(prev){printf "  ep %5d: %5.1f s/ep\n", ep, ($1-prev)/100} prev=$1}'
