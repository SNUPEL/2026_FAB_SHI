"""Phase 2 validation/train 의 Bay·계열군별 policy-vs-teacher gap 진단.

목적: "NP∙NC(Bay 22/23/24)가 FN∙FL(Bay 25/trans)보다 검증 성능이 일관되게 좋다"는
관측이 (a) 모델 학습 gap 인지 (b) 문제 난이도/척도 차이인지 (c) train 은 균형인데
val 만 벌어지는 일반화(과적합) gap 인지를 데이터로 구분한다.

읽는 파일 (둘 다 현재 스키마 기준):
  --val   output/<run>/validation/validation_bay_history.csv
            per-Bay 로 best(teacher)/best_heuristic/proposed(배포 정책)/greedy 점수 기록.
  --train output/<run>/subproblem_metrics.csv
            per-Bay 로 best(teacher) 점수 + CE loss 기록 (학습측).

score tuple 순서(사전식):
  [hard_violation, makespan, cut_length_gap, wo_count_gap, bevel_quantity_gap, occupancy_gap]

torch 불필요 — 표준 라이브러리만 사용하므로 아무 python 으로나 돌아간다.
계열군 매핑은 Phase2.merged.MIXED_PHASE2_ELIGIBLE_FAMILIES 와 반드시 일치해야 한다
(여기서는 import 시 torch 를 끌어오지 않으려고 로컬 상수로 두되, 불일치 시 fail-fast).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

# Windows 콘솔(cp949)에서도 한글 판정문이 깨지지 않도록 stdout 을 UTF-8 로 고정.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Phase2.merged.MIXED_PHASE2_ELIGIBLE_FAMILIES 와 동일해야 한다.
BAY_FAMILY_GROUP = {
    "22": "NP_NC",
    "23": "NP_NC",
    "24": "NP_NC",
    "25": "FN_FL",
    "trans": "FN_FL",
}
GROUP_ORDER = ("NP_NC", "FN_FL")

# score tuple index
HARD, MAKESPAN, CUT_GAP, WO_GAP, BEVEL_GAP, OCC_GAP = range(6)
TERM_NAMES = ("hard", "makespan", "cut_gap", "wo_gap", "bevel_gap", "occ_gap")


def _group_of(bay_id: str) -> str:
    if bay_id not in BAY_FAMILY_GROUP:
        raise RuntimeError(
            f"[phase2_bay_teacher_gap] unknown bay_id={bay_id!r}; "
            f"expected one of {sorted(BAY_FAMILY_GROUP)} "
            f"(BAY_FAMILY_GROUP 가 MIXED_PHASE2_ELIGIBLE_FAMILIES 와 어긋났는지 확인)"
        )
    return BAY_FAMILY_GROUP[bay_id]


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"[phase2_bay_teacher_gap] empty csv: {path}")
    if "bay_id" not in rows[0]:
        raise RuntimeError(
            f"[phase2_bay_teacher_gap] {path} 에 bay_id 컬럼이 없음 "
            f"(구버전 스키마). 현재 코드로 생성한 validation_bay_history.csv / "
            f"subproblem_metrics.csv 가 필요함. header={list(rows[0])[:8]}..."
        )
    return rows


def _score(cell: str) -> List[float] | None:
    if cell is None or cell == "":
        return None
    return [float(x) for x in json.loads(cell)]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _rel_gap(proposed: float, teacher: float) -> float:
    """teacher 대비 상대 초과율. teacher==0 이면 절대차(0이면 0)."""
    denom = abs(teacher)
    if denom < 1e-9:
        return 0.0 if abs(proposed - teacher) < 1e-9 else float("inf")
    return (proposed - teacher) / denom


# --------------------------------------------------------------------------- #
# validation 측 집계
# --------------------------------------------------------------------------- #
def analyze_val(path: str, latest_only: bool) -> None:
    rows = _read_rows(path)
    episodes = sorted({int(r["train_episode"]) for r in rows})
    if latest_only:
        keep = {episodes[-1]}
        rows = [r for r in rows if int(r["train_episode"]) in keep]
        scope = f"train_episode={episodes[-1]} (latest checkpoint)"
    else:
        scope = f"train_episodes={episodes[0]}..{episodes[-1]} (all)"

    print(f"\n=== VALIDATION per-family gap  [{path}] ===")
    print(f"    scope: {scope}   rows={len(rows)}")

    agg: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {
            "makespan_relgap": [],  # proposed vs teacher(best)
            "match_teacher": [],  # proposed tuple == teacher tuple (사전식 동일)
            "vs_heur": [],  # win/tie/loss vs best_heuristic
            "agent_is_teacher": [],  # best_source 가 agent 인가
            "teacher_makespan": [],  # 난이도 프록시(teacher 절대 makespan)
            "teacher_hard": [],  # teacher hard 위반>0 비율(문제 난이도)
            "proposed_hard": [],  # 정책 hard 위반>0 비율
        }
    )

    for r in rows:
        g = _group_of(r["bay_id"])
        best = _score(r["best_score_json"])
        prop = _score(r["proposed_best_score_json"])
        a = agg[g]
        a["teacher_makespan"].append(best[MAKESPAN])
        a["teacher_hard"].append(1.0 if best[HARD] > 0 else 0.0)
        a["agent_is_teacher"].append(1.0 if r["best_source"].startswith("agent") else 0.0)
        rel = r.get("proposed_vs_best_heuristic", "")
        a["vs_heur"].append(rel)
        if prop is None:
            a["proposed_hard"].append(float("nan"))
            continue
        a["proposed_hard"].append(1.0 if prop[HARD] > 0 else 0.0)
        a["makespan_relgap"].append(_rel_gap(prop[MAKESPAN], best[MAKESPAN]))
        a["match_teacher"].append(1.0 if prop == best else 0.0)

    hdr = (
        f"{'group':<7} {'n':>4} {'mksp_relgap':>12} {'match_teacher':>14} "
        f"{'win/tie/loss':>16} {'agent=teacher':>14} {'teacher_mksp':>13} {'teacher_hard%':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for g in GROUP_ORDER:
        a = agg.get(g)
        if not a:
            continue
        vs = a["vs_heur"]
        win = 100 * sum(1 for x in vs if x == "win") / len(vs)
        tie = 100 * sum(1 for x in vs if x == "tie") / len(vs)
        loss = 100 * sum(1 for x in vs if x == "loss") / len(vs)
        print(
            f"{g:<7} {len(vs):>4} "
            f"{_mean(a['makespan_relgap'])*100:>11.2f}% "
            f"{_mean(a['match_teacher'])*100:>13.1f}% "
            f"{win:>5.0f}/{tie:>3.0f}/{loss:<5.0f} "
            f"{_mean(a['agent_is_teacher'])*100:>13.1f}% "
            f"{_mean(a['teacher_makespan']):>13.1f} "
            f"{_mean(a['teacher_hard'])*100:>12.1f}%"
        )
    _verdict_val(agg)


def _verdict_val(agg: Dict[str, Dict[str, list]]) -> None:
    if not all(g in agg for g in GROUP_ORDER):
        return
    np_gap = _mean(agg["NP_NC"]["makespan_relgap"])
    fn_gap = _mean(agg["FN_FL"]["makespan_relgap"])
    np_lvl = _mean(agg["NP_NC"]["teacher_makespan"])
    fn_lvl = _mean(agg["FN_FL"]["teacher_makespan"])
    print("\n  판정 힌트:")
    print(
        f"   - teacher-gap(makespan 상대): NP_NC={np_gap*100:.2f}%  FN_FL={fn_gap*100:.2f}%"
        f"   Δ={((fn_gap-np_gap))*100:+.2f}%p"
    )
    if math.isfinite(np_gap) and math.isfinite(fn_gap):
        if fn_gap > np_gap + 0.02:
            print("     → FN_FL 이 teacher 를 유의하게 못 따라감 = **학습(모델) gap**. reweighting 검토 대상.")
        else:
            print("     → 두 군의 teacher-gap 유사 = 모델은 양쪽 다 비슷하게 최적을 따라감.")
    print(
        f"   - teacher 절대 makespan 수준: NP_NC={np_lvl:.1f}  FN_FL={fn_lvl:.1f}"
    )
    if math.isfinite(np_lvl) and math.isfinite(fn_lvl) and fn_lvl > np_lvl * 1.05:
        print("     → FN_FL 의 도달가능 최적 자체가 나쁨 = **문제 난이도/척도 차이**. 절대점수 비교는 오해 소지.")


# --------------------------------------------------------------------------- #
# train 측 집계 (subproblem_metrics.csv)
# --------------------------------------------------------------------------- #
def analyze_train(path: str, last_frac: float) -> None:
    rows = _read_rows(path)
    episodes = sorted({int(r["episode"]) for r in rows})
    cut = episodes[int(len(episodes) * (1 - last_frac))]
    rows = [r for r in rows if int(r["episode"]) >= cut]
    print(f"\n=== TRAIN per-family (subproblem_metrics)  [{path}] ===")
    print(f"    scope: episode>={cut} (last {last_frac:.0%})   rows={len(rows)}")

    agg: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"loss": [], "agent_is_teacher": [], "teacher_makespan": [], "job": []}
    )
    for r in rows:
        g = _group_of(r["bay_id"])
        a = agg[g]
        a["loss"].append(float(r["loss"]))
        a["agent_is_teacher"].append(1.0 if r["best_source"].startswith("agent") else 0.0)
        a["teacher_makespan"].append(_score(r["score_json"])[MAKESPAN])
        a["job"].append(float(r["job_count"]))

    hdr = f"{'group':<7} {'n':>5} {'mean_loss':>10} {'agent=teacher':>14} {'teacher_mksp':>13} {'jobs/subprob':>13}"
    print(hdr)
    print("-" * len(hdr))
    for g in GROUP_ORDER:
        a = agg.get(g)
        if not a:
            continue
        print(
            f"{g:<7} {len(a['loss']):>5} {_mean(a['loss']):>10.4f} "
            f"{_mean(a['agent_is_teacher'])*100:>13.1f}% "
            f"{_mean(a['teacher_makespan']):>13.1f} {_mean(a['job']):>13.1f}"
        )
    if all(g in agg for g in GROUP_ORDER):
        np_loss = _mean(agg["NP_NC"]["loss"])
        fn_loss = _mean(agg["FN_FL"]["loss"])
        print("\n  판정 힌트:")
        print(f"   - train CE loss: NP_NC={np_loss:.4f}  FN_FL={fn_loss:.4f}  Δ={fn_loss-np_loss:+.4f}")
        if fn_loss > np_loss * 1.15:
            print("     → FN_FL 이 train 에서도 loss 높음 = train 자체가 underfit = gradient/capacity 불균형. reweighting 이 지렛대.")
        else:
            print("     → train loss 는 균형. val 에서만 벌어지면 일반화(과적합) gap 이거나 난이도 차이.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 Bay/계열군별 policy-vs-teacher gap 진단")
    ap.add_argument("--val", nargs="*", default=[], help="validation_bay_history.csv (glob 허용)")
    ap.add_argument("--train", nargs="*", default=[], help="subproblem_metrics.csv (glob 허용)")
    ap.add_argument("--all-episodes", action="store_true", help="val 을 최신 체크포인트만이 아니라 전 구간 집계")
    ap.add_argument("--train-last-frac", type=float, default=0.2, help="train 은 마지막 몇 %% episode 만 집계")
    args = ap.parse_args()

    if not args.val and not args.train:
        ap.error("--val 또는 --train 중 최소 하나는 필요")

    for pattern in args.train:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            analyze_train(path, args.train_last_frac)
    for pattern in args.val:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            analyze_val(path, latest_only=not args.all_episodes)


if __name__ == "__main__":
    main()
