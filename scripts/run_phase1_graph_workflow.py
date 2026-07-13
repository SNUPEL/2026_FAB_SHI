"""Run Phase 1 graph workflow on one actual workday.

Example:
    python scripts/run_phase1_graph_workflow.py \
      --wo-xlsx input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx \
      --workday 20260331 \
      --bay-ids 22,23,24 \
      --heuristic-algorithms all8 \
      --output-dir output/phase1_graph_workflow_20260331
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Phase1.orchestrator import run_phase1_graph_workflow
from Phase1.self_labeling import PHASE1_SELF_LABEL_HEURISTIC_BANK
from Utils.phase1.phase1_episode_dataset import build_phase1_actual_workday_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 graph workflow and export debug artifacts.")
    parser.add_argument("--wo-xlsx", required=True, help="Actual W/O Excel/CSV path")
    parser.add_argument("--workday", required=True, help="One 08:00 workday, e.g. 20260331")
    parser.add_argument("--bay-ids", default="22,23,24", help="Comma-separated Bay IDs")
    parser.add_argument("--gyel", default="NP")
    parser.add_argument("--heuristic-algorithms", default="all8", help="all8 or comma-separated heuristic names")
    parser.add_argument("--score-mode", default="steel_first", choices=["steel_first"])
    parser.add_argument("--no-long-cut-hard-mask", action="store_true")
    parser.add_argument(
        "--require-complete-block-workday",
        action="store_true",
        help="Use only blocks whose every W/O starts and ends inside the same 08:00 workday.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    bay_ids = [item.strip() for item in args.bay_ids.split(",") if item.strip()]
    heuristic_algorithms = _resolve_heuristics(args.heuristic_algorithms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[CHECK][run_phase1_graph_workflow] "
        f"wo_xlsx={args.wo_xlsx} workday={args.workday} bay_ids={bay_ids} "
        f"heuristics={','.join(heuristic_algorithms)} output_dir={output_dir}"
    )

    problems = build_phase1_actual_workday_jobs(
        source_path=args.wo_xlsx,
        workdays=[args.workday],
        bay_ids=bay_ids,
        gyel=args.gyel,
        require_complete_block_workday=args.require_complete_block_workday,
    )
    if len(problems) != 1:
        print(f"[ERROR][run_phase1_graph_workflow] cause=expected_one_problem count={len(problems)}")
        raise RuntimeError(f"expected one Phase 1 problem, got {len(problems)}")

    problem = problems[0]
    result = run_phase1_graph_workflow(
        jobs=problem["jobs"],
        bay_ids=bay_ids,
        heuristic_algorithms=heuristic_algorithms,
        score_mode=args.score_mode,
        long_cut_hard_mask=not args.no_long_cut_hard_mask,
    )
    summary = {
        "workday": args.workday,
        "job_count": problem["job_count"],
        "block_count": problem["block_count"],
        "bay_ids": bay_ids,
        "candidate_count": result["candidate_count"],
        "best_source": result["best_source"],
        "best_score": result["plan"]["score"],
        "long_cut_hard_mask": not args.no_long_cut_hard_mask,
        "require_complete_block_workday": args.require_complete_block_workday,
    }

    _write_json(output_dir / "phase1_graph.json", result["graph"])
    _write_json(output_dir / "phase1_plan.json", result["plan"])
    _write_json(output_dir / "phase1_summary.json", summary)
    _write_candidates_csv(output_dir / "phase1_candidates.csv", result["candidates"])
    _write_assignments_csv(output_dir / "phase1_plan_assignments.csv", result["plan"]["assignments"])

    print(
        "[VALIDATION][run_phase1_graph_workflow] "
        f"passed=true workday={args.workday} blocks={summary['block_count']} "
        f"candidates={summary['candidate_count']} best_source={summary['best_source']} "
        f"score={summary['best_score']} output_dir={output_dir}"
    )


def _resolve_heuristics(value: str) -> Sequence[str]:
    if value == "all8":
        return PHASE1_SELF_LABEL_HEURISTIC_BANK
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        print("[ERROR][run_phase1_graph_workflow._resolve_heuristics] cause=no_heuristics")
        raise RuntimeError("at least one heuristic is required")
    unknown = sorted(set(items) - set(PHASE1_SELF_LABEL_HEURISTIC_BANK))
    if unknown:
        print(f"[ERROR][run_phase1_graph_workflow._resolve_heuristics] cause=unknown_heuristics unknown={unknown}")
        raise RuntimeError(f"unknown heuristics: {unknown}")
    return tuple(items)


def _write_json(path: Path, payload: Mapping) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_candidates_csv(path: Path, candidates: Sequence[Mapping]) -> None:
    rows = []
    for candidate in candidates:
        score = list(candidate["score"])
        rows.append(
            {
                "source": candidate["source"],
                "assignment_count": candidate["assignment_count"],
                "transition_count": candidate["transition_count"],
                "score_0_steel_gap": score[0],
                "score_1_cut_gap": score[1],
                "score_2_bevel_gap": score[2],
                "long_cut_bay24_count": candidate.get("long_cut_bay24_count", ""),
            }
        )
    _write_csv(path, rows)


def _write_assignments_csv(path: Path, assignments: Sequence[Mapping]) -> None:
    _write_csv(path, assignments)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        print(f"[ERROR][run_phase1_graph_workflow._write_csv] cause=no_rows path={path}")
        raise RuntimeError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
