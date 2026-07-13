"""Export one Phase 1 Block-Bay graph state as JSON.

Example:
    python scripts/export_phase1_graph_state.py \
      --wo-xlsx input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx \
      --workday 20260331 \
      --bay-ids 22,23,24,25 \
      --output-json output/phase1_graph_20260331.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.phase1.phase1_episode_dataset import build_phase1_actual_workday_jobs
from Utils.learning.phase_graph_mdp import build_phase1_block_bay_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a variable-size Phase 1 graph state.")
    parser.add_argument("--wo-xlsx", required=True, help="Actual W/O Excel/CSV path")
    parser.add_argument("--workday", required=True, help="One 08:00 workday, e.g. 20260331")
    parser.add_argument("--bay-ids", default="22,23,24", help="Comma-separated Bay IDs")
    parser.add_argument("--gyel", default="NP")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-long-cut-hard-mask", action="store_true")
    parser.add_argument(
        "--require-complete-block-workday",
        action="store_true",
        help="Keep only blocks whose all W/Os start and end inside this 08:00 workday.",
    )
    args = parser.parse_args()

    bay_ids = [bay_id.strip() for bay_id in args.bay_ids.split(",") if bay_id.strip()]
    print(
        "[CHECK][export_phase1_graph_state] "
        f"wo_xlsx={args.wo_xlsx} workday={args.workday} bay_ids={bay_ids} output={args.output_json}"
    )
    problems = build_phase1_actual_workday_jobs(
        source_path=args.wo_xlsx,
        workdays=[args.workday],
        bay_ids=bay_ids,
        gyel=args.gyel,
        require_complete_block_workday=args.require_complete_block_workday,
    )
    if len(problems) != 1:
        print(f"[ERROR][export_phase1_graph_state] cause=expected_one_problem count={len(problems)}")
        raise RuntimeError(f"expected one Phase 1 problem, got {len(problems)}")

    graph = build_phase1_block_bay_graph(
        jobs=problems[0]["jobs"],
        bay_ids=bay_ids,
        long_cut_hard_mask=not args.no_long_cut_hard_mask,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[VALIDATION][export_phase1_graph_state] "
        f"passed=true blocks={graph['metadata']['block_count']} "
        f"edges={graph['metadata']['candidate_edge_count']} output={output_path}"
    )


if __name__ == "__main__":
    main()
