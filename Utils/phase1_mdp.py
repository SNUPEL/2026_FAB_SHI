"""Phase 1 MDP/self-labeling trace builder.

This module converts the current Phase 1 block-to-Bay heuristic into a
supervised learning package. It does not train a policy by itself.

Decision model:
1. `SELECT_BLOCK`: choose one remaining block set.
2. `SELECT_BAY`: choose the Bay for that selected block.

The teacher label is the deterministic Phase 1 plan from
`Utils.phase1_bay_balancer`. This keeps the learning interface aligned with the
validated heuristic and avoids creating a second hidden planner.

입출력 요약:
- 입력: 환경 Job mapping, Bay 후보, teacher heuristic 이름.
- 출력: 사람이 보는 trace CSV row, 학습용 action JSONL row, feature 이름 manifest, 원본 Phase 1 plan.
- 사용처: `phase1-mdp-trace` CLI, imitation trainer의 입력 action table 생성.
"""

# LINE-BY-LINE: 미래 타입 힌트를 문자열로 늦게 평가합니다. 사용: Python annotation 호환성 유지.
from __future__ import annotations

# LINE-BY-LINE: trace CSV 저장에 사용합니다.
import csv
# LINE-BY-LINE: action table JSONL과 manifest JSON 저장에 사용합니다.
import json
# LINE-BY-LINE: 출력 파일 path를 OS 독립적으로 다루기 위해 사용합니다.
from pathlib import Path
# LINE-BY-LINE: 함수 입력/출력 타입을 명확히 표시하기 위한 typing import입니다.
from typing import Dict, List, Mapping, Sequence, Tuple

# LINE-BY-LINE: Phase 1 휴리스틱 plan 생성과 block/Bay 부하 helper를 재사용합니다.
from Utils.phase1_bay_balancer import (
    LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
    MULTI_OBJECTIVE_PHASE1_HEURISTIC,
    PRIORITY_GREEDY_PHASE1_HEURISTIC,
    PRIORITY_SWEEP_PHASE1_HEURISTIC,
    Phase1Block,
    _add_block_load,
    _collect_blocks,
    _normalize_algorithm,
    _normalize_bay_ids,
    build_phase1_bay_plan,
    write_phase1_bay_plan,
)


# LINE-BY-LINE: split-action MDP phase 순서입니다. 이 파일은 과거 `SELECT_BLOCK -> SELECT_BAY` trace용입니다.
PHASE1_DECISION_PHASES = ["SELECT_BLOCK", "SELECT_BAY"]
# LINE-BY-LINE: block 후보 feature 이름입니다. JSONL candidate feature dict의 key 순서 계약입니다.
PHASE1_BLOCK_FEATURE_NAMES = [
    # LINE-BY-LINE: block 단위 강재수량 합입니다.
    "steel_quantity_sum",
    # LINE-BY-LINE: block 단위 절단장 합입니다.
    "cut_length_sum",
    # LINE-BY-LINE: block 단위 베벨수량 합입니다.
    "bevel_quantity_sum",
    # LINE-BY-LINE: block에 속한 W/O 수입니다.
    "wo_count",
    # LINE-BY-LINE: 절단장 1000 초과 여부입니다.
    "long_cut_over_1000",
    # LINE-BY-LINE: 해당 block이 갈 수 있는 Bay 후보 개수입니다.
    "allowed_bay_count",
    # LINE-BY-LINE: 현재 step에서 아직 남아 있는 후보인지 표시합니다.
    "remaining_flag",
    # LINE-BY-LINE: 이미 배정된 block인지 표시합니다.
    "assigned_flag",
]
# LINE-BY-LINE: Bay 후보 feature 이름입니다. `SELECT_BAY` phase에서 candidate feature dict의 key 순서 계약입니다.
PHASE1_BAY_FEATURE_NAMES = [
    # LINE-BY-LINE: Bay ID를 숫자로 바꾼 값입니다. 예: 22.0.
    "bay_id_numeric",
    # LINE-BY-LINE: 현재 Bay에 누적된 강재수량입니다.
    "steel_quantity_sum",
    # LINE-BY-LINE: 현재 Bay에 누적된 절단장입니다.
    "cut_length_sum",
    # LINE-BY-LINE: 현재 Bay에 누적된 베벨수량입니다.
    "bevel_quantity_sum",
    # LINE-BY-LINE: 현재 Bay에 누적된 W/O 수입니다.
    "wo_count",
    # LINE-BY-LINE: 현재 Bay에 누적된 block 수입니다.
    "block_count",
    # LINE-BY-LINE: 장척 block이 Bay24로 간 누적 수입니다.
    "long_cut_bay24_count",
    # LINE-BY-LINE: 현재 선택 block이 이 Bay에 갈 수 있으면 1입니다.
    "candidate_allowed_flag",
]
# LINE-BY-LINE: 전역 환경 feature 이름입니다. 모든 후보가 공유하는 현재 부하 상태입니다.
PHASE1_ENV_FEATURE_NAMES = [
    # LINE-BY-LINE: 배정 완료 block 비율입니다.
    "progress_ratio",
    # LINE-BY-LINE: 남은 block 비율입니다.
    "remaining_block_ratio",
    # LINE-BY-LINE: 현재 Bay별 강재수량 max-min gap입니다.
    "steel_quantity_gap",
    # LINE-BY-LINE: 현재 Bay별 절단장 max-min gap입니다.
    "cut_length_gap",
    # LINE-BY-LINE: 현재 Bay별 베벨수량 max-min gap입니다.
    "bevel_quantity_gap",
    # LINE-BY-LINE: 현재까지 장척 block이 Bay24로 간 수입니다.
    "long_cut_bay24_count",
    # LINE-BY-LINE: 현재까지 배정한 block 수입니다.
    "assigned_block_count",
]
# LINE-BY-LINE: CSV trace에 저장할 column 이름입니다.
PHASE1_TRACE_FIELDS = [
    "step",
    "block_step",
    "phase",
    "selected_block_set_id",
    "selected_bay",
    "selected_action_index",
    "candidate_count",
    "remaining_block_count",
    "assigned_block_count",
    "env_progress_ratio",
    "env_remaining_block_ratio",
    "env_steel_quantity_gap",
    "env_cut_length_gap",
    "env_bevel_quantity_gap",
    "env_long_cut_bay24_count",
    "bay_loads_json",
    "candidate_ids",
]


# LINE-BY-LINE: Phase 1 heuristic plan을 학습 가능한 MDP trace/action table로 바꾸는 entry function입니다.
def build_phase1_mdp_trace_package(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str = LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
) -> Dict:
    """Build a JSON-serializable Phase 1 self-label package.

    Inputs:
    - `jobs`: environment Job mapping.
    - `bay_ids`: Phase 1 Bay candidates, e.g. `["22", "23", "24"]`.
    - `algorithm`: existing Phase 1 heuristic name.

    Output:
    - `trace_rows`: CSV-friendly decision trace.
    - `action_table_rows`: JSONL-friendly candidate table for imitation learning.
    - `manifest`: feature names and label-source metadata.
    - `plan`: the underlying Phase 1 block-to-Bay plan.
    """

    # LINE-BY-LINE: algorithm alias를 canonical heuristic 이름으로 정리합니다.
    normalized_algorithm = _normalize_algorithm(algorithm)
    # LINE-BY-LINE: Bay ID를 deterministic tuple로 정규화합니다.
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    # LINE-BY-LINE: multi-objective 계열 heuristic이면 강재/절단장/베벨수량 필드를 필수로 요구합니다.
    require_multi_objective = normalized_algorithm in {
        MULTI_OBJECTIVE_PHASE1_HEURISTIC,
        PRIORITY_SWEEP_PHASE1_HEURISTIC,
        PRIORITY_GREEDY_PHASE1_HEURISTIC,
        LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
    }
    # LINE-BY-LINE: W/O job들을 block 단위로 묶어 Phase1Block 목록을 만듭니다.
    blocks = _collect_blocks(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        require_multi_objective=require_multi_objective,
    )
    # LINE-BY-LINE: teacher heuristic으로 실제 block->Bay plan을 만듭니다.
    plan = build_phase1_bay_plan(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        algorithm=normalized_algorithm,
    )
    # LINE-BY-LINE: plan의 assignment row를 block_set_id -> assigned_bay dict로 바꿉니다.
    assignment_by_block = _assignment_by_block(plan, blocks)
    # LINE-BY-LINE: teacher가 block을 선택한 순서를 복원합니다.
    ordered_blocks = _teacher_block_order(blocks, plan, normalized_algorithm)

    # LINE-BY-LINE: Bay별 누적 부하를 0으로 초기화합니다.
    bay_loads = _empty_bay_loads(normalized_bay_ids)
    # LINE-BY-LINE: 아직 선택되지 않은 block 후보 dict입니다.
    remaining_blocks = {block.block_set_id: block for block in blocks}
    # LINE-BY-LINE: 사람이 보기 쉬운 CSV trace row를 누적합니다.
    trace_rows: List[Dict] = []
    # LINE-BY-LINE: trainer가 직접 읽는 JSONL action table row를 누적합니다.
    action_table_rows: List[Dict] = []
    # LINE-BY-LINE: 전체 decision step counter입니다. block 하나당 SELECT_BLOCK/SELECT_BAY 2 step이 생깁니다.
    step = 0

    for block_step, selected_block in enumerate(ordered_blocks):
        if selected_block.block_set_id not in remaining_blocks:
            print(
                "[ERROR][phase1_mdp.build_phase1_mdp_trace_package] "
                f"cause=duplicate_teacher_block block_set_id={selected_block.block_set_id}"
            )
            raise RuntimeError(f"duplicate_teacher_block: {selected_block.block_set_id}")

        block_candidates = sorted(remaining_blocks.values(), key=lambda block: block.block_set_id)
        selected_block_index = _candidate_index(
            [block.block_set_id for block in block_candidates],
            selected_block.block_set_id,
            "selected_block_set_id",
        )
        env_features = _env_features(
            bay_loads=bay_loads,
            assigned_block_count=block_step,
            total_block_count=len(blocks),
        )
        trace_rows.append(
            _trace_row(
                step=step,
                block_step=block_step,
                phase="SELECT_BLOCK",
                selected_block_set_id=selected_block.block_set_id,
                selected_bay="",
                selected_action_index=selected_block_index,
                candidates=[block.block_set_id for block in block_candidates],
                remaining_block_count=len(remaining_blocks),
                assigned_block_count=block_step,
                env_features=env_features,
                bay_loads=bay_loads,
            )
        )
        action_table_rows.append(
            {
                "step": step,
                "block_step": block_step,
                "phase": "SELECT_BLOCK",
                "selected_action_index": selected_block_index,
                "selected_block_set_id": selected_block.block_set_id,
                "selected_bay": "",
                "env_features": env_features,
                "bay_loads": _snapshot_bay_loads(bay_loads),
                "candidates": [
                    _block_candidate(block=block, assigned=False)
                    for block in block_candidates
                ],
            }
        )
        step += 1

        selected_bay = assignment_by_block[selected_block.block_set_id]
        bay_candidates = list(selected_block.allowed_bay_ids)
        selected_bay_index = _candidate_index(bay_candidates, selected_bay, "selected_bay")
        env_features = _env_features(
            bay_loads=bay_loads,
            assigned_block_count=block_step,
            total_block_count=len(blocks),
        )
        trace_rows.append(
            _trace_row(
                step=step,
                block_step=block_step,
                phase="SELECT_BAY",
                selected_block_set_id=selected_block.block_set_id,
                selected_bay=selected_bay,
                selected_action_index=selected_bay_index,
                candidates=bay_candidates,
                remaining_block_count=len(remaining_blocks),
                assigned_block_count=block_step,
                env_features=env_features,
                bay_loads=bay_loads,
            )
        )
        action_table_rows.append(
            {
                "step": step,
                "block_step": block_step,
                "phase": "SELECT_BAY",
                "selected_action_index": selected_bay_index,
                "selected_block_set_id": selected_block.block_set_id,
                "selected_bay": selected_bay,
                "selected_block_features": _block_feature_dict(selected_block, assigned=False),
                "env_features": env_features,
                "bay_loads": _snapshot_bay_loads(bay_loads),
                "candidates": [
                    _bay_candidate(
                        bay_id=bay_id,
                        loads=bay_loads[bay_id],
                        candidate_allowed=bay_id in selected_block.allowed_bay_ids,
                    )
                    for bay_id in normalized_bay_ids
                    if bay_id in selected_block.allowed_bay_ids
                ],
            }
        )
        step += 1

        _add_block_load(bay_loads[selected_bay], selected_bay, selected_block)
        del remaining_blocks[selected_block.block_set_id]

    return {
        "trace_rows": trace_rows,
        "action_table_rows": action_table_rows,
        "manifest": _manifest(
            algorithm=normalized_algorithm,
            bay_ids=normalized_bay_ids,
            plan=plan,
            trace_rows=trace_rows,
            teacher_order=[block.block_set_id for block in ordered_blocks],
        ),
        "plan": plan,
    }


def write_phase1_mdp_trace_package(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Write Phase 1 MDP trace, action table, manifest, and plan files."""

    print("[phase1-mdp-trace]")
    print(f"- algorithm: {algorithm}")
    print(f"- bay_ids: {','.join(str(bay_id) for bay_id in bay_ids)}")
    print(f"- output_dir: {output_dir}")

    package = build_phase1_mdp_trace_package(
        jobs=jobs,
        bay_ids=bay_ids,
        algorithm=algorithm,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trace_path = output_path / "phase1_mdp_trace.csv"
    action_table_path = output_path / "phase1_action_table.jsonl"
    manifest_path = output_path / "phase1_mdp_manifest.json"
    plan_paths = write_phase1_bay_plan(package["plan"], output_path)

    _write_csv(trace_path, PHASE1_TRACE_FIELDS, package["trace_rows"])
    with action_table_path.open("w", encoding="utf-8") as file:
        for row in package["action_table_rows"]:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(package["manifest"], file, ensure_ascii=False, indent=2, sort_keys=True)

    print(
        "[CHECK][phase1_mdp.write_phase1_mdp_trace_package] "
        f"blocks={package['manifest']['summary']['block_count']} "
        f"trace_rows={len(package['trace_rows'])} output={output_path}"
    )
    return {
        "trace_csv": str(trace_path),
        "action_table_jsonl": str(action_table_path),
        "manifest_json": str(manifest_path),
        "plan_json": plan_paths["json"],
        "assignments_csv": plan_paths["assignments_csv"],
        "bay_loads_csv": plan_paths["bay_loads_csv"],
    }


def _assignment_by_block(plan: Mapping, blocks: Sequence[Phase1Block]) -> Dict[str, str]:
    """Validate and return `block_set_id -> assigned_bay` teacher labels."""

    assignment_rows = plan.get("assignments")
    if not isinstance(assignment_rows, list) or not assignment_rows:
        print(
            "[ERROR][phase1_mdp._assignment_by_block] "
            f"cause=no_assignments rows_type={type(assignment_rows).__name__}"
        )
        raise RuntimeError("Phase 1 MDP trace requires non-empty plan assignments")
    assignment_by_block: Dict[str, str] = {}
    for index, row in enumerate(assignment_rows):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][phase1_mdp._assignment_by_block] "
                f"cause=invalid_assignment_row index={index} row_type={type(row).__name__}"
            )
            raise RuntimeError(f"invalid Phase 1 assignment row: {index}")
        block_set_id = str(row.get("block_set_id") or "")
        assigned_bay = str(row.get("assigned_bay") or "")
        if not block_set_id or not assigned_bay:
            print(
                "[ERROR][phase1_mdp._assignment_by_block] "
                f"cause=missing_assignment_key index={index} row={row}"
            )
            raise RuntimeError(f"missing Phase 1 assignment key at index={index}")
        assignment_by_block[block_set_id] = assigned_bay

    expected = {block.block_set_id for block in blocks}
    actual = set(assignment_by_block)
    if expected != actual:
        print(
            "[ERROR][phase1_mdp._assignment_by_block] "
            f"cause=assignment_block_mismatch missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
        raise RuntimeError("Phase 1 assignment block mismatch")
    return assignment_by_block


def _teacher_block_order(
    blocks: Sequence[Phase1Block],
    plan: Mapping,
    algorithm: str,
) -> List[Phase1Block]:
    """Return deterministic teacher order for the two-stage MDP trace."""

    block_by_id = {block.block_set_id: block for block in blocks}
    if algorithm == PRIORITY_GREEDY_PHASE1_HEURISTIC:
        ordered_ids = [str(row["block_set_id"]) for row in plan.get("assignments", [])]
        return [block_by_id[block_id] for block_id in ordered_ids]
    if algorithm in {
        MULTI_OBJECTIVE_PHASE1_HEURISTIC,
        PRIORITY_SWEEP_PHASE1_HEURISTIC,
        LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
    }:
        return sorted(
            blocks,
            key=lambda block: (
                -block.long_cut_over_1000,
                -block.steel_quantity_sum,
                -block.cut_length_sum,
                -block.bevel_quantity_sum,
                block.block_set_id,
            ),
        )
    return sorted(blocks, key=lambda block: (-block.steel_quantity_sum, -block.wo_count, block.block_set_id))


def _empty_bay_loads(bay_ids: Tuple[str, ...]) -> Dict[str, Dict]:
    """Create the Bay-load shape used by Phase 1 balancing."""

    return {
        bay_id: {
            "steel_quantity_sum": 0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "long_cut_bay24_count": 0,
            "wo_count": 0,
            "block_count": 0,
        }
        for bay_id in bay_ids
    }


def _env_features(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    assigned_block_count: int,
    total_block_count: int,
) -> Dict[str, float]:
    """Return global state features before the current action is applied."""

    if total_block_count <= 0:
        print("[ERROR][phase1_mdp._env_features] cause=no_blocks")
        raise RuntimeError("Phase 1 MDP requires at least one block")
    steel_values = [float(row["steel_quantity_sum"]) for row in bay_loads.values()]
    cut_values = [float(row["cut_length_sum"]) for row in bay_loads.values()]
    bevel_values = [float(row["bevel_quantity_sum"]) for row in bay_loads.values()]
    remaining_block_count = total_block_count - assigned_block_count
    return {
        "progress_ratio": assigned_block_count / total_block_count,
        "remaining_block_ratio": remaining_block_count / total_block_count,
        "steel_quantity_gap": _gap(steel_values),
        "cut_length_gap": _gap(cut_values),
        "bevel_quantity_gap": _gap(bevel_values),
        "long_cut_bay24_count": float(
            sum(int(row["long_cut_bay24_count"]) for row in bay_loads.values())
        ),
        "assigned_block_count": float(assigned_block_count),
    }


def _trace_row(
    step: int,
    block_step: int,
    phase: str,
    selected_block_set_id: str,
    selected_bay: str,
    selected_action_index: int,
    candidates: Sequence[str],
    remaining_block_count: int,
    assigned_block_count: int,
    env_features: Mapping[str, float],
    bay_loads: Mapping[str, Mapping[str, int | float]],
) -> Dict:
    """Build one CSV row for a Phase 1 decision."""

    return {
        "step": step,
        "block_step": block_step,
        "phase": phase,
        "selected_block_set_id": selected_block_set_id,
        "selected_bay": selected_bay,
        "selected_action_index": selected_action_index,
        "candidate_count": len(candidates),
        "remaining_block_count": remaining_block_count,
        "assigned_block_count": assigned_block_count,
        "env_progress_ratio": env_features["progress_ratio"],
        "env_remaining_block_ratio": env_features["remaining_block_ratio"],
        "env_steel_quantity_gap": env_features["steel_quantity_gap"],
        "env_cut_length_gap": env_features["cut_length_gap"],
        "env_bevel_quantity_gap": env_features["bevel_quantity_gap"],
        "env_long_cut_bay24_count": env_features["long_cut_bay24_count"],
        "bay_loads_json": json.dumps(_snapshot_bay_loads(bay_loads), ensure_ascii=False, sort_keys=True),
        "candidate_ids": "|".join(candidates),
    }


def _block_candidate(block: Phase1Block, assigned: bool) -> Dict:
    """Return one block candidate row for JSONL action table."""

    return {
        "action_id": f"select_block:{block.block_set_id}",
        "block_set_id": block.block_set_id,
        "project_no": block.project_no,
        "block_no": block.block_no,
        "job_ids": list(block.job_ids),
        "allowed_bay_ids": list(block.allowed_bay_ids),
        "features": _block_feature_vector(block, assigned=assigned),
        "feature_dict": _block_feature_dict(block, assigned=assigned),
    }


def _bay_candidate(
    bay_id: str,
    loads: Mapping[str, int | float],
    candidate_allowed: bool,
) -> Dict:
    """Return one Bay candidate row for JSONL action table."""

    feature_dict = _bay_feature_dict(bay_id, loads, candidate_allowed)
    return {
        "action_id": f"select_bay:{bay_id}",
        "bay_id": bay_id,
        "features": [feature_dict[name] for name in PHASE1_BAY_FEATURE_NAMES],
        "feature_dict": feature_dict,
    }


def _block_feature_vector(block: Phase1Block, assigned: bool) -> List[float]:
    """Return model-ready block features without teacher-label leakage."""

    feature_dict = _block_feature_dict(block, assigned)
    return [feature_dict[name] for name in PHASE1_BLOCK_FEATURE_NAMES]


def _block_feature_dict(block: Phase1Block, assigned: bool) -> Dict[str, float]:
    """Return named block features used by the Phase 1 pointer policy."""

    return {
        "steel_quantity_sum": float(block.steel_quantity_sum),
        "cut_length_sum": float(block.cut_length_sum),
        "bevel_quantity_sum": float(block.bevel_quantity_sum),
        "wo_count": float(block.wo_count),
        "long_cut_over_1000": float(block.long_cut_over_1000),
        "allowed_bay_count": float(len(block.allowed_bay_ids)),
        "remaining_flag": 0.0 if assigned else 1.0,
        "assigned_flag": 1.0 if assigned else 0.0,
    }


def _bay_feature_dict(
    bay_id: str,
    loads: Mapping[str, int | float],
    candidate_allowed: bool,
) -> Dict[str, float]:
    """Return named Bay-load features used by the Phase 1 pointer policy."""

    return {
        "bay_id_numeric": float(bay_id),
        "steel_quantity_sum": float(loads["steel_quantity_sum"]),
        "cut_length_sum": float(loads["cut_length_sum"]),
        "bevel_quantity_sum": float(loads["bevel_quantity_sum"]),
        "wo_count": float(loads["wo_count"]),
        "block_count": float(loads["block_count"]),
        "long_cut_bay24_count": float(loads["long_cut_bay24_count"]),
        "candidate_allowed_flag": 1.0 if candidate_allowed else 0.0,
    }


def _snapshot_bay_loads(bay_loads: Mapping[str, Mapping[str, int | float]]) -> Dict[str, Dict[str, float]]:
    """Return a plain dict copy for JSON/CSV output."""

    return {
        bay_id: {
            "steel_quantity_sum": float(loads["steel_quantity_sum"]),
            "cut_length_sum": float(loads["cut_length_sum"]),
            "bevel_quantity_sum": float(loads["bevel_quantity_sum"]),
            "long_cut_bay24_count": float(loads["long_cut_bay24_count"]),
            "wo_count": float(loads["wo_count"]),
            "block_count": float(loads["block_count"]),
        }
        for bay_id, loads in bay_loads.items()
    }


def _candidate_index(candidates: Sequence[str], selected: str, field_name: str) -> int:
    """Find selected candidate index and fail loudly if masking is inconsistent."""

    try:
        return list(candidates).index(selected)
    except ValueError as exc:
        print(
            "[ERROR][phase1_mdp._candidate_index] "
            f"cause=selected_not_in_candidates field={field_name} selected={selected} "
            f"candidates={list(candidates)}"
        )
        raise RuntimeError(f"selected_not_in_candidates: {field_name}={selected}") from exc


def _manifest(
    algorithm: str,
    bay_ids: Tuple[str, ...],
    plan: Mapping,
    trace_rows: Sequence[Mapping],
    teacher_order: Sequence[str],
) -> Dict:
    """Build metadata that explains how the learning package was made."""

    return {
        "phase": "phase1_block_to_bay_mdp",
        "label_source": "deterministic_phase1_heuristic",
        "algorithm": algorithm,
        "bay_ids": list(bay_ids),
        "decision_phases": PHASE1_DECISION_PHASES,
        "block_feature_names": PHASE1_BLOCK_FEATURE_NAMES,
        "bay_feature_names": PHASE1_BAY_FEATURE_NAMES,
        "env_feature_names": PHASE1_ENV_FEATURE_NAMES,
        "teacher_order": list(teacher_order),
        "teacher_order_note": (
            "Multi-objective final assignments are converted to a deterministic "
            "SELECT_BLOCK order; SELECT_BAY labels come from the validated Phase 1 plan."
        ),
        "summary": {
            **dict(plan["summary"]),
            "trace_row_count": len(trace_rows),
            "decision_count_per_block": 2,
        },
        "mdp_definition": {
            "state": "remaining block candidates, current Bay loads, and progress features",
            "action": "SELECT_BLOCK then SELECT_BAY",
            "transition": "after SELECT_BAY, add the selected block load to the selected Bay",
            "reward_or_label": "self-label target from Phase 1 heuristic; RL reward can be added later",
        },
    }


def _gap(values: Sequence[float]) -> float:
    """Return max-min gap for one workload metric."""

    if not values:
        print("[ERROR][phase1_mdp._gap] cause=no_values")
        raise RuntimeError("cannot compute gap for empty values")
    return round(max(values) - min(values), 6)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping]) -> None:
    """Write CSV with deterministic field order."""

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
