"""학습용 trajectory package 생성 helper.

이 모듈은 실적 Excel에서 만든 DES scenario를 직접 학습 정답으로 보지 않는다.
현재 1차 목적은 실제 W/O 입력으로 환경과 계층형 action mask가 끝까지 동작하는지
확인하는 smoke dataset을 만들고, heuristic pseudo-label을 명확히 남기는 것이다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Sequence

from Environment.gym_wrapper import run_hierarchical_trace_export


ALLOWED_DATA_ROLES = {
    "actual_source_smoke",
    "clean_slice_smoke",
    "heuristic_baseline",
}


def _validate_non_empty_sequence(name: str, values: Sequence[str]) -> None:
    """빈 입력 list를 조용히 기본값으로 대체하지 않기 위한 검증."""

    if not values:
        print(f"[ERROR][learning_data_builder._validate_non_empty_sequence] cause=empty_{name}")
        raise ValueError(f"{name} must not be empty")
    empty_items = [index for index, value in enumerate(values) if str(value).strip() == ""]
    if empty_items:
        print(
            "[ERROR][learning_data_builder._validate_non_empty_sequence] "
            f"cause=blank_item name={name} indices={empty_items}"
        )
        raise ValueError(f"{name} contains blank item")


def _validate_data_role(data_role: str) -> None:
    """학습데이터 역할이 명시된 허용값인지 확인한다."""

    if data_role not in ALLOWED_DATA_ROLES:
        print(
            "[ERROR][learning_data_builder._validate_data_role] "
            f"cause=unknown_data_role data_role={data_role} "
            f"allowed={sorted(ALLOWED_DATA_ROLES)}"
        )
        raise ValueError(f"unknown learning data role: {data_role}")


def _validate_trace_summary(summary: Dict, *, config_path: str, heuristic: str) -> None:
    """trace exporter 결과가 manifest에 필요한 필드를 갖는지 확인한다."""

    required_keys = {
        "scheduled_jobs",
        "unscheduled_jobs",
        "hierarchical_step_count",
        "flat_decision_count",
        "phase_counts",
        "trace_csv",
        "action_table_jsonl",
        "summary_json",
    }
    missing = sorted(required_keys - set(summary))
    if missing:
        print(
            "[ERROR][learning_data_builder._validate_trace_summary] "
            f"cause=missing_summary_keys config={config_path} heuristic={heuristic} missing={missing}"
        )
        raise KeyError("hierarchical trace summary is missing required keys")
    if int(summary["scheduled_jobs"]) <= 0:
        print(
            "[ERROR][learning_data_builder._validate_trace_summary] "
            f"cause=no_scheduled_jobs config={config_path} heuristic={heuristic}"
        )
        raise RuntimeError("learning data trace has no scheduled jobs")


def _dataset_output_dir(output_dir: Path, config_path: str, heuristic: str) -> Path:
    """config/heuristic별 trace 산출물 디렉터리를 만든다."""

    config_stem = Path(config_path).stem
    heuristic_key = str(heuristic).strip()
    if not config_stem or not heuristic_key:
        print(
            "[ERROR][learning_data_builder._dataset_output_dir] "
            f"cause=blank_config_or_heuristic config={config_path} heuristic={heuristic}"
        )
        raise ValueError("config stem and heuristic must not be blank")
    return output_dir / config_stem / heuristic_key


def build_learning_data_package(
    *,
    config_paths: Sequence[str],
    heuristics: Sequence[str],
    output_dir: str,
    max_actions: int,
    data_role: str,
    algorithm_plan: Sequence[str],
    env_builder: Callable[[str], object],
    trace_exporter: Callable[..., Dict] = run_hierarchical_trace_export,
) -> Dict:
    """여러 config/heuristic의 계층형 trace를 학습용 package로 묶는다.

    `label_source`는 항상 `heuristic_pseudo_label`로 기록한다. 실적 데이터에서
    읽은 W/O를 사용하더라도 현장 실적이 최적 정답이라는 뜻은 아니다.
    """

    _validate_non_empty_sequence("config_paths", config_paths)
    _validate_non_empty_sequence("heuristics", heuristics)
    _validate_non_empty_sequence("algorithm_plan", algorithm_plan)
    _validate_data_role(data_role)
    if int(max_actions) <= 0:
        print(
            "[ERROR][learning_data_builder.build_learning_data_package] "
            f"cause=invalid_max_actions max_actions={max_actions}"
        )
        raise ValueError("max_actions must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    datasets = []

    for config_path in config_paths:
        if not Path(config_path).exists():
            print(
                "[ERROR][learning_data_builder.build_learning_data_package] "
                f"cause=config_path_not_found config={config_path}"
            )
            raise FileNotFoundError(f"config path not found: {config_path}")
        for heuristic in heuristics:
            dataset_dir = _dataset_output_dir(output_path, config_path, heuristic)
            env = env_builder(config_path)
            summary = trace_exporter(
                env=env,
                heuristic_name=heuristic,
                max_actions=int(max_actions),
                output_dir=str(dataset_dir),
            )
            _validate_trace_summary(summary, config_path=config_path, heuristic=heuristic)
            datasets.append(
                {
                    "config_path": str(config_path),
                    "heuristic": str(heuristic),
                    "data_role": data_role,
                    "label_source": "heuristic_pseudo_label",
                    "scheduled_jobs": int(summary["scheduled_jobs"]),
                    "unscheduled_jobs": list(summary["unscheduled_jobs"]),
                    "hierarchical_step_count": int(summary["hierarchical_step_count"]),
                    "flat_decision_count": int(summary["flat_decision_count"]),
                    "phase_counts": dict(summary["phase_counts"]),
                    "trace_csv": str(summary["trace_csv"]),
                    "action_table_jsonl": str(summary["action_table_jsonl"]),
                    "summary_json": str(summary["summary_json"]),
                }
            )

    manifest = {
        "data_role": data_role,
        "label_source": "heuristic_pseudo_label",
        "label_warning": (
            "실적 데이터 기반 W/O를 사용하더라도 label은 현장 실적 정답이 아니라 "
            "휴리스틱 pseudo-label이다."
        ),
        "max_actions": int(max_actions),
        "algorithm_plan": [str(item) for item in algorithm_plan],
        "dataset_count": len(datasets),
        "datasets": datasets,
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result = dict(manifest)
    result["manifest_json"] = str(manifest_path)
    print(
        "[CHECK][learning_data_builder.build_learning_data_package] "
        f"data_role={data_role} dataset_count={len(datasets)} output={manifest_path}"
    )
    return result
