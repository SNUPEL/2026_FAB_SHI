"""휴리스틱 baseline 비교와 회사 송부용 정적 HTML 패키지를 만든다.

이 모듈은 강화학습 전 단계의 검증 산출물을 하나로 묶는다.

입력 예시:
- config 목록: `config_clean_50.yaml`, `config_clean_100.yaml`, `config_clean_200.yaml`
- heuristic 목록: `batch_fill_spt`, `balanced_batch`, `balanced_batch_count`, `spt`, `load_balance`, `priority`

출력 예시:
- `output/share/html_viewer_package/index.html`
- `output/share/html_viewer_package/baseline_summary.csv`
- `output/share/html_viewer_package/baseline_summary.json`
- `output/share/html_viewer_package/clean_100/generated/batch_fill_spt/playback.html`
- `output/share/html_viewer_package/clean_100/actual/actual_replay.html`
- `output/share/html_viewer_package.zip`

중요한 검증 기준:
- generated schedule은 모든 W/O를 스케줄해야 한다.
- generated schedule의 hard violation은 0이어야 한다.
- actual replay의 identity error는 0이어야 한다.
- 실패하면 조용히 넘어가지 않고 원인과 입력 config/heuristic을 출력한 뒤 예외를 발생시킨다.
"""

from __future__ import annotations

import copy
import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from Agent.heuristics import select_action_by_rule
from Environment.environment import CuttingShopEnvironment
from Utils.config import load_config
from Utils.data.io import load_scenario_for_config
from Utils.reporting.playback_builder import write_actual_replay_artifacts, write_playback_artifacts


DEFAULT_CLEAN_CONFIGS = (
    "config_clean_50.yaml",
    "config_clean_100.yaml",
    "config_clean_200.yaml",
)

DEFAULT_HEURISTICS = (
    "batch_fill_spt",
    "balanced_batch",
    "balanced_batch_count",
    "spt",
    "load_balance",
    "priority",
)


def parse_csv_argument(value: str | None, default: Sequence[str]) -> List[str]:
    """CLI의 comma-separated 문자열을 공백 제거된 list로 바꾼다."""

    if value is None or str(value).strip() == "":
        return list(default)
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        print(
            "[ERROR][share_report_builder.parse_csv_argument] "
            f"cause=empty_csv_argument input={value!r}"
        )
        raise ValueError("comma-separated argument produced no items")
    return items


def build_html_package(
    config_paths: Sequence[str] | None = None,
    heuristics: Sequence[str] | None = None,
    output_dir: str = "output/share/html_viewer_package",
    zip_path: str = "output/share/html_viewer_package.zip",
) -> Dict[str, Any]:
    """clean 데이터별 actual replay와 heuristic playback을 만들고 index.html로 묶는다."""

    resolved_configs = list(config_paths or DEFAULT_CLEAN_CONFIGS)
    resolved_heuristics = list(heuristics or DEFAULT_HEURISTICS)
    package_dir = Path(output_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[CHECK][share_report_builder.build_html_package] "
        f"configs={len(resolved_configs)} heuristics={len(resolved_heuristics)} "
        f"output_dir={package_dir} zip_path={zip_path}"
    )

    baseline_rows: List[Dict[str, Any]] = []
    actual_rows: List[Dict[str, Any]] = []
    manifest_paths: List[Path] = []
    dataset_summaries: List[Dict[str, Any]] = []

    for config_path in resolved_configs:
        config = load_config(config_path)
        dataset_label = _dataset_label(config_path)
        scenario = load_scenario_for_config(config)
        job_count = len(scenario.get("jobs", []))
        if job_count <= 0:
            print(
                "[ERROR][share_report_builder.build_html_package] "
                f"cause=empty_scenario config={config_path}"
            )
            raise RuntimeError(f"scenario has no jobs: {config_path}")

        print(
            "[CHECK][share_report_builder.build_html_package] "
            f"dataset={dataset_label} config={config_path} jobs={job_count}"
        )

        actual_output_dir = package_dir / dataset_label / "actual"
        actual_row, actual_files = _run_actual_replay(
            config=config,
            scenario=scenario,
            config_path=config_path,
            dataset_label=dataset_label,
            job_count=job_count,
            output_dir=actual_output_dir,
            package_dir=package_dir,
        )
        actual_rows.append(actual_row)
        manifest_paths.extend(actual_files)

        for heuristic in resolved_heuristics:
            generated_output_dir = package_dir / dataset_label / "generated" / _safe_path_name(heuristic)
            baseline_row, generated_files = _run_generated_playback(
                config=config,
                scenario=scenario,
                config_path=config_path,
                dataset_label=dataset_label,
                heuristic=heuristic,
                job_count=job_count,
                output_dir=generated_output_dir,
                package_dir=package_dir,
            )
            baseline_rows.append(baseline_row)
            manifest_paths.extend(generated_files)

        dataset_summaries.append(
            {
                "dataset": dataset_label,
                "config_path": config_path,
                "source_data_path": config.get("paths", {}).get("source_data_path", ""),
                "job_count": job_count,
                "actual_replay_html": actual_row["actual_replay_html"],
                "actual_identity_error_count": actual_row["identity_error_count"],
            }
        )

    summary = {
        "generated_at": "2026-05-27",
        "package_note": "Static HTML package. Open index.html directly in a browser.",
        "process_time_basis": "Generated simulation uses TACT_TIME minutes. Actual replay uses RT_CUT_ST_DTM/RT_CUT_ED_DTM only for identity/time-axis replay.",
        "batch_constraint": "Each generated machine batch contains 1-3 W/O and 길이 sum <= 55,000. Batch W/O enter and finish together.",
        "datasets": dataset_summaries,
        "heuristics": resolved_heuristics,
        "baseline_rows": baseline_rows,
        "actual_rows": actual_rows,
    }

    baseline_csv = package_dir / "baseline_summary.csv"
    actual_csv = package_dir / "actual_replay_summary.csv"
    summary_json = package_dir / "baseline_summary.json"
    index_html = package_dir / "index.html"

    _write_csv(baseline_rows, baseline_csv)
    _write_csv(actual_rows, actual_csv)
    _write_json(summary, summary_json)
    index_html.write_text(_build_index_html(summary), encoding="utf-8")
    manifest_paths.extend([baseline_csv, actual_csv, summary_json, index_html])

    zip_output = Path(zip_path)
    _write_zip(zip_output, package_dir, manifest_paths)

    result = {
        "package_dir": str(package_dir),
        "index_html": str(index_html),
        "zip_path": str(zip_output),
        "baseline_rows": len(baseline_rows),
        "actual_rows": len(actual_rows),
        "datasets": len(dataset_summaries),
        "heuristics": len(resolved_heuristics),
    }
    print(
        "[VALIDATION][share_report_builder.build_html_package] "
        f"passed=true baseline_rows={result['baseline_rows']} "
        f"actual_rows={result['actual_rows']} index={index_html} zip={zip_output}"
    )
    return result


def _run_generated_playback(
    config: Dict[str, Any],
    scenario: Dict[str, Any],
    config_path: str,
    dataset_label: str,
    heuristic: str,
    job_count: int,
    output_dir: Path,
    package_dir: Path,
) -> tuple[Dict[str, Any], List[Path]]:
    """하나의 config/heuristic 조합을 실행하고 baseline row를 만든다."""

    env = CuttingShopEnvironment(config=copy.deepcopy(config), scenario=scenario)
    run_result = _run_heuristic_with_trace(env, heuristic)
    artifacts = write_playback_artifacts(
        env.simulation,
        env.config,
        run_result["trace_records"],
        str(output_dir),
    )
    metrics = _read_json(output_dir / "metrics.json")
    validation = _read_json(output_dir / "validation_result.json")

    row = {
        "dataset": dataset_label,
        "config_path": config_path,
        "source_data_path": config.get("paths", {}).get("source_data_path", ""),
        "heuristic": heuristic,
        "action_mode": config.get("action_space", {}).get("mode", ""),
        "job_count": job_count,
        "scheduled_jobs": int(metrics["scheduled_jobs"]),
        "unscheduled_count": len(env.simulation.state.unscheduled_jobs),
        "makespan_minutes": round(float(metrics["makespan_minutes"]), 6),
        "machine_load_imbalance": round(float(metrics["machine_load_imbalance"]), 6),
        "bay_load_imbalance": round(float(metrics["bay_process_load_imbalance"]), 6),
        "machine_job_count_imbalance": round(float(metrics["machine_job_count_imbalance"]), 6),
        "bay_job_count_imbalance": round(float(metrics["bay_job_count_imbalance"]), 6),
        "hard_violation_count": int(validation["hard_violation_count"]),
        "event_identity_error_count": int(metrics["event_log"]["event_identity_error_count"]),
        "event_constraint_violation_count": int(metrics["event_log"]["event_constraint_violation_count"]),
        "playback_html": _relative_path(package_dir, output_dir / "playback.html"),
        "action_trace_csv": _relative_path(package_dir, output_dir / "action_trace.csv"),
        "metrics_json": _relative_path(package_dir, output_dir / "metrics.json"),
        "validation_json": _relative_path(package_dir, output_dir / "validation_result.json"),
        "event_log_json": _relative_path(package_dir, output_dir / "event_log.json"),
    }
    _validate_generated_row(row, artifacts)
    print(
        "[VALIDATION][share_report_builder._run_generated_playback] "
        f"dataset={dataset_label} heuristic={heuristic} passed=true "
        f"scheduled_jobs={row['scheduled_jobs']} makespan={row['makespan_minutes']:.2f} "
        f"hard_violation_count={row['hard_violation_count']}"
    )
    return row, _collect_existing_files(output_dir)


def _run_actual_replay(
    config: Dict[str, Any],
    scenario: Dict[str, Any],
    config_path: str,
    dataset_label: str,
    job_count: int,
    output_dir: Path,
    package_dir: Path,
) -> tuple[Dict[str, Any], List[Path]]:
    """하나의 config에 대해 실적 timestamp/장비/Bay identity replay를 실행한다."""

    artifacts = write_actual_replay_artifacts(
        scenario,
        copy.deepcopy(config),
        str(output_dir),
    )
    metrics = _read_json(output_dir / "actual_metrics.json")
    validation = _read_json(output_dir / "actual_validation_result.json")

    row = {
        "dataset": dataset_label,
        "config_path": config_path,
        "source_data_path": config.get("paths", {}).get("source_data_path", ""),
        "job_count": job_count,
        "scheduled_jobs": int(metrics["scheduled_jobs"]),
        "makespan_minutes": round(float(metrics["makespan_minutes"]), 6),
        "identity_error_count": int(validation["identity_error_count"]),
        "event_identity_error_count": int(metrics["event_log"]["event_identity_error_count"]),
        "event_constraint_violation_count": int(metrics["event_log"]["event_constraint_violation_count"]),
        "constraint_audit_violation_count": int(metrics.get("constraint_audit", {}).get("hard_violation_count", 0)),
        "actual_replay_html": _relative_path(package_dir, output_dir / "actual_replay.html"),
        "actual_metrics_json": _relative_path(package_dir, output_dir / "actual_metrics.json"),
        "actual_validation_json": _relative_path(package_dir, output_dir / "actual_validation_result.json"),
        "actual_event_log_json": _relative_path(package_dir, output_dir / "actual_event_log.json"),
    }
    _validate_actual_row(row, artifacts)
    print(
        "[VALIDATION][share_report_builder._run_actual_replay] "
        f"dataset={dataset_label} passed=true scheduled_jobs={row['scheduled_jobs']} "
        f"identity_error_count={row['identity_error_count']}"
    )
    return row, _collect_existing_files(output_dir)


def _run_heuristic_with_trace(env: CuttingShopEnvironment, heuristic_name: str) -> Dict[str, Any]:
    """기존 DES 후보 목록에서 휴리스틱 action을 선택하며 action trace를 남긴다."""

    observation, _ = env.reset()
    trace_records: List[Dict[str, Any]] = []
    step_index = 0
    terminated = False
    truncated = False
    horizon = float(env.config["simulation"]["horizon_minutes"])

    while not (terminated or truncated) and env.simulation._has_pending_work() and env.simulation.state.current_time <= horizon:  # noqa: SLF001
        candidates = env.get_action_candidates()
        selected = select_action_by_rule(candidates, env.simulation, heuristic_name) if candidates else None

        if selected is None:
            before_time = float(env.simulation.state.current_time)
            env.simulation._advance_to_decision_epoch()  # noqa: SLF001
            after_time = float(env.simulation.state.current_time)
            if after_time <= before_time + 1e-9:
                print(
                    "[ERROR][share_report_builder._run_heuristic_with_trace] "
                    f"cause=no_candidate_and_time_cannot_advance heuristic={heuristic_name} "
                    f"current_time={before_time:.6f} pending_work={env.simulation._has_pending_work()}"  # noqa: SLF001
                )
                raise RuntimeError(f"heuristic cannot progress: {heuristic_name}")
            continue

        trace_record = {
            "step": step_index,
            "decision_time_min": round(float(env.simulation.state.current_time), 6),
            "candidate_count": len(candidates),
            "remaining_jobs_before": len(env.simulation.state.unscheduled_jobs),
            "action_id": selected.action_id,
            "action_kind": selected.action_kind,
            "batch_id": selected.batch_id or "",
            "batch_job_ids": "|".join(selected.batch_job_ids),
            "batch_wo_count": selected.batch_wo_count,
            "batch_length_sum": round(float(selected.batch_length_sum), 6),
            "job_id": selected.job.job_id,
            "machine_id": selected.machine.machine_id,
            "machine_bay_id": selected.machine.bay_id or "",
            "estimated_minutes": round(float(selected.estimated_minutes), 6),
            "processing_minutes": round(float(selected.processing_minutes), 6),
            "finish_time_min": round(float(selected.finish_time), 6),
            "soft_penalty": round(float(selected.soft_penalty), 6),
            "soft_reasons": "|".join(selected.soft_reasons),
        }

        observation, reward, terminated, truncated, _ = env.simulation.step_candidate(
            selected,
            build_observation_result=False,
            advance_decision_epoch=True,
        )
        trace_record.update(
            {
                "reward": round(float(reward), 6),
                "current_time_after_min": round(float(observation["current_time"]), 6),
                "remaining_jobs_after": int(observation["remaining_job_count"]),
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        trace_records.append(trace_record)
        step_index += 1

    return {
        "observation": observation,
        "trace_records": trace_records,
        "terminated": terminated,
        "truncated": truncated,
    }


def _validate_generated_row(row: Dict[str, Any], artifacts: Dict[str, str]) -> None:
    """generated heuristic 결과가 강화학습 전 baseline으로 쓸 수 있는지 검사한다."""

    failed_reasons = []
    if row["scheduled_jobs"] != row["job_count"]:
        failed_reasons.append("scheduled_jobs_mismatch")
    if row["unscheduled_count"] != 0:
        failed_reasons.append("unscheduled_jobs_remain")
    if row["hard_violation_count"] != 0:
        failed_reasons.append("hard_violation")
    if row["event_identity_error_count"] != 0:
        failed_reasons.append("event_identity_error")
    if row["event_constraint_violation_count"] != 0:
        failed_reasons.append("event_constraint_violation")

    if failed_reasons:
        print(
            "[ERROR][share_report_builder._validate_generated_row] "
            f"cause={','.join(failed_reasons)} dataset={row['dataset']} "
            f"heuristic={row['heuristic']} scheduled={row['scheduled_jobs']} "
            f"job_count={row['job_count']} artifacts={artifacts}"
        )
        raise RuntimeError(f"generated heuristic validation failed: {row['dataset']}::{row['heuristic']}")


def _validate_actual_row(row: Dict[str, Any], artifacts: Dict[str, str]) -> None:
    """actual replay가 원본 실적 장비/Bay/시각을 그대로 재현했는지 검사한다."""

    failed_reasons = []
    if row["scheduled_jobs"] != row["job_count"]:
        failed_reasons.append("scheduled_jobs_mismatch")
    if row["identity_error_count"] != 0:
        failed_reasons.append("identity_error")
    if row["event_identity_error_count"] != 0:
        failed_reasons.append("event_identity_error")
    if row["event_constraint_violation_count"] != 0:
        failed_reasons.append("event_constraint_violation")

    if failed_reasons:
        print(
            "[ERROR][share_report_builder._validate_actual_row] "
            f"cause={','.join(failed_reasons)} dataset={row['dataset']} "
            f"scheduled={row['scheduled_jobs']} job_count={row['job_count']} "
            f"artifacts={artifacts}"
        )
        raise RuntimeError(f"actual replay validation failed: {row['dataset']}")


def _dataset_label(config_path: str) -> str:
    """config 파일명을 HTML 패키지 내부 dataset 이름으로 바꾼다."""

    stem = Path(config_path).stem
    if stem.startswith("config_"):
        stem = stem[len("config_") :]
    label = _safe_path_name(stem)
    if not label:
        print(
            "[ERROR][share_report_builder._dataset_label] "
            f"cause=empty_dataset_label config_path={config_path}"
        )
        raise ValueError(f"cannot derive dataset label from config path: {config_path}")
    return label


def _safe_path_name(value: str) -> str:
    """파일/디렉터리명에 안전한 ASCII 중심 이름으로 제한한다."""

    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return safe.strip("_")


def _load_imbalance(loads: Dict[str, Any]) -> float:
    """부하평준화 비교용 단순 지표: max(load) - min(load)."""

    if not loads:
        print("[ERROR][share_report_builder._load_imbalance] cause=empty_loads")
        raise ValueError("cannot compute imbalance from empty loads")
    values = [float(value) for value in loads.values()]
    return round(max(values) - min(values), 6)


def _read_json(path: Path) -> Any:
    """필수 JSON 파일을 읽는다. 없거나 깨져 있으면 즉시 실패한다."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR][share_report_builder._read_json] cause={exc} path={path}")
        raise RuntimeError(f"failed to read required json: {path}") from exc


def _write_json(data: Any, path: Path) -> None:
    """dict/list 데이터를 UTF-8 JSON으로 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    """dict row 목록을 Excel에서도 바로 열 수 있게 UTF-8 BOM CSV로 저장한다."""

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[ERROR][share_report_builder._write_csv] cause=empty_rows path={path}")
        raise ValueError(f"refuse to write empty csv: {path}")

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_existing_files(root: Path) -> List[Path]:
    """zip manifest에 넣을 현재 run 산출물만 수집한다."""

    if not root.exists():
        print(f"[ERROR][share_report_builder._collect_existing_files] cause=missing_root root={root}")
        raise RuntimeError(f"missing output root: {root}")
    return [path for path in root.rglob("*") if path.is_file()]


def _relative_path(base: Path, path: Path) -> str:
    """index.html에서 클릭 가능한 package-relative path를 만든다."""

    try:
        return path.relative_to(base).as_posix()
    except ValueError as exc:
        print(
            "[ERROR][share_report_builder._relative_path] "
            f"cause=path_not_under_base base={base} path={path}"
        )
        raise RuntimeError(f"path is outside package dir: {path}") from exc


def _write_zip(zip_path: Path, package_dir: Path, paths: Sequence[Path]) -> None:
    """생성한 정적 파일만 zip으로 묶는다."""

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    unique_paths = []
    seen = set()
    for path in paths:
        if not path.exists():
            print(f"[ERROR][share_report_builder._write_zip] cause=missing_manifest_file path={path}")
            raise RuntimeError(f"zip manifest file does not exist: {path}")
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in unique_paths:
            archive.write(path, _relative_path(package_dir, path))


def _build_index_html(summary: Dict[str, Any]) -> str:
    """회사 송부용 정적 index.html 본문을 만든다."""

    payload = json.dumps(summary, ensure_ascii=False)
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PMSP 절단 DES 휴리스틱 baseline</title>
  <style>
    :root {
      --ink: #17201b;
      --muted: #66736b;
      --paper: #f4efe3;
      --panel: #fffaf0;
      --line: #d8cdb6;
      --accent: #b85c38;
      --good: #1f7a4d;
      --warn: #b45309;
      --bad: #b42318;
      --machine: #243b53;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Aptos", "Segoe UI", "Malgun Gothic", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 10%, rgba(184, 92, 56, 0.16), transparent 28rem),
        linear-gradient(135deg, #f4efe3 0%, #e9dfca 100%);
    }
    header {
      padding: 28px 34px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 250, 240, 0.82);
      backdrop-filter: blur(8px);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(26px, 3vw, 42px);
      letter-spacing: -0.04em;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }
    main { padding: 24px 34px 44px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 14px 36px rgba(59, 45, 27, 0.08);
    }
    .card small {
      display: block;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .metric {
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
      margin: 18px 0;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 700;
    }
    select, input {
      min-width: 180px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fffdf7;
      color: var(--ink);
      font-size: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 16px;
      background: #fffdf7;
      border: 1px solid var(--line);
    }
    th, td {
      padding: 11px 12px;
      border-bottom: 1px solid #eadfca;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }
    th {
      background: #efe3ca;
      color: #3b3020;
      position: sticky;
      top: 111px;
      z-index: 4;
    }
    tr:hover td { background: #fff7e5; }
    a {
      color: var(--machine);
      font-weight: 800;
      text-decoration: none;
      border-bottom: 1px solid rgba(36, 59, 83, 0.35);
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      background: #ede1c8;
      color: #3b3020;
    }
    .ok { color: var(--good); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .section-title {
      margin: 28px 0 12px;
      font-size: 20px;
      letter-spacing: -0.02em;
    }
    .note {
      color: var(--muted);
      line-height: 1.6;
      margin: 0 0 14px;
    }
    .checklist-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
      gap: 16px;
      align-items: start;
    }
    .checklist-items {
      display: grid;
      gap: 9px;
    }
    .check-item {
      display: flex;
      gap: 9px;
      align-items: flex-start;
      color: var(--ink);
      font-size: 14px;
      font-weight: 650;
      line-height: 1.55;
    }
    .check-item input {
      min-width: auto;
      width: 16px;
      height: 16px;
      margin: 3px 0 0;
      accent-color: var(--accent);
    }
    .link-list {
      display: grid;
      gap: 8px;
    }
    .link-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border: 1px solid #eadfca;
      border-radius: 12px;
      background: #fffdf7;
    }
    .link-row strong {
      display: block;
      margin-bottom: 4px;
    }
    .status-line {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    @media (max-width: 920px) {
      header, main { padding-left: 18px; padding-right: 18px; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .checklist-grid { grid-template-columns: 1fr; }
      table { display: block; overflow-x: auto; }
      th { position: static; }
    }
    @media (max-width: 560px) {
      .grid { grid-template-columns: 1fr; }
      select, input { min-width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>절단 공장 DES baseline viewer</h1>
    <p class="subtitle">clean 50/100/200 데이터 기준 actual replay와 generated heuristic 결과를 같은 기준으로 비교합니다. 처리시간은 택트타임 분 단위, generated batch 제약은 W/O 1~3개 및 길이합 55,000 이하입니다.</p>
  </header>
  <main>
    <section class="grid" id="kpis"></section>
    <section class="card">
      <h2 class="section-title">검증 기준</h2>
      <p class="note" id="basis"></p>
      <p class="note">Actual replay는 원본 실적 장비명, 절단 베이, 실적 착수시간, 실적 종료시간을 그대로 재현하는 identity 검증입니다. Generated schedule은 휴리스틱이 만든 계획이며 hard violation 0이어야 합니다.</p>
    </section>
    <section class="card">
      <h2 class="section-title">clean 100 generated/actual 수동 검수 checklist</h2>
      <p class="note">강화학습 전 clean_100 기준으로 actual replay와 generated playback을 직접 열어 확인합니다. 이 패키지에 clean_100이 없으면 다른 데이터로 대체하지 않고 누락 상태를 표시합니다.</p>
      <div class="checklist-grid">
        <div class="checklist-items">
          <label class="check-item"><input type="checkbox">Actual replay에서 scheduled_jobs=100, identity error=0, event identity error=0인지 확인</label>
          <label class="check-item"><input type="checkbox">Actual replay의 event constraint violation과 constraint audit count가 0인지 확인하고, 0이 아니면 데이터 품질 후보로 분리</label>
          <label class="check-item"><input type="checkbox">Generated playback별 scheduled_jobs=100, unscheduled=0, hard violation=0 확인</label>
          <label class="check-item"><input type="checkbox">Generated event identity error=0, event constraint violation=0 확인</label>
          <label class="check-item"><input type="checkbox">Playback에서 batch W/O 수 1~3개와 길이합 55,000 이하, machine/Bay 일관성, timeline 진행을 샘플 W/O 검색으로 확인</label>
        </div>
        <div>
          <p class="note">clean_100 바로가기와 핵심 count</p>
          <div class="link-list" id="clean100ChecklistLinks"></div>
        </div>
      </div>
    </section>
    <section>
      <h2 class="section-title">휴리스틱 baseline 비교</h2>
      <div class="controls">
        <label>데이터
          <select id="datasetFilter"></select>
        </label>
        <label>휴리스틱
          <select id="heuristicFilter"></select>
        </label>
        <label>검색
          <input id="searchInput" type="search" placeholder="dataset, heuristic, path">
        </label>
      </div>
      <div id="baselineTable"></div>
    </section>
    <section>
      <h2 class="section-title">Actual replay 링크</h2>
      <div id="actualTable"></div>
    </section>
  </main>
  <script>
    const DATA = __PAYLOAD__;
    const fmt = (value, digits = 2) => Number(value || 0).toFixed(digits);
    const missingValue = value => value === undefined || value === null || value === '';
    const countText = value => missingValue(value) ? 'missing' : value;
    const statusClass = value => !missingValue(value) && Number(value) === 0 ? 'ok' : 'bad';
    function optionList(values) {
      return ['ALL', ...Array.from(new Set(values)).sort()];
    }
    function renderFilters() {
      const datasetFilter = document.getElementById('datasetFilter');
      const heuristicFilter = document.getElementById('heuristicFilter');
      datasetFilter.innerHTML = optionList(DATA.baseline_rows.map(row => row.dataset)).map(value => `<option>${value}</option>`).join('');
      heuristicFilter.innerHTML = optionList(DATA.baseline_rows.map(row => row.heuristic)).map(value => `<option>${value}</option>`).join('');
      datasetFilter.addEventListener('change', renderBaseline);
      heuristicFilter.addEventListener('change', renderBaseline);
      document.getElementById('searchInput').addEventListener('input', renderBaseline);
    }
    function renderKpis() {
      const generatedRows = DATA.baseline_rows;
      const actualRows = DATA.actual_rows;
      const bestMakespan = Math.min(...generatedRows.map(row => Number(row.makespan_minutes)));
      const hardViolations = generatedRows.reduce((sum, row) => sum + Number(row.hard_violation_count || 0), 0);
      const identityErrors = actualRows.reduce((sum, row) => sum + Number(row.identity_error_count || 0), 0);
      document.getElementById('kpis').innerHTML = [
        ['데이터셋', DATA.datasets.length],
        ['generated runs', generatedRows.length],
        ['best makespan', `${fmt(bestMakespan)}분`],
        ['hard/identity errors', `${hardViolations} / ${identityErrors}`],
      ].map(([label, value]) => `<article class="card"><small>${label}</small><div class="metric">${value}</div></article>`).join('');
      document.getElementById('basis').textContent = `${DATA.process_time_basis} ${DATA.batch_constraint}`;
    }
    function renderClean100Checklist() {
      const actual = DATA.actual_rows.find(row => row.dataset === 'clean_100');
      const generatedRows = DATA.baseline_rows.filter(row => row.dataset === 'clean_100');
      const rows = [];
      if (actual) {
        rows.push(`
          <div class="link-row">
            <div>
              <strong>actual replay</strong>
              <div class="status-line">
                scheduled=${actual.scheduled_jobs}/${actual.job_count},
                identity=${countText(actual.identity_error_count)},
                event identity=${countText(actual.event_identity_error_count)},
                event constraint=${countText(actual.event_constraint_violation_count)},
                audit=${countText(actual.constraint_audit_violation_count)}
              </div>
            </div>
            <a href="${actual.actual_replay_html}">open</a>
          </div>`);
      } else {
        rows.push(`
          <div class="link-row">
            <div>
              <strong class="bad">actual replay missing</strong>
              <div class="status-line">config_clean_100.yaml을 포함해 패키지를 다시 만들면 링크가 표시됩니다.</div>
            </div>
          </div>`);
      }
      if (generatedRows.length > 0) {
        generatedRows.forEach(row => {
          rows.push(`
            <div class="link-row">
              <div>
                <strong>${row.heuristic}</strong>
                <div class="status-line">
                  scheduled=${row.scheduled_jobs}/${row.job_count},
                  unscheduled=${countText(row.unscheduled_count)},
                  hard=${countText(row.hard_violation_count)},
                  event identity=${countText(row.event_identity_error_count)},
                  event constraint=${countText(row.event_constraint_violation_count)}
                </div>
              </div>
              <a href="${row.playback_html}">open</a>
            </div>`);
        });
      } else {
        rows.push(`
          <div class="link-row">
            <div>
              <strong class="bad">generated playback missing</strong>
              <div class="status-line">clean_100 generated run이 없어서 대체 링크를 만들지 않았습니다.</div>
            </div>
          </div>`);
      }
      document.getElementById('clean100ChecklistLinks').innerHTML = rows.join('');
    }
    function renderBaseline() {
      const dataset = document.getElementById('datasetFilter').value;
      const heuristic = document.getElementById('heuristicFilter').value;
      const query = document.getElementById('searchInput').value.toLowerCase();
      const rows = DATA.baseline_rows.filter(row => {
        const datasetOk = dataset === 'ALL' || row.dataset === dataset;
        const heuristicOk = heuristic === 'ALL' || row.heuristic === heuristic;
        const text = Object.values(row).join(' ').toLowerCase();
        return datasetOk && heuristicOk && text.includes(query);
      });
      document.getElementById('baselineTable').innerHTML = `
        <table>
          <thead><tr>
            <th>데이터</th><th>휴리스틱</th><th>완료</th><th>Makespan</th><th>Bay 시간편차</th><th>설비 시간편차</th><th>Bay 작업수편차</th><th>설비 작업수편차</th><th>Hard</th><th>Event identity error</th><th>Event constraint violation</th><th>Playback</th><th>Trace</th>
          </tr></thead>
          <tbody>
            ${rows.map(row => `
              <tr>
                <td><span class="pill">${row.dataset}</span></td>
                <td>${row.heuristic}</td>
                <td>${row.scheduled_jobs}/${row.job_count}</td>
                <td>${fmt(row.makespan_minutes)}</td>
                <td>${fmt(row.bay_load_imbalance)}</td>
                <td>${fmt(row.machine_load_imbalance)}</td>
                <td>${fmt(row.bay_job_count_imbalance)}</td>
                <td>${fmt(row.machine_job_count_imbalance)}</td>
                <td class="${statusClass(row.hard_violation_count)}">${countText(row.hard_violation_count)}</td>
                <td class="${statusClass(row.event_identity_error_count)}">${countText(row.event_identity_error_count)}</td>
                <td class="${statusClass(row.event_constraint_violation_count)}">${countText(row.event_constraint_violation_count)}</td>
                <td><a href="${row.playback_html}">open</a></td>
                <td><a href="${row.action_trace_csv}">csv</a></td>
              </tr>
            `).join('')}
          </tbody>
        </table>`;
    }
    function renderActual() {
      document.getElementById('actualTable').innerHTML = `
        <table>
          <thead><tr>
            <th>데이터</th><th>완료</th><th>Makespan</th><th>Identity error</th><th>Event identity error</th><th>Event constraint violation</th><th>Constraint audit</th><th>Actual replay</th>
          </tr></thead>
          <tbody>
            ${DATA.actual_rows.map(row => `
              <tr>
                <td><span class="pill">${row.dataset}</span></td>
                <td>${row.scheduled_jobs}/${row.job_count}</td>
                <td>${fmt(row.makespan_minutes)}</td>
                <td class="${statusClass(row.identity_error_count)}">${countText(row.identity_error_count)}</td>
                <td class="${statusClass(row.event_identity_error_count)}">${countText(row.event_identity_error_count)}</td>
                <td class="${statusClass(row.event_constraint_violation_count)}">${countText(row.event_constraint_violation_count)}</td>
                <td class="${statusClass(row.constraint_audit_violation_count)}">${countText(row.constraint_audit_violation_count)}</td>
                <td><a href="${row.actual_replay_html}">open</a></td>
              </tr>
            `).join('')}
          </tbody>
        </table>`;
    }
    renderFilters();
    renderKpis();
    renderClean100Checklist();
    renderBaseline();
    renderActual();
  </script>
</body>
</html>""".replace("__PAYLOAD__", payload)
