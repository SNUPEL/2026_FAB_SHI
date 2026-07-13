"""Pygame 기반 공장 playback viewer.

이 모듈은 DES가 만든 산출물을 다시 시뮬레이션하지 않는다. 이미 검증된
`event_log.json`, `factory_layout.json`, `job_schedule.csv`, `metrics.json`를
read-only로 읽어서 사람이 공장 흐름을 눈으로 확인할 수 있게 표시한다.

입력 예시:
  event_log.json      : DES event schema 원본. event 개수와 source를 확인한다.
  factory_layout.json : Bay와 설비 목록. 설비 card 배치에 사용한다.
  job_schedule.csv    : W/O별 machine, batch, start/finish minute. 화면의 실제 작업 막대다.
  metrics.json        : scheduled_jobs, makespan, Bay/설비 load 지표. 우측 KPI에 표시한다.

실행 예시:
  python3 main.py pygame-viewer \
    --event-log output/share/html_viewer_package/clean_100/generated/balanced_batch/event_log.json \
    --layout output/share/html_viewer_package/clean_100/generated/balanced_batch/factory_layout.json \
    --schedule output/share/html_viewer_package/clean_100/generated/balanced_batch/job_schedule.csv \
    --metrics output/share/html_viewer_package/clean_100/generated/balanced_batch/metrics.json
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ViewerMachine:
    """화면에 표시할 절단 설비 1대.

    `machine_id`는 PLS21 같은 설비명이고, `bay_id`는 해당 설비가 속한 절단 Bay다.
    `position`은 layout JSON에 들어 있는 상대 좌표이며, 없으면 화면에서 정렬 순서만 쓴다.
    """

    machine_id: str
    bay_id: str
    machine_type: str
    position: tuple[float, float] | None
    parallel_capacity: int


@dataclass(frozen=True)
class ViewerOperation:
    """화면에서 시간축 위에 재생할 W/O 작업 1건.

    한 batch에 W/O가 2~3개 있으면 `ViewerOperation`도 W/O별로 2~3개 생긴다.
    같은 batch의 W/O들은 같은 `machine_id`, `start_time_min`, `finish_time_min`를 가져야 한다.
    """

    job_id: str
    work_order_no: str
    machine_id: str
    bay_id: str
    batch_id: str
    batch_job_ids: tuple[str, ...]
    batch_wo_count: int
    batch_length_sum: float
    start_time_min: float
    finish_time_min: float
    process_min: float
    plate_length: float
    thickness: float
    source_machine_id: str
    source_cut_bay: str

    def progress_ratio(self, current_time_min: float) -> float:
        """현재 시각에서 이 W/O가 몇 % 진행됐는지 0~1 범위로 반환한다."""

        duration = self.finish_time_min - self.start_time_min
        if duration <= 0:
            print(
                "[ERROR][ViewerOperation.progress_ratio] "
                f"cause=non_positive_duration job_id={self.job_id} "
                f"start={self.start_time_min} finish={self.finish_time_min}"
            )
            raise ValueError(f"operation duration must be positive: {self.job_id}")
        if current_time_min <= self.start_time_min:
            return 0.0
        if current_time_min >= self.finish_time_min:
            return 1.0
        return (current_time_min - self.start_time_min) / duration


@dataclass(frozen=True)
class ViewerData:
    """Pygame viewer가 쓰는 검증 완료 입력 묶음."""

    event_log_path: Path
    layout_path: Path
    schedule_path: Path
    metrics_path: Path
    machines: tuple[ViewerMachine, ...]
    operations: tuple[ViewerOperation, ...]
    metrics: dict[str, Any]
    event_count: int
    start_time_min: float
    end_time_min: float


@dataclass(frozen=True)
class WorkloadSummary:
    """화면 비교용 부하평준화 지표.

    `work_order_counts`는 W/O row 개수다. `workload_minutes`는 W/O별 시간을
    단순 합산하지 않고, 같은 batch가 같은 설비를 점유한 시간을 한 번만 더한다.
    이렇게 분리해야 "할당 작업개수 부하평준화"와 "설비 점유시간 부하평준화"를
    같은 화면에서 비교할 수 있다.
    """

    machine_workload_minutes: dict[str, float]
    machine_work_order_counts: dict[str, int]
    machine_batch_counts: dict[str, int]
    bay_workload_minutes: dict[str, float]
    bay_work_order_counts: dict[str, int]
    bay_batch_counts: dict[str, int]
    machine_workload_imbalance_minutes: float
    machine_work_order_imbalance: int
    machine_batch_imbalance: int
    bay_workload_imbalance_minutes: float
    bay_work_order_imbalance: int
    bay_batch_imbalance: int


@dataclass(frozen=True)
class CompletionSummary:
    """현재 playback 시각까지 완료된 W/O와 batch 수.

    완료 기준은 `finish_time_min <= current_time_min`이다. Batch는 같은
    `batch_id + machine + start + finish` window를 한 번만 센다.
    """

    machine_completed_work_orders: dict[str, int]
    machine_completed_batches: dict[str, int]
    bay_completed_work_orders: dict[str, int]
    bay_completed_batches: dict[str, int]


def _require_path(path: str | Path, label: str) -> Path:
    """입력 파일이 실제로 있는지 확인하고 없으면 원인을 출력한 뒤 실패한다."""

    resolved = Path(path)
    if not resolved.exists():
        print(f"[ERROR][pygame_factory_viewer._require_path] cause=missing_{label} path={resolved}")
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not resolved.is_file():
        print(f"[ERROR][pygame_factory_viewer._require_path] cause=not_file label={label} path={resolved}")
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _read_json(path: Path, label: str) -> Any:
    """JSON 파일을 읽고 parsing 실패 시 어느 파일이 문제인지 출력한다."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(
            "[ERROR][pygame_factory_viewer._read_json] "
            f"cause=json_decode_error label={label} path={path} message={exc}"
        )
        raise


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    """CSV를 dict row 목록으로 읽는다. BOM이 있어도 컬럼명이 깨지지 않도록 utf-8-sig를 쓴다."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            return list(csv.DictReader(file_obj))
    except csv.Error as exc:
        print(
            "[ERROR][pygame_factory_viewer._read_csv_rows] "
            f"cause=csv_error label={label} path={path} message={exc}"
        )
        raise


def _require_text(row: dict[str, str], field: str, row_key: str) -> str:
    """CSV row에서 빈 문자열이 아닌 필수 문자열을 읽는다."""

    value = row.get(field)
    if value is None or str(value).strip() == "":
        print(
            "[ERROR][pygame_factory_viewer._require_text] "
            f"cause=missing_field field={field} row_key={row_key}"
        )
        raise ValueError(f"missing required field {field} in {row_key}")
    return str(value).strip()


def _has_text(row: dict[str, str], field: str) -> bool:
    """CSV row에 공백이 아닌 값이 있는지 확인한다."""

    value = row.get(field)
    return value is not None and str(value).strip() != ""


def _parse_float(row: dict[str, str], field: str, row_key: str) -> float:
    """CSV row의 숫자 필드를 float로 읽는다. 실패하면 어떤 row/field인지 출력한다."""

    raw_value = _require_text(row, field, row_key)
    try:
        return float(raw_value)
    except ValueError as exc:
        print(
            "[ERROR][pygame_factory_viewer._parse_float] "
            f"cause=float_parse_failed field={field} row_key={row_key} value={raw_value}"
        )
        raise ValueError(f"cannot parse float field {field}={raw_value!r} in {row_key}") from exc


def _parse_int(row: dict[str, str], field: str, row_key: str) -> int:
    """CSV row의 정수 필드를 int로 읽는다. `2.0` 같은 표기도 허용하지 않는다."""

    raw_value = _require_text(row, field, row_key)
    try:
        return int(raw_value)
    except ValueError as exc:
        print(
            "[ERROR][pygame_factory_viewer._parse_int] "
            f"cause=int_parse_failed field={field} row_key={row_key} value={raw_value}"
        )
        raise ValueError(f"cannot parse int field {field}={raw_value!r} in {row_key}") from exc


def _split_batch_job_ids(raw_value: str, row_key: str) -> tuple[str, ...]:
    """`job_a|job_b|job_c` 형태 batch W/O 목록을 tuple로 바꾼다."""

    job_ids = tuple(item.strip() for item in raw_value.split("|") if item.strip())
    if not job_ids:
        print(f"[ERROR][pygame_factory_viewer._split_batch_job_ids] cause=empty_batch_job_ids row_key={row_key}")
        raise ValueError(f"empty batch_job_ids in {row_key}")
    return job_ids


def _is_actual_replay_row(row: dict[str, str]) -> bool:
    """actual replay schedule row인지 명시 flag로 판정한다."""

    return str(row.get("replay_mode", "")).strip() == "actual_historical"


def _read_batch_display_fields(
    row: dict[str, str],
    job_id: str,
    row_key: str,
    plate_length: float,
) -> tuple[str, tuple[str, ...], int, float]:
    """generated/actual schedule의 batch 표시 필드를 읽는다.

    generated schedule은 `batch_id`, `batch_job_ids`, `batch_wo_count`,
    `batch_length_sum`이 필수다. actual replay schedule은 실적 timestamp를 그대로
    재생하는 row라 batch 컬럼이 없을 수 있다. 이때는 `replay_mode=actual_historical`
    가 명시된 경우에만 W/O 1건을 표시용 batch 1건으로 변환한다.
    """

    batch_fields = ("batch_id", "batch_job_ids", "batch_wo_count", "batch_length_sum")
    missing_fields = [field for field in batch_fields if not _has_text(row, field)]
    if not missing_fields:
        batch_job_ids = _split_batch_job_ids(_require_text(row, "batch_job_ids", row_key), row_key)
        return (
            _require_text(row, "batch_id", row_key),
            batch_job_ids,
            _parse_int(row, "batch_wo_count", row_key),
            _parse_float(row, "batch_length_sum", row_key),
        )

    if _is_actual_replay_row(row) and len(missing_fields) == len(batch_fields):
        batch_id = f"actual:{job_id}"
        return batch_id, (job_id,), 1, float(plate_length)

    if _is_actual_replay_row(row):
        print(
            "[ERROR][pygame_factory_viewer._read_batch_display_fields] "
            f"cause=partial_actual_batch_fields row_key={row_key} missing={missing_fields}"
        )
        raise ValueError(f"actual replay row has partial batch fields in {row_key}: {missing_fields}")

    return (
        _require_text(row, "batch_id", row_key),
        _split_batch_job_ids(_require_text(row, "batch_job_ids", row_key), row_key),
        _parse_int(row, "batch_wo_count", row_key),
        _parse_float(row, "batch_length_sum", row_key),
    )


def _load_machines(layout_path: Path) -> tuple[ViewerMachine, ...]:
    """factory_layout.json에서 Bay와 설비 목록을 읽는다."""

    layout = _read_json(layout_path, "layout")
    if not isinstance(layout, dict):
        print(f"[ERROR][pygame_factory_viewer._load_machines] cause=layout_not_object path={layout_path}")
        raise ValueError(f"layout must be a JSON object: {layout_path}")

    bays = layout.get("bays")
    if not isinstance(bays, list) or not bays:
        print(f"[ERROR][pygame_factory_viewer._load_machines] cause=missing_bays path={layout_path}")
        raise ValueError(f"layout.bays must be a non-empty list: {layout_path}")

    machines: list[ViewerMachine] = []
    for bay in bays:
        if not isinstance(bay, dict):
            print(f"[ERROR][pygame_factory_viewer._load_machines] cause=bay_not_object value={bay}")
            raise ValueError("each layout bay must be an object")
        bay_id = str(bay.get("bay_id", "")).strip()
        if not bay_id:
            print(f"[ERROR][pygame_factory_viewer._load_machines] cause=missing_bay_id bay={bay}")
            raise ValueError("layout bay_id is required")
        bay_machines = bay.get("machines")
        if not isinstance(bay_machines, list):
            print(f"[ERROR][pygame_factory_viewer._load_machines] cause=machines_not_list bay_id={bay_id}")
            raise ValueError(f"layout bay machines must be a list: {bay_id}")
        for raw_machine in bay_machines:
            if not isinstance(raw_machine, dict):
                print(
                    "[ERROR][pygame_factory_viewer._load_machines] "
                    f"cause=machine_not_object bay_id={bay_id} value={raw_machine}"
                )
                raise ValueError(f"layout machine must be an object in bay {bay_id}")
            machine_id = str(raw_machine.get("machine_id", "")).strip()
            if not machine_id:
                print(f"[ERROR][pygame_factory_viewer._load_machines] cause=missing_machine_id bay_id={bay_id}")
                raise ValueError(f"layout machine_id is required in bay {bay_id}")
            position = raw_machine.get("position")
            parsed_position: tuple[float, float] | None = None
            if position is not None:
                if not isinstance(position, list) or len(position) != 2:
                    print(
                        "[ERROR][pygame_factory_viewer._load_machines] "
                        f"cause=invalid_position machine_id={machine_id} position={position}"
                    )
                    raise ValueError(f"layout position must be [x, y] for {machine_id}")
                parsed_position = (float(position[0]), float(position[1]))
            machines.append(
                ViewerMachine(
                    machine_id=machine_id,
                    bay_id=bay_id,
                    machine_type=str(raw_machine.get("machine_type", "")).strip(),
                    position=parsed_position,
                    parallel_capacity=int(raw_machine.get("parallel_capacity", 1)),
                )
            )

    if not machines:
        print(f"[ERROR][pygame_factory_viewer._load_machines] cause=no_machines path={layout_path}")
        raise ValueError(f"layout has no machines: {layout_path}")
    return tuple(machines)


def _load_operations(schedule_path: Path) -> tuple[ViewerOperation, ...]:
    """job_schedule.csv에서 화면에 재생할 W/O operation 목록을 읽는다."""

    rows = _read_csv_rows(schedule_path, "schedule")
    if not rows:
        print(f"[ERROR][pygame_factory_viewer._load_operations] cause=empty_schedule path={schedule_path}")
        raise ValueError(f"schedule csv is empty: {schedule_path}")

    operations: list[ViewerOperation] = []
    actual_display_batch_count = 0
    for row_index, row in enumerate(rows, start=2):
        row_key = f"excel_row={row_index}"
        job_id = _require_text(row, "job_id", row_key)
        row_key = f"job_id={job_id}"
        plate_length = _parse_float(row, "plate_length", row_key)
        batch_id, batch_job_ids, batch_wo_count, batch_length_sum = _read_batch_display_fields(
            row,
            job_id,
            row_key,
            plate_length,
        )
        if batch_id == f"actual:{job_id}":
            actual_display_batch_count += 1
        operation = ViewerOperation(
            job_id=job_id,
            work_order_no=_require_text(row, "work_order_no", row_key),
            machine_id=_require_text(row, "algorithm_machine_id", row_key),
            bay_id=_require_text(row, "algorithm_cut_bay", row_key),
            batch_id=batch_id,
            batch_job_ids=batch_job_ids,
            batch_wo_count=batch_wo_count,
            batch_length_sum=batch_length_sum,
            start_time_min=_parse_float(row, "start_min", row_key),
            finish_time_min=_parse_float(row, "finish_min", row_key),
            process_min=_parse_float(row, "process_min", row_key),
            plate_length=plate_length,
            thickness=_parse_float(row, "thickness", row_key),
            source_machine_id=str(row.get("source_machine_id", "")).strip(),
            source_cut_bay=str(row.get("source_cut_bay", "")).strip(),
        )
        if operation.finish_time_min < operation.start_time_min:
            print(
                "[ERROR][pygame_factory_viewer._load_operations] "
                f"cause=finish_before_start job_id={job_id} "
                f"start={operation.start_time_min} finish={operation.finish_time_min}"
            )
            raise ValueError(f"finish before start for {job_id}")
        if operation.batch_wo_count != len(operation.batch_job_ids):
            print(
                "[ERROR][pygame_factory_viewer._load_operations] "
                f"cause=batch_count_mismatch job_id={job_id} "
                f"batch_id={operation.batch_id} count={operation.batch_wo_count} ids={operation.batch_job_ids}"
            )
            raise ValueError(f"batch count mismatch for {job_id}")
        operations.append(operation)

    if actual_display_batch_count:
        print(
            "[CHECK][pygame_factory_viewer._load_operations] "
            f"schedule_schema=actual_replay display_batches={actual_display_batch_count}"
        )
    return tuple(operations)


def _load_event_count(event_log_path: Path) -> int:
    """event_log.json에서 event 개수를 읽는다."""

    event_log = _read_json(event_log_path, "event_log")
    if not isinstance(event_log, list):
        print(f"[ERROR][pygame_factory_viewer._load_event_count] cause=event_log_not_list path={event_log_path}")
        raise ValueError(f"event log must be a JSON list: {event_log_path}")
    return len(event_log)


def _load_metrics(metrics_path: Path) -> dict[str, Any]:
    """metrics.json을 읽는다. KPI 표시용이므로 dict가 아니면 실패한다."""

    metrics = _read_json(metrics_path, "metrics")
    if not isinstance(metrics, dict):
        print(f"[ERROR][pygame_factory_viewer._load_metrics] cause=metrics_not_object path={metrics_path}")
        raise ValueError(f"metrics must be a JSON object: {metrics_path}")
    return metrics


def _validate_machine_mapping(machines: tuple[ViewerMachine, ...], operations: tuple[ViewerOperation, ...]) -> None:
    """schedule의 모든 설비가 layout에 존재하고 Bay가 일치하는지 검증한다."""

    machine_by_id = {machine.machine_id: machine for machine in machines}
    for operation in operations:
        machine = machine_by_id.get(operation.machine_id)
        if machine is None:
            print(
                "[ERROR][pygame_factory_viewer._validate_machine_mapping] "
                f"cause=machine_missing_in_layout job_id={operation.job_id} machine_id={operation.machine_id}"
            )
            raise ValueError(f"schedule machine is missing in layout: {operation.machine_id}")
        if str(machine.bay_id) != str(operation.bay_id):
            print(
                "[ERROR][pygame_factory_viewer._validate_machine_mapping] "
                f"cause=bay_mismatch job_id={operation.job_id} machine_id={operation.machine_id} "
                f"layout_bay={machine.bay_id} schedule_bay={operation.bay_id}"
            )
            raise ValueError(f"schedule bay differs from layout for {operation.machine_id}")


def _validate_batch_consistency(operations: tuple[ViewerOperation, ...]) -> None:
    """같은 batch 안 W/O들이 같은 설비/시작/종료/길이합을 공유하는지 확인한다."""

    by_batch: dict[str, list[ViewerOperation]] = {}
    for operation in operations:
        by_batch.setdefault(operation.batch_id, []).append(operation)

    for batch_id, batch_operations in by_batch.items():
        first = batch_operations[0]
        if len(batch_operations) != first.batch_wo_count:
            print(
                "[ERROR][pygame_factory_viewer._validate_batch_consistency] "
                f"cause=batch_row_count_mismatch batch_id={batch_id} "
                f"row_count={len(batch_operations)} declared={first.batch_wo_count}"
            )
            raise ValueError(f"batch row count mismatch: {batch_id}")
        for operation in batch_operations[1:]:
            same_machine = operation.machine_id == first.machine_id
            same_start = abs(operation.start_time_min - first.start_time_min) < 1e-9
            same_finish = abs(operation.finish_time_min - first.finish_time_min) < 1e-9
            same_length = abs(operation.batch_length_sum - first.batch_length_sum) < 1e-9
            if not (same_machine and same_start and same_finish and same_length):
                print(
                    "[ERROR][pygame_factory_viewer._validate_batch_consistency] "
                    f"cause=inconsistent_batch_fields batch_id={batch_id} job_id={operation.job_id}"
                )
                raise ValueError(f"inconsistent batch fields: {batch_id}")


def load_viewer_data(
    event_log_path: str | Path,
    layout_path: str | Path,
    schedule_path: str | Path,
    metrics_path: str | Path,
) -> ViewerData:
    """Pygame viewer 입력 파일 4종을 읽고 내부 dataclass로 변환한다."""

    event_log_file = _require_path(event_log_path, "event_log")
    layout_file = _require_path(layout_path, "layout")
    schedule_file = _require_path(schedule_path, "schedule")
    metrics_file = _require_path(metrics_path, "metrics")

    event_count = _load_event_count(event_log_file)
    machines = _load_machines(layout_file)
    operations = _load_operations(schedule_file)
    metrics = _load_metrics(metrics_file)
    _validate_machine_mapping(machines, operations)
    _validate_batch_consistency(operations)

    start_time_min = min(operation.start_time_min for operation in operations)
    end_time_min = max(operation.finish_time_min for operation in operations)
    data = ViewerData(
        event_log_path=event_log_file,
        layout_path=layout_file,
        schedule_path=schedule_file,
        metrics_path=metrics_file,
        machines=machines,
        operations=operations,
        metrics=metrics,
        event_count=event_count,
        start_time_min=start_time_min,
        end_time_min=end_time_min,
    )
    print(
        "[CHECK][pygame_factory_viewer.load_viewer_data] "
        f"operations={len(operations)} machines={len(machines)} events={event_count} "
        f"start={start_time_min:.2f} end={end_time_min:.2f}"
    )
    return data


def validate_viewer_data(data: ViewerData) -> dict[str, Any]:
    """viewer 입력 구조가 화면 재생 가능한 최소 조건을 만족하는지 검증한다."""

    if not data.operations:
        print("[ERROR][pygame_factory_viewer.validate_viewer_data] cause=no_operations")
        raise ValueError("viewer data has no operations")
    if not data.machines:
        print("[ERROR][pygame_factory_viewer.validate_viewer_data] cause=no_machines")
        raise ValueError("viewer data has no machines")
    if data.end_time_min < data.start_time_min:
        print(
            "[ERROR][pygame_factory_viewer.validate_viewer_data] "
            f"cause=invalid_time_range start={data.start_time_min} end={data.end_time_min}"
        )
        raise ValueError("viewer data has invalid time range")

    batch_ids = sorted({operation.batch_id for operation in data.operations})
    bay_ids = sorted({machine.bay_id for machine in data.machines})
    summary = {
        "operation_count": len(data.operations),
        "machine_count": len(data.machines),
        "batch_count": len(batch_ids),
        "bay_ids": bay_ids,
        "event_count": data.event_count,
        "start_time_min": data.start_time_min,
        "end_time_min": data.end_time_min,
        "scheduled_jobs": data.metrics.get("scheduled_jobs", len(data.operations)),
        "makespan_minutes": data.metrics.get("makespan_minutes", data.end_time_min),
    }
    print(
        "[VALIDATION][pygame_factory_viewer.validate_viewer_data] "
        f"passed=true operations={summary['operation_count']} machines={summary['machine_count']} "
        f"batches={summary['batch_count']} events={summary['event_count']}"
    )
    return summary


def build_active_machine_view(data: ViewerData, current_time_min: float) -> dict[str, list[ViewerOperation]]:
    """특정 simulation minute에서 설비별 active W/O 목록을 만든다."""

    active_by_machine = {machine.machine_id: [] for machine in data.machines}
    for operation in data.operations:
        if operation.start_time_min <= current_time_min < operation.finish_time_min:
            active_by_machine[operation.machine_id].append(operation)
    for active_operations in active_by_machine.values():
        active_operations.sort(key=lambda item: (item.batch_id, item.job_id))
    return active_by_machine


def _numeric_imbalance(values: dict[str, float] | dict[str, int]) -> float:
    """부하평준화 지표인 max-min 차이를 계산한다."""

    if not values:
        return 0.0
    numeric_values = [float(value) for value in values.values()]
    return max(numeric_values) - min(numeric_values)


def _batch_window_key(operation: ViewerOperation) -> tuple[str, str, float, float]:
    """Batch count와 batch 처리분 중복 합산을 막기 위한 동일 window key."""

    return (
        operation.batch_id,
        operation.machine_id,
        operation.start_time_min,
        operation.finish_time_min,
    )


def build_workload_summary(data: ViewerData) -> WorkloadSummary:
    """schedule 기준으로 장비/Bay 부하평준화 표시 지표를 계산한다.

    metrics.json에 있는 값을 그대로 보여주면 어떤 산식인지 숨겨질 수 있다.
    viewer에서는 layout에 보이는 모든 장비와 Bay를 기준으로, 작업개수는 W/O
    row 수로 세고 처리분 부하는 batch window를 한 번만 더해 비교한다.
    """

    machine_to_bay = {machine.machine_id: machine.bay_id for machine in data.machines}
    machine_workload_minutes = {machine.machine_id: 0.0 for machine in data.machines}
    machine_work_order_counts = {machine.machine_id: 0 for machine in data.machines}
    machine_batch_counts = {machine.machine_id: 0 for machine in data.machines}
    bay_workload_minutes = {machine.bay_id: 0.0 for machine in data.machines}
    bay_work_order_counts = {machine.bay_id: 0 for machine in data.machines}
    bay_batch_counts = {machine.bay_id: 0 for machine in data.machines}
    counted_batch_windows: set[tuple[str, str, float, float]] = set()

    for operation in data.operations:
        bay_id = machine_to_bay.get(operation.machine_id)
        if bay_id is None:
            print(
                "[ERROR][pygame_factory_viewer.build_workload_summary] "
                f"cause=machine_missing_in_layout job_id={operation.job_id} machine_id={operation.machine_id}"
            )
            raise ValueError(f"operation machine is missing in layout: {operation.machine_id}")
        if str(bay_id) != str(operation.bay_id):
            print(
                "[ERROR][pygame_factory_viewer.build_workload_summary] "
                f"cause=bay_mismatch job_id={operation.job_id} machine_id={operation.machine_id} "
                f"layout_bay={bay_id} schedule_bay={operation.bay_id}"
            )
            raise ValueError(f"operation bay differs from layout: {operation.machine_id}")

        machine_work_order_counts[operation.machine_id] += 1
        bay_work_order_counts[bay_id] += 1

        batch_window_key = _batch_window_key(operation)
        if batch_window_key in counted_batch_windows:
            continue
        counted_batch_windows.add(batch_window_key)
        machine_batch_counts[operation.machine_id] += 1
        bay_batch_counts[bay_id] += 1

        duration_min = operation.finish_time_min - operation.start_time_min
        if duration_min <= 0:
            print(
                "[ERROR][pygame_factory_viewer.build_workload_summary] "
                f"cause=non_positive_batch_duration batch_id={operation.batch_id} "
                f"machine_id={operation.machine_id} start={operation.start_time_min} finish={operation.finish_time_min}"
            )
            raise ValueError(f"batch duration must be positive: {operation.batch_id}")
        machine_workload_minutes[operation.machine_id] += duration_min
        bay_workload_minutes[bay_id] += duration_min

    return WorkloadSummary(
        machine_workload_minutes=machine_workload_minutes,
        machine_work_order_counts=machine_work_order_counts,
        machine_batch_counts=machine_batch_counts,
        bay_workload_minutes=bay_workload_minutes,
        bay_work_order_counts=bay_work_order_counts,
        bay_batch_counts=bay_batch_counts,
        machine_workload_imbalance_minutes=_numeric_imbalance(machine_workload_minutes),
        machine_work_order_imbalance=int(_numeric_imbalance(machine_work_order_counts)),
        machine_batch_imbalance=int(_numeric_imbalance(machine_batch_counts)),
        bay_workload_imbalance_minutes=_numeric_imbalance(bay_workload_minutes),
        bay_work_order_imbalance=int(_numeric_imbalance(bay_work_order_counts)),
        bay_batch_imbalance=int(_numeric_imbalance(bay_batch_counts)),
    )


def build_completion_summary(data: ViewerData, current_time_min: float) -> CompletionSummary:
    """현재 시각까지 완료된 W/O와 batch 수를 Bay/설비별로 계산한다."""

    machine_to_bay = {machine.machine_id: machine.bay_id for machine in data.machines}
    machine_completed_work_orders = {machine.machine_id: 0 for machine in data.machines}
    machine_completed_batches = {machine.machine_id: 0 for machine in data.machines}
    bay_completed_work_orders = {machine.bay_id: 0 for machine in data.machines}
    bay_completed_batches = {machine.bay_id: 0 for machine in data.machines}
    counted_completed_batches: set[tuple[str, str, float, float]] = set()

    for operation in data.operations:
        bay_id = machine_to_bay.get(operation.machine_id)
        if bay_id is None:
            print(
                "[ERROR][pygame_factory_viewer.build_completion_summary] "
                f"cause=machine_missing_in_layout job_id={operation.job_id} machine_id={operation.machine_id}"
            )
            raise ValueError(f"operation machine is missing in layout: {operation.machine_id}")
        if operation.finish_time_min > current_time_min:
            continue

        machine_completed_work_orders[operation.machine_id] += 1
        bay_completed_work_orders[bay_id] += 1

        batch_window_key = _batch_window_key(operation)
        if batch_window_key in counted_completed_batches:
            continue
        counted_completed_batches.add(batch_window_key)
        machine_completed_batches[operation.machine_id] += 1
        bay_completed_batches[bay_id] += 1

    return CompletionSummary(
        machine_completed_work_orders=machine_completed_work_orders,
        machine_completed_batches=machine_completed_batches,
        bay_completed_work_orders=bay_completed_work_orders,
        bay_completed_batches=bay_completed_batches,
    )


def build_run_summary_lines(data: ViewerData, workload: WorkloadSummary) -> list[str]:
    """화면 우측 요약 패널에 표시할 사람이 읽는 label을 만든다."""

    hard_violations = data.metrics.get("hard_violation_count", "not reported")
    return [
        "Scope: NP only",
        "Metric: W/O + Batch count",
        "Leveling rule: max-min",
        "Lower is better",
        f"Work orders: {len(data.operations)}",
        f"Batches: {len({operation.batch_id for operation in data.operations})}",
        f"Bay W/O leveling: {workload.bay_work_order_imbalance}",
        f"Machine W/O leveling: {workload.machine_work_order_imbalance}",
        f"Bay Batch leveling: {workload.bay_batch_imbalance}",
        f"Machine Batch leveling: {workload.machine_batch_imbalance}",
        f"Finish span sub: {data.end_time_min - data.start_time_min:.1f}m",
        f"Hard violations: {hard_violations}",
    ]


def _capacity_warning(active_operations: list[ViewerOperation]) -> bool:
    """화면 표시용 capacity warning 여부를 계산한다.

    actual replay는 과거 데이터 오류나 제약 위반 후보도 그대로 봐야 하므로 여기서 실패시키지 않는다.
    대신 현재 시점 active W/O가 3개를 넘거나 길이 합이 55,000을 넘으면 빨간 card로 표시한다.
    """

    if len(active_operations) > 3:
        return True
    active_length_sum = sum(operation.plate_length for operation in active_operations)
    return active_length_sum > 55000.0


def _group_machines_by_bay(data: ViewerData) -> dict[str, list[ViewerMachine]]:
    """화면 배치를 위해 Bay별 설비 목록을 만든다."""

    grouped: dict[str, list[ViewerMachine]] = {}
    for machine in data.machines:
        grouped.setdefault(machine.bay_id, []).append(machine)
    for machines in grouped.values():
        machines.sort(key=lambda item: item.machine_id)
    return dict(sorted(grouped.items()))


def _format_minute(value: float) -> str:
    """simulation minute를 `Day 0 03:20` 형태로 표시한다."""

    safe_value = max(0.0, float(value))
    day = int(safe_value // 1440)
    minute_of_day = int(safe_value % 1440)
    hour = minute_of_day // 60
    minute = minute_of_day % 60
    return f"Day {day} {hour:02d}:{minute:02d}"


def _short_work_order(value: str, max_len: int = 24) -> str:
    """W/O명이 너무 길 때 설비 card에서 읽을 수 있는 길이로 줄인다."""

    if len(value) <= max_len:
        return value
    return f"...{value[-(max_len - 3):]}"


def _fit_text_to_width(font: Any, text: Any, max_width: int) -> str:
    """Pygame font 기준으로 텍스트가 지정 폭을 넘지 않게 뒤쪽 정보를 보존해 줄인다."""

    text_value = str(text)
    if max_width <= 0:
        return ""
    if font.size(text_value)[0] <= max_width:
        return text_value
    marker = "..."
    if font.size(marker)[0] > max_width:
        return ""

    low = 0
    high = len(text_value)
    best = marker
    while low <= high:
        mid = (low + high) // 2
        suffix = text_value[-mid:] if mid else ""
        candidate = f"{marker}{suffix}"
        if font.size(candidate)[0] <= max_width:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _font_height(font: Any) -> int:
    """Pygame font 높이를 읽는다. 테스트 stub은 get_height가 없을 수 있어 size로 계산한다."""

    if hasattr(font, "get_height"):
        return int(font.get_height())
    return int(font.size("Ag")[1])


def _format_metric_value(value: Any) -> str:
    """우측 metric grid에 넣을 값을 짧고 읽기 좋게 바꾼다."""

    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _draw_text_fit(
    surface: Any,
    font: Any,
    text: Any,
    position: tuple[int, int],
    color: tuple[int, int, int],
    max_width: int,
) -> None:
    """지정 폭을 넘는 문자열을 줄인 뒤 Pygame surface에 그린다."""

    clipped_text = _fit_text_to_width(font, text, max_width)
    surface.blit(font.render(clipped_text, True, color), position)


def _draw_metric_grid(
    pygame: Any,
    surface: Any,
    title_font: Any,
    item_font: Any,
    title: str,
    values: Any,
    rect: Any,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    """설비별 metric을 정해진 rect 안에만 1~2열로 표시한다."""

    if not isinstance(values, dict) or rect.height <= 0 or rect.width <= 0:
        return

    previous_clip = surface.get_clip()
    surface.set_clip(rect)
    _draw_text_fit(surface, title_font, title, (rect.left, rect.top), colors["text"], rect.width)

    items = sorted(values.items())
    line_height = max(_font_height(item_font) + 3, 15)
    top = rect.top + _font_height(title_font) + 8
    available_height = max(rect.bottom - top, 0)
    columns = 2 if rect.width >= 210 else 1
    column_gap = 12
    column_width = int((rect.width - column_gap * (columns - 1)) / columns)
    rows_per_column = max(available_height // line_height, 0)
    capacity = rows_per_column * columns
    if capacity <= 0:
        surface.set_clip(previous_clip)
        return

    visible_items = items[:capacity]
    if len(items) > capacity:
        visible_items = items[: max(capacity - 1, 0)]
        visible_items.append(("+more", len(items) - len(visible_items)))

    for item_index, (key, value) in enumerate(visible_items):
        column_index = item_index // rows_per_column if rows_per_column else 0
        row_index = item_index % rows_per_column if rows_per_column else 0
        x = rect.left + column_index * (column_width + column_gap)
        y = top + row_index * line_height
        if key == "+more":
            text = f"+{value} more rows"
        else:
            text = f"{key}: {_format_metric_value(value)}"
        _draw_text_fit(surface, item_font, text, (x, y), colors["muted"], column_width)

    surface.set_clip(previous_clip)


def _draw_factory_frame(
    pygame: Any,
    surface: Any,
    data: ViewerData,
    current_time: float,
    playback_speed: float,
    paused: bool,
    fonts: dict[str, Any],
    colors: dict[str, tuple[int, int, int]],
    title: str,
    workload: WorkloadSummary | None = None,
) -> None:
    """단일 Pygame surface 안에 공장 상태 1 frame을 그린다."""

    width, height = surface.get_size()
    active_view = build_active_machine_view(data, current_time)
    workload_summary = workload if workload is not None else build_workload_summary(data)
    completion_summary = build_completion_summary(data, current_time)
    surface.fill(colors["bg"])

    title_font = fonts["title"]
    header_font = fonts["header"]
    body_font = fonts["body"]
    small_font = fonts["small"]

    _draw_text_fit(surface, title_font, title, (24, 22), colors["text"], width - 48)
    status = "Paused" if paused else "Playing"
    _draw_text_fit(
        surface,
        header_font,
        f"{status} | NP only | W/O+Batch leveling | t={current_time:.1f}m ({_format_minute(current_time)}) | speed {playback_speed:.1f}m/s",
        (24, 56),
        colors["accent"] if paused else colors["progress"],
        width - 48,
    )
    _draw_text_fit(
        surface,
        small_font,
        "Controls: Space pause/resume | Left/Right time seek | Up/Down speed | Home/End | Click timeline | Esc quit",
        (24, 84),
        colors["muted"],
        width - 48,
    )

    floor_left = 24
    floor_top = 118
    floor_width = max(420, int(width * 0.67))
    if width - (floor_left + floor_width + 24) < 260:
        floor_width = max(420, width - floor_left - 24)
    floor_height = height - 208
    grouped_machines = _group_machines_by_bay(data)
    bay_count = max(1, len(grouped_machines))
    bay_gap = 14
    bay_width = max(96, int((floor_width - bay_gap * (bay_count - 1)) / bay_count))

    for bay_index, (bay_id, machines) in enumerate(grouped_machines.items()):
        bay_x = floor_left + bay_index * (bay_width + bay_gap)
        bay_rect = pygame.Rect(bay_x, floor_top, bay_width, floor_height)
        pygame.draw.rect(surface, colors["panel"], bay_rect, border_radius=12)
        pygame.draw.rect(surface, colors["line"], bay_rect, width=2, border_radius=12)
        _draw_text_fit(surface, header_font, f"Bay {bay_id}", (bay_x + 14, floor_top + 12), colors["text"], bay_width - 28)
        _draw_text_fit(
            surface,
            small_font,
            f"Done W/O {completion_summary.bay_completed_work_orders[bay_id]}/{workload_summary.bay_work_order_counts[bay_id]}",
            (bay_x + 14, floor_top + 34),
            colors["muted"],
            bay_width - 28,
        )
        _draw_text_fit(
            surface,
            small_font,
            f"Batch {completion_summary.bay_completed_batches[bay_id]}/{workload_summary.bay_batch_counts[bay_id]}",
            (bay_x + 14, floor_top + 52),
            colors["muted"],
            bay_width - 28,
        )

        card_gap = 12
        card_top = floor_top + 76
        card_height = int((floor_height - 92 - card_gap * (len(machines) - 1)) / max(1, len(machines)))
        card_height = max(96, min(118, card_height))
        for machine_index, machine in enumerate(machines):
            card_x = bay_x + 14
            card_y = card_top + machine_index * (card_height + card_gap)
            card_rect = pygame.Rect(card_x, card_y, bay_width - 28, card_height)
            active_operations = active_view[machine.machine_id]
            if _capacity_warning(active_operations):
                fill_color = colors["danger"]
            elif active_operations:
                fill_color = colors["active"]
            else:
                fill_color = colors["idle"]
            pygame.draw.rect(surface, fill_color, card_rect, border_radius=10)
            pygame.draw.rect(surface, colors["line"], card_rect, width=1, border_radius=10)

            previous_clip = surface.get_clip()
            surface.set_clip(card_rect)
            text_width = max(card_rect.width - 20, 20)
            _draw_text_fit(surface, header_font, machine.machine_id, (card_x + 10, card_y + 10), colors["text"], text_width)
            _draw_text_fit(
                surface,
                small_font,
                (
                    f"Done W/O {completion_summary.machine_completed_work_orders[machine.machine_id]}/"
                    f"{workload_summary.machine_work_order_counts[machine.machine_id]}"
                ),
                (card_x + 10, card_y + 32),
                colors["text"] if active_operations else colors["muted"],
                text_width,
            )
            _draw_text_fit(
                surface,
                small_font,
                (
                    f"Batch {completion_summary.machine_completed_batches[machine.machine_id]}/"
                    f"{workload_summary.machine_batch_counts[machine.machine_id]}"
                ),
                (card_x + 10, card_y + 50),
                colors["text"] if active_operations else colors["muted"],
                text_width,
            )

            if active_operations:
                first = active_operations[0]
                progress = first.progress_ratio(current_time)
                progress_rect = pygame.Rect(card_x + 10, card_y + card_height - 17, card_rect.width - 20, 8)
                pygame.draw.rect(surface, colors["progress_bg"], progress_rect, border_radius=4)
                pygame.draw.rect(
                    surface,
                    colors["progress"],
                    pygame.Rect(progress_rect.left, progress_rect.top, int(progress_rect.width * progress), 8),
                    border_radius=4,
                )
                _draw_text_fit(
                    surface,
                    small_font,
                    f"Active W/O {len(active_operations)}/{first.batch_wo_count}",
                    (card_x + 10, card_y + 68),
                    colors["text"],
                    text_width,
                )
                _draw_text_fit(
                    surface,
                    small_font,
                    f"Len {first.batch_length_sum:,.0f}/55k",
                    (card_x + 10, card_y + 86),
                    colors["text"],
                    text_width,
                )
            else:
                _draw_text_fit(surface, body_font, "Available", (card_x + 10, card_y + 72), colors["muted"], text_width)
            surface.set_clip(previous_clip)

    side_left = floor_left + floor_width + 22
    side_width = width - side_left - 24
    if side_width >= 190:
        side_rect = pygame.Rect(side_left, floor_top, side_width, floor_height)
        pygame.draw.rect(surface, colors["panel_alt"], side_rect, border_radius=12)
        pygame.draw.rect(surface, colors["line"], side_rect, width=2, border_radius=12)
        previous_clip = surface.get_clip()
        surface.set_clip(side_rect)
        x = side_rect.left + 16
        y = side_rect.top + 16
        inner_width = side_rect.width - 32
        _draw_text_fit(surface, header_font, "Run Summary", (x, y), colors["text"], inner_width)
        y += 32
        metric_lines = build_run_summary_lines(data, workload_summary)
        for line in metric_lines:
            if y + _font_height(body_font) > side_rect.bottom - 260:
                break
            _draw_text_fit(surface, body_font, line, (x, y), colors["text"], inner_width)
            y += 22

        remaining_top = y + 10
        remaining_height = max(side_rect.bottom - remaining_top - 14, 0)
        section_gap = 8
        section_height = max(int((remaining_height - section_gap * 3) / 4), 0)
        sections = [
            ("Bay work orders", workload_summary.bay_work_order_counts),
            ("Bay batch count", workload_summary.bay_batch_counts),
            ("Machine W/O count", workload_summary.machine_work_order_counts),
            ("Machine batch count", workload_summary.machine_batch_counts),
        ]
        for section_index, (section_title, values) in enumerate(sections):
            section_top = remaining_top + section_index * (section_height + section_gap)
            section_rect = pygame.Rect(x, section_top, inner_width, section_height)
            _draw_metric_grid(pygame, surface, header_font, small_font, section_title, values, section_rect, colors)
        surface.set_clip(previous_clip)

    timeline_rect = pygame.Rect(32, height - 58, width - 64, 18)
    pygame.draw.rect(surface, colors["progress_bg"], timeline_rect, border_radius=8)
    ratio = 0.0
    if data.end_time_min > data.start_time_min:
        ratio = (current_time - data.start_time_min) / (data.end_time_min - data.start_time_min)
    ratio = min(max(ratio, 0.0), 1.0)
    pygame.draw.rect(
        surface,
        colors["accent"],
        pygame.Rect(timeline_rect.left, timeline_rect.top, int(timeline_rect.width * ratio), timeline_rect.height),
        border_radius=8,
    )
    _draw_text_fit(surface, small_font, f"{data.start_time_min:.1f} min", (timeline_rect.left, timeline_rect.bottom + 7), colors["muted"], 120)
    end_text = f"{data.end_time_min:.1f} min"
    _draw_text_fit(
        surface,
        small_font,
        end_text,
        (timeline_rect.right - min(120, timeline_rect.width), timeline_rect.bottom + 7),
        colors["muted"],
        min(120, timeline_rect.width),
    )


def run_pygame_viewer(
    data: ViewerData,
    width: int = 1280,
    height: int = 760,
    speed: float = 30.0,
    start_paused: bool = False,
    screenshot_path: str | Path | None = None,
    max_frames: int | None = None,
) -> None:
    """Pygame 창을 열고 공장 playback을 재생한다.

    조작:
      Space  : 재생/일시정지
      Left   : 5분 뒤로
      Right  : 5분 앞으로
      Down   : 재생 속도 절반
      Up     : 재생 속도 2배
      Home   : 처음으로 이동
      End    : 끝으로 이동
      Mouse  : 하단 timeline 클릭으로 시각 이동
      Esc    : 종료

    검증용:
      screenshot_path : 현재 화면을 PNG로 저장한다.
      max_frames      : 지정한 frame 수만 렌더링하고 자동 종료한다.
    """

    try:
        import pygame
    except ImportError as exc:
        print("[ERROR][pygame_factory_viewer.run_pygame_viewer] cause=pygame_not_installed")
        raise RuntimeError("pygame is required for pygame-viewer") from exc

    try:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PMSP Cutting Factory Playback Viewer")
    except Exception as exc:
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_viewer] "
            f"cause=pygame_display_init_failed width={width} height={height} message={exc}"
        )
        raise RuntimeError("failed to initialize pygame display") from exc

    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("consolas", 24, bold=True),
        "header": pygame.font.SysFont("consolas", 18, bold=True),
        "body": pygame.font.SysFont("consolas", 15),
        "small": pygame.font.SysFont("consolas", 12),
    }

    colors = {
        "bg": (18, 22, 28),
        "panel": (33, 40, 50),
        "panel_alt": (42, 51, 64),
        "line": (82, 96, 115),
        "text": (232, 238, 245),
        "muted": (159, 170, 184),
        "active": (61, 132, 104),
        "idle": (57, 65, 78),
        "accent": (237, 174, 73),
        "danger": (221, 93, 93),
        "progress_bg": (54, 62, 74),
        "progress": (109, 194, 132),
    }

    current_time = data.start_time_min
    playback_speed = float(speed)
    paused = start_paused
    running = True
    rendered_frames = 0
    screenshot_file = Path(screenshot_path) if screenshot_path is not None else None
    workload_summary = build_workload_summary(data)

    def clamp_time(value: float) -> float:
        """viewer 시간이 입력 schedule 범위를 벗어나지 않게 자른다."""

        return min(max(value, data.start_time_min), data.end_time_min)

    while running:
        elapsed_seconds = clock.tick(30) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_LEFT:
                    current_time = clamp_time(current_time - 5.0)
                elif event.key == pygame.K_RIGHT:
                    current_time = clamp_time(current_time + 5.0)
                elif event.key == pygame.K_DOWN:
                    playback_speed = max(1.0, playback_speed / 2.0)
                elif event.key == pygame.K_UP:
                    playback_speed = min(1440.0, playback_speed * 2.0)
                elif event.key == pygame.K_HOME:
                    current_time = data.start_time_min
                elif event.key == pygame.K_END:
                    current_time = data.end_time_min
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                timeline_rect = pygame.Rect(40, height - 60, width - 80, 18)
                if timeline_rect.collidepoint(x, y):
                    ratio = (x - timeline_rect.left) / timeline_rect.width
                    current_time = clamp_time(data.start_time_min + ratio * (data.end_time_min - data.start_time_min))

        if not paused:
            current_time = clamp_time(current_time + elapsed_seconds * playback_speed)
            if current_time >= data.end_time_min:
                paused = True

        _draw_factory_frame(
            pygame=pygame,
            surface=screen,
            data=data,
            current_time=current_time,
            playback_speed=playback_speed,
            paused=paused,
            fonts=fonts,
            colors=colors,
            title="PMSP Cutting Factory Playback",
            workload=workload_summary,
        )

        pygame.display.flip()
        rendered_frames += 1

        if screenshot_file is not None and rendered_frames == 1:
            try:
                screenshot_file.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, str(screenshot_file))
                print(f"[CHECK][pygame_factory_viewer.run_pygame_viewer] screenshot={screenshot_file}")
            except Exception as exc:
                print(
                    "[ERROR][pygame_factory_viewer.run_pygame_viewer] "
                    f"cause=screenshot_save_failed path={screenshot_file} message={exc}"
                )
                raise RuntimeError(f"failed to save pygame screenshot: {screenshot_file}") from exc

        if max_frames is not None and rendered_frames >= max_frames:
            running = False

    pygame.quit()


def run_pygame_comparison(
    left_data: ViewerData,
    right_data: ViewerData,
    left_label: str = "ACTUAL REPLAY",
    right_label: str = "GENERATED PLAN",
    width: int = 1920,
    height: int = 820,
    speed: float = 30.0,
    start_paused: bool = False,
    screenshot_path: str | Path | None = None,
    max_frames: int | None = None,
) -> None:
    """actual/generated 같은 두 run을 한 Pygame 창에서 좌우 동시 재생한다."""

    try:
        import pygame
    except ImportError as exc:
        print("[ERROR][pygame_factory_viewer.run_pygame_comparison] cause=pygame_not_installed")
        raise RuntimeError("pygame is required for pygame-compare") from exc

    try:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PMSP Cutting Factory Comparison Viewer")
    except Exception as exc:
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_comparison] "
            f"cause=pygame_display_init_failed width={width} height={height} message={exc}"
        )
        raise RuntimeError("failed to initialize pygame comparison display") from exc

    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("consolas", 22, bold=True),
        "header": pygame.font.SysFont("consolas", 16, bold=True),
        "body": pygame.font.SysFont("consolas", 13),
        "small": pygame.font.SysFont("consolas", 11),
    }
    colors = {
        "bg": (18, 22, 28),
        "panel": (33, 40, 50),
        "panel_alt": (42, 51, 64),
        "line": (82, 96, 115),
        "text": (232, 238, 245),
        "muted": (159, 170, 184),
        "active": (61, 132, 104),
        "idle": (57, 65, 78),
        "accent": (237, 174, 73),
        "danger": (221, 93, 93),
        "progress_bg": (54, 62, 74),
        "progress": (109, 194, 132),
    }

    left_duration = max(left_data.end_time_min - left_data.start_time_min, 1.0)
    right_duration = max(right_data.end_time_min - right_data.start_time_min, 1.0)
    reference_duration = max(left_duration, right_duration)
    progress_ratio = 0.0
    playback_speed = float(speed)
    paused = start_paused
    running = True
    rendered_frames = 0
    screenshot_file = Path(screenshot_path) if screenshot_path is not None else None
    left_workload = build_workload_summary(left_data)
    right_workload = build_workload_summary(right_data)

    def current_time_for(data: ViewerData) -> float:
        return data.start_time_min + progress_ratio * max(data.end_time_min - data.start_time_min, 0.0)

    while running:
        elapsed_seconds = clock.tick(30) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_LEFT:
                    progress_ratio = max(0.0, progress_ratio - 5.0 / reference_duration)
                elif event.key == pygame.K_RIGHT:
                    progress_ratio = min(1.0, progress_ratio + 5.0 / reference_duration)
                elif event.key == pygame.K_DOWN:
                    playback_speed = max(1.0, playback_speed / 2.0)
                elif event.key == pygame.K_UP:
                    playback_speed = min(1440.0, playback_speed * 2.0)
                elif event.key == pygame.K_HOME:
                    progress_ratio = 0.0
                elif event.key == pygame.K_END:
                    progress_ratio = 1.0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                timeline_rect = pygame.Rect(32, height - 58, width - 64, 18)
                if timeline_rect.collidepoint(x, y):
                    progress_ratio = min(max((x - timeline_rect.left) / timeline_rect.width, 0.0), 1.0)

        if not paused:
            progress_ratio = min(1.0, progress_ratio + (elapsed_seconds * playback_speed / reference_duration))
            if progress_ratio >= 1.0:
                paused = True

        split_x = width // 2
        left_surface = screen.subsurface(pygame.Rect(0, 0, split_x, height))
        right_surface = screen.subsurface(pygame.Rect(split_x, 0, width - split_x, height))
        _draw_factory_frame(
            pygame=pygame,
            surface=left_surface,
            data=left_data,
            current_time=current_time_for(left_data),
            playback_speed=playback_speed,
            paused=paused,
            fonts=fonts,
            colors=colors,
            title=left_label,
            workload=left_workload,
        )
        _draw_factory_frame(
            pygame=pygame,
            surface=right_surface,
            data=right_data,
            current_time=current_time_for(right_data),
            playback_speed=playback_speed,
            paused=paused,
            fonts=fonts,
            colors=colors,
            title=right_label,
            workload=right_workload,
        )
        pygame.draw.line(screen, colors["accent"], (split_x, 0), (split_x, height), width=3)

        pygame.display.flip()
        rendered_frames += 1

        if screenshot_file is not None and rendered_frames == 1:
            try:
                screenshot_file.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, str(screenshot_file))
                print(f"[CHECK][pygame_factory_viewer.run_pygame_comparison] screenshot={screenshot_file}")
            except Exception as exc:
                print(
                    "[ERROR][pygame_factory_viewer.run_pygame_comparison] "
                    f"cause=screenshot_save_failed path={screenshot_file} message={exc}"
                )
                raise RuntimeError(f"failed to save pygame comparison screenshot: {screenshot_file}") from exc

        if max_frames is not None and rendered_frames >= max_frames:
            running = False

    pygame.quit()


def run_pygame_comparison_gif(
    left_data: ViewerData,
    right_data: ViewerData,
    gif_path: str | Path,
    left_label: str = "ACTUAL REPLAY",
    right_label: str = "GENERATED PLAN",
    width: int = 1920,
    height: int = 820,
    speed: float = 30.0,
    frame_count: int = 48,
    frame_duration_ms: int = 120,
) -> None:
    """actual/generated 비교 playback을 GUI 창 없이 GIF 파일로 저장한다.

    GIF는 두 run의 전체 진행률을 0~100%로 정규화해서 같은 frame index에 그린다.
    actual replay와 generated plan의 makespan이 다르므로, 같은 frame은 같은 절대분이
    아니라 "각 run에서 같은 진행률"을 의미한다.
    """

    if frame_count < 2:
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_comparison_gif] "
            f"cause=invalid_frame_count frame_count={frame_count}"
        )
        raise ValueError("--gif-frames must be at least 2")
    if frame_duration_ms <= 0:
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_comparison_gif] "
            f"cause=invalid_frame_duration_ms frame_duration_ms={frame_duration_ms}"
        )
        raise ValueError("--gif-duration-ms must be a positive integer")

    try:
        import pygame
    except ImportError as exc:
        print("[ERROR][pygame_factory_viewer.run_pygame_comparison_gif] cause=pygame_not_installed")
        raise RuntimeError("pygame is required for pygame GIF export") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        print("[ERROR][pygame_factory_viewer.run_pygame_comparison_gif] cause=pillow_not_installed")
        raise RuntimeError("Pillow is required for pygame GIF export") from exc

    gif_file = Path(gif_path)
    pygame.font.init()
    try:
        surface = pygame.Surface((width, height))
        fonts = {
            "title": pygame.font.SysFont("consolas", 22, bold=True),
            "header": pygame.font.SysFont("consolas", 16, bold=True),
            "body": pygame.font.SysFont("consolas", 13),
            "small": pygame.font.SysFont("consolas", 11),
        }
        colors = {
            "bg": (18, 22, 28),
            "panel": (33, 40, 50),
            "panel_alt": (42, 51, 64),
            "line": (82, 96, 115),
            "text": (232, 238, 245),
            "muted": (159, 170, 184),
            "active": (61, 132, 104),
            "idle": (57, 65, 78),
            "accent": (237, 174, 73),
            "danger": (221, 93, 93),
            "progress_bg": (54, 62, 74),
            "progress": (109, 194, 132),
        }
        left_workload = build_workload_summary(left_data)
        right_workload = build_workload_summary(right_data)
        split_x = width // 2
        frames = []
        for frame_index in range(frame_count):
            progress_ratio = frame_index / (frame_count - 1)
            left_time = left_data.start_time_min + progress_ratio * (
                left_data.end_time_min - left_data.start_time_min
            )
            right_time = right_data.start_time_min + progress_ratio * (
                right_data.end_time_min - right_data.start_time_min
            )
            left_surface = surface.subsurface(pygame.Rect(0, 0, split_x, height))
            right_surface = surface.subsurface(pygame.Rect(split_x, 0, width - split_x, height))
            _draw_factory_frame(
                pygame=pygame,
                surface=left_surface,
                data=left_data,
                current_time=left_time,
                playback_speed=float(speed),
                paused=False,
                fonts=fonts,
                colors=colors,
                title=left_label,
                workload=left_workload,
            )
            _draw_factory_frame(
                pygame=pygame,
                surface=right_surface,
                data=right_data,
                current_time=right_time,
                playback_speed=float(speed),
                paused=False,
                fonts=fonts,
                colors=colors,
                title=right_label,
                workload=right_workload,
            )
            pygame.draw.line(surface, colors["accent"], (split_x, 0), (split_x, height), width=3)
            raw_bytes = pygame.image.tobytes(surface, "RGB")
            frames.append(Image.frombytes("RGB", (width, height), raw_bytes))

        try:
            gif_file.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                gif_file,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
        except Exception as exc:
            print(
                "[ERROR][pygame_factory_viewer.run_pygame_comparison_gif] "
                f"cause=gif_save_failed path={gif_file} message={exc}"
            )
            raise RuntimeError(f"failed to save pygame comparison GIF: {gif_file}") from exc
    except Exception as exc:
        if isinstance(exc, (RuntimeError, ValueError)):
            raise
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_comparison_gif] "
            f"cause=gif_render_failed path={gif_file} message={exc}"
        )
        raise RuntimeError(f"failed to render pygame comparison GIF: {gif_file}") from exc
    finally:
        pygame.quit()

    print(
        "[CHECK][pygame_factory_viewer.run_pygame_comparison_gif] "
        f"gif={gif_file} frames={frame_count} duration_ms={frame_duration_ms} "
        f"width={width} height={height}"
    )


def run_pygame_viewer_from_paths(
    event_log_path: str | Path,
    layout_path: str | Path,
    schedule_path: str | Path,
    metrics_path: str | Path,
    width: int = 1280,
    height: int = 760,
    speed: float = 30.0,
    start_paused: bool = False,
    dry_run: bool = False,
    screenshot_path: str | Path | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """CLI에서 호출하는 entrypoint. dry-run이면 GUI 없이 입력 검증만 수행한다."""

    if max_frames is not None and max_frames <= 0:
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_viewer_from_paths] "
            f"cause=invalid_max_frames max_frames={max_frames}"
        )
        raise ValueError("--max-frames must be a positive integer")
    if dry_run and (screenshot_path is not None or max_frames is not None):
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_viewer_from_paths] "
            "cause=conflicting_options dry_run_cannot_render=true"
        )
        raise ValueError("--dry-run cannot be combined with --screenshot or --max-frames")

    print("[pygame viewer]")
    print(f"- event_log: {event_log_path}")
    print(f"- layout: {layout_path}")
    print(f"- schedule: {schedule_path}")
    print(f"- metrics: {metrics_path}")
    data = load_viewer_data(
        event_log_path=event_log_path,
        layout_path=layout_path,
        schedule_path=schedule_path,
        metrics_path=metrics_path,
    )
    summary = validate_viewer_data(data)
    workload_summary = build_workload_summary(data)
    summary["workload"] = {
        "machine_workload_imbalance_minutes": workload_summary.machine_workload_imbalance_minutes,
        "machine_work_order_imbalance": workload_summary.machine_work_order_imbalance,
        "machine_batch_imbalance": workload_summary.machine_batch_imbalance,
        "bay_workload_imbalance_minutes": workload_summary.bay_workload_imbalance_minutes,
        "bay_work_order_imbalance": workload_summary.bay_work_order_imbalance,
        "bay_batch_imbalance": workload_summary.bay_batch_imbalance,
    }
    if dry_run:
        print("[VALIDATION][pygame_factory_viewer.run_pygame_viewer_from_paths] passed=true dry_run=true")
        return summary

    print("[CHECK][pygame_factory_viewer.run_pygame_viewer_from_paths] dry_run=false launching_pygame=true")
    run_pygame_viewer(
        data=data,
        width=width,
        height=height,
        speed=speed,
        start_paused=start_paused,
        screenshot_path=screenshot_path,
        max_frames=max_frames,
    )
    return summary


def run_pygame_comparison_from_paths(
    left_event_log_path: str | Path,
    left_layout_path: str | Path,
    left_schedule_path: str | Path,
    left_metrics_path: str | Path,
    right_event_log_path: str | Path,
    right_layout_path: str | Path,
    right_schedule_path: str | Path,
    right_metrics_path: str | Path,
    left_label: str = "ACTUAL REPLAY",
    right_label: str = "GENERATED PLAN",
    width: int = 1920,
    height: int = 820,
    speed: float = 30.0,
    start_paused: bool = False,
    dry_run: bool = False,
    screenshot_path: str | Path | None = None,
    max_frames: int | None = None,
    gif_path: str | Path | None = None,
    gif_frames: int = 48,
    gif_duration_ms: int = 120,
) -> dict[str, Any]:
    """CLI에서 호출하는 actual/generated 동시 Pygame 비교 entrypoint."""

    if max_frames is not None and max_frames <= 0:
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_comparison_from_paths] "
            f"cause=invalid_max_frames max_frames={max_frames}"
        )
        raise ValueError("--max-frames must be a positive integer")
    if gif_frames < 2:
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_comparison_from_paths] "
            f"cause=invalid_gif_frames gif_frames={gif_frames}"
        )
        raise ValueError("--gif-frames must be at least 2")
    if gif_duration_ms <= 0:
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_comparison_from_paths] "
            f"cause=invalid_gif_duration_ms gif_duration_ms={gif_duration_ms}"
        )
        raise ValueError("--gif-duration-ms must be a positive integer")
    if dry_run and (screenshot_path is not None or max_frames is not None or gif_path is not None):
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_comparison_from_paths] "
            "cause=conflicting_options dry_run_cannot_render=true"
        )
        raise ValueError("--dry-run cannot be combined with --screenshot, --max-frames, or --gif")
    if gif_path is not None and (screenshot_path is not None or max_frames is not None):
        print(
            "[ERROR][pygame_factory_viewer.run_pygame_comparison_from_paths] "
            "cause=conflicting_options gif_is_standalone_render=true"
        )
        raise ValueError("--gif cannot be combined with --screenshot or --max-frames")

    print("[pygame compare]")
    print(f"- left_event_log: {left_event_log_path}")
    print(f"- left_layout: {left_layout_path}")
    print(f"- left_schedule: {left_schedule_path}")
    print(f"- left_metrics: {left_metrics_path}")
    print(f"- right_event_log: {right_event_log_path}")
    print(f"- right_layout: {right_layout_path}")
    print(f"- right_schedule: {right_schedule_path}")
    print(f"- right_metrics: {right_metrics_path}")

    left_data = load_viewer_data(
        event_log_path=left_event_log_path,
        layout_path=left_layout_path,
        schedule_path=left_schedule_path,
        metrics_path=left_metrics_path,
    )
    right_data = load_viewer_data(
        event_log_path=right_event_log_path,
        layout_path=right_layout_path,
        schedule_path=right_schedule_path,
        metrics_path=right_metrics_path,
    )
    left_summary = validate_viewer_data(left_data)
    right_summary = validate_viewer_data(right_data)
    left_workload = build_workload_summary(left_data)
    right_workload = build_workload_summary(right_data)
    left_summary["workload"] = {
        "machine_workload_imbalance_minutes": left_workload.machine_workload_imbalance_minutes,
        "machine_work_order_imbalance": left_workload.machine_work_order_imbalance,
        "machine_batch_imbalance": left_workload.machine_batch_imbalance,
        "bay_workload_imbalance_minutes": left_workload.bay_workload_imbalance_minutes,
        "bay_work_order_imbalance": left_workload.bay_work_order_imbalance,
        "bay_batch_imbalance": left_workload.bay_batch_imbalance,
    }
    right_summary["workload"] = {
        "machine_workload_imbalance_minutes": right_workload.machine_workload_imbalance_minutes,
        "machine_work_order_imbalance": right_workload.machine_work_order_imbalance,
        "machine_batch_imbalance": right_workload.machine_batch_imbalance,
        "bay_workload_imbalance_minutes": right_workload.bay_workload_imbalance_minutes,
        "bay_work_order_imbalance": right_workload.bay_work_order_imbalance,
        "bay_batch_imbalance": right_workload.bay_batch_imbalance,
    }
    print(
        "[CHECK][pygame_factory_viewer.run_pygame_comparison_from_paths] "
        "comparison_metric=leveling_score_max_minus_min lower_is_better=true "
        f"left_bay_wo_leveling={left_workload.bay_work_order_imbalance} "
        f"right_bay_wo_leveling={right_workload.bay_work_order_imbalance} "
        f"left_machine_wo_leveling={left_workload.machine_work_order_imbalance} "
        f"right_machine_wo_leveling={right_workload.machine_work_order_imbalance} "
        f"left_bay_batch_leveling={left_workload.bay_batch_imbalance} "
        f"right_bay_batch_leveling={right_workload.bay_batch_imbalance} "
        f"left_machine_batch_leveling={left_workload.machine_batch_imbalance} "
        f"right_machine_batch_leveling={right_workload.machine_batch_imbalance}"
    )
    summary = {"left": left_summary, "right": right_summary}
    if dry_run:
        print("[VALIDATION][pygame_factory_viewer.run_pygame_comparison_from_paths] passed=true dry_run=true")
        return summary

    if gif_path is not None:
        print("[CHECK][pygame_factory_viewer.run_pygame_comparison_from_paths] rendering_gif=true")
        run_pygame_comparison_gif(
            left_data=left_data,
            right_data=right_data,
            gif_path=gif_path,
            left_label=left_label,
            right_label=right_label,
            width=width,
            height=height,
            speed=speed,
            frame_count=gif_frames,
            frame_duration_ms=gif_duration_ms,
        )
        summary["gif_path"] = str(Path(gif_path))
        return summary

    print("[CHECK][pygame_factory_viewer.run_pygame_comparison_from_paths] dry_run=false launching_pygame=true")
    run_pygame_comparison(
        left_data=left_data,
        right_data=right_data,
        left_label=left_label,
        right_label=right_label,
        width=width,
        height=height,
        speed=speed,
        start_paused=start_paused,
        screenshot_path=screenshot_path,
        max_frames=max_frames,
    )
    return summary
