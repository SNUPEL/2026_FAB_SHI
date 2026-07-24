"""두 Phase 2 full-flow 스케줄(휴리스틱 vs 제안)을 비교 시각화한다.

이 스크립트는 **시각화 전용**이다. full-flow 실행은 기존 CLI
``python main.py phase2-run-full-workflow ...``로 미리 수행하고, 이 스크립트는 그
결과 output 디렉터리 두 개를 읽어 Gantt 차트·batch 구성 표·비교 요약만 생성한다.
파이프라인 코드(main.py / Phase2 / Environment)는 전혀 건드리지 않는다.

핵심 계약(코드 확인 기준):
- 한 batch는 병렬 실행이라 ``batch_duration = max(멤버 processing_time)``이다
  (Environment/hierarchical.py:604-637, ``duration_rule="max_processing_time"``).
  ``job_processing_time_sum``은 합이며 clock에 반영되지 않는다.
- ``machine_timeline.csv`` / ``batch_list.csv``에는 ``bay_id``가 없으므로
  ``wo_machine_assignment.csv``의 ``machine_id -> bay_id``로 조인한다.
- family는 ``block_set_id = PROJ_NO::GYEL::BLK_NO``의 가운데 토큰이다
  (Utils/data/multi_series_cutting_data.py:188-202).
- score_tuple 순서: hard_violation -> makespan -> CUT gap -> WO gap -> BV gap
  -> occupancy gap (Environment/metrics.py:11-18).

순위(rank)는 "같은 배치를 형성할 수 있는 동종 계열" = 같은 Bay·같은 family 안에서
processing_time 내림차순 등수다(기본 ``--rank-scope bay_family``). 두 방법이 동일
seed·동일 Phase 1 upstream을 쓰면 작업 집합·작업시간·순위가 동일하고 batch 구성과
timeline만 달라진다.

사용 예:
    python scripts/compare_full_flow_schedules.py \
        --method-a-dir output/_smoke_min --label-a min_makespan \
        --method-b-dir output/_smoke_lpt --label-b lpt_batch \
        --output-dir output/analysis/schedule_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 15대 planning 설비의 확정 Bay 매핑(소스 오브 트루스). idle 설비까지 Gantt 축에
# 그리기 위해 사용하며, import 실패 시 assignment에서 관측된 매핑으로 대체한다.
try:  # pragma: no cover - 단순 import 가드
    from Utils.data.multi_series_cutting_data import MIXED_PLANNING_MACHINE_IDS_BY_BAY
except Exception:  # noqa: BLE001 - 시각화 도구는 파이프라인 없이도 동작해야 한다
    MIXED_PLANNING_MACHINE_IDS_BY_BAY = None

BAY_ORDER: Tuple[str, ...] = ("22", "23", "24", "25", "trans")
FAMILY_ORDER: Tuple[str, ...] = ("NP", "NC", "FN", "FL")
FAMILY_COLORS: Dict[str, str] = {
    "NP": "#1f77b4",
    "NC": "#4c9be8",
    "FN": "#d62728",
    "FL": "#ff9d4d",
}
UNKNOWN_FAMILY_COLOR = "#7f7f7f"
BAY_BAND_COLORS = ("#f4f6fa", "#e9edf5")
SCORE_LABELS: Tuple[str, ...] = (
    "hard_violation",
    "makespan",
    "cut_gap",
    "wo_gap",
    "bevel_gap",
    "occupancy_gap",
)
MAX_BATCH_WO_COUNT = 3
FLOAT_TOLERANCE = 1e-3


# --------------------------------------------------------------------------- #
# 데이터 모델
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JobRow:
    job_id: str
    block_set_id: str
    family: str
    bay_id: str
    machine_id: str
    processing_time: float
    rank: int = 0
    group_size: int = 0


@dataclass
class Batch:
    batch_id: str
    machine_id: str
    bay_id: str
    start_time: float
    finish_time: float
    batch_duration: float
    job_processing_time_sum: float
    member_ids: List[str] = field(default_factory=list)


@dataclass
class MethodData:
    label: str
    source_dir: Path
    jobs: Dict[str, JobRow]
    batches: List[Batch]
    machines_by_bay: Dict[str, List[str]]
    score_tuple: List[float]
    score_mode: str
    makespan: float


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _fail(location: str, cause: str, **fields: object) -> "RuntimeError":
    detail = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[ERROR][compare_full_flow_schedules.{location}] cause={cause} {detail}".rstrip())
    return RuntimeError(f"{location}: {cause}")


def _read_csv(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    if not path.is_file():
        raise _fail("_read_csv", "missing_file", path=path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames or []
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise _fail("_read_csv", "missing_columns", path=path, missing=missing, header=header)
    if not rows:
        raise _fail("_read_csv", "empty_file", path=path)
    return rows


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise _fail("_read_json", "missing_file", path=path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise _fail("_read_json", "not_object", path=path, type=type(payload).__name__)
    return payload


def _to_float(value: object, field_name: str, row_key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _fail("_to_float", "not_numeric", field=field_name, row_key=row_key, value=value) from exc
    return number


def _family_from_block_set_id(block_set_id: str, job_id: str) -> str:
    parts = block_set_id.split("::")
    if len(parts) != 3 or not parts[1].strip():
        raise _fail("_family_from_block_set_id", "unexpected_block_set_id", job_id=job_id, block_set_id=block_set_id)
    return parts[1].strip().upper()


# --------------------------------------------------------------------------- #
# 로딩 + 순위 + 검증
# --------------------------------------------------------------------------- #
def _rank_group_key(job: JobRow, rank_scope: str) -> Tuple[str, ...]:
    if rank_scope == "bay_family":
        return (job.bay_id, job.family)
    if rank_scope == "family":
        return (job.family,)
    if rank_scope == "machine":
        return (job.machine_id,)
    raise _fail("_rank_group_key", "unknown_rank_scope", rank_scope=rank_scope)


def _assign_ranks(jobs: Mapping[str, JobRow], rank_scope: str) -> Dict[str, JobRow]:
    groups: Dict[Tuple[str, ...], List[JobRow]] = {}
    for job in jobs.values():
        groups.setdefault(_rank_group_key(job, rank_scope), []).append(job)
    ranked: Dict[str, JobRow] = {}
    for group_jobs in groups.values():
        # 내림차순(긴 작업이 1등). 동점은 job_id로 안정 정렬.
        ordered = sorted(group_jobs, key=lambda item: (-item.processing_time, item.job_id))
        size = len(ordered)
        for index, job in enumerate(ordered, start=1):
            ranked[job.job_id] = JobRow(
                job_id=job.job_id,
                block_set_id=job.block_set_id,
                family=job.family,
                bay_id=job.bay_id,
                machine_id=job.machine_id,
                processing_time=job.processing_time,
                rank=index,
                group_size=size,
            )
    return ranked


def _machines_by_bay(jobs: Mapping[str, JobRow]) -> Dict[str, List[str]]:
    if MIXED_PLANNING_MACHINE_IDS_BY_BAY is not None:
        return {bay_id: list(machine_ids) for bay_id, machine_ids in MIXED_PLANNING_MACHINE_IDS_BY_BAY.items()}
    observed: Dict[str, set] = {}
    for job in jobs.values():
        observed.setdefault(job.bay_id, set()).add(job.machine_id)
    return {bay_id: sorted(machine_ids) for bay_id, machine_ids in observed.items()}


def load_method(source_dir: Path, label: str, rank_scope: str) -> MethodData:
    assignment_rows = _read_csv(
        source_dir / "wo_machine_assignment.csv",
        ("job_id", "block_set_id", "machine_id", "bay_id", "processing_time"),
    )
    jobs: Dict[str, JobRow] = {}
    for row in assignment_rows:
        job_id = str(row["job_id"]).strip()
        block_set_id = str(row["block_set_id"]).strip()
        if job_id in jobs:
            raise _fail("load_method", "duplicate_job", label=label, job_id=job_id)
        jobs[job_id] = JobRow(
            job_id=job_id,
            block_set_id=block_set_id,
            family=_family_from_block_set_id(block_set_id, job_id),
            bay_id=str(row["bay_id"]).strip(),
            machine_id=str(row["machine_id"]).strip(),
            processing_time=_to_float(row["processing_time"], "processing_time", job_id),
        )
    jobs = _assign_ranks(jobs, rank_scope)

    machine_to_bay = {job.machine_id: job.bay_id for job in jobs.values()}
    timeline_rows = _read_csv(
        source_dir / "machine_timeline.csv",
        ("batch_id", "machine_id", "job_ids", "start_time", "finish_time", "batch_duration", "job_processing_time_sum"),
    )
    batches: List[Batch] = []
    for row in timeline_rows:
        batch_id = str(row["batch_id"]).strip()
        machine_id = str(row["machine_id"]).strip()
        member_ids = [part.strip() for part in str(row["job_ids"]).split("|") if part.strip()]
        if not member_ids:
            raise _fail("load_method", "empty_batch", label=label, batch_id=batch_id)
        batches.append(
            Batch(
                batch_id=batch_id,
                machine_id=machine_id,
                bay_id=machine_to_bay.get(machine_id, ""),
                start_time=_to_float(row["start_time"], "start_time", batch_id),
                finish_time=_to_float(row["finish_time"], "finish_time", batch_id),
                batch_duration=_to_float(row["batch_duration"], "batch_duration", batch_id),
                job_processing_time_sum=_to_float(row["job_processing_time_sum"], "job_processing_time_sum", batch_id),
                member_ids=member_ids,
            )
        )

    report = _read_json(source_dir / "phase2_workflow_report.json")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict):
        raise _fail("load_method", "missing_evaluation", label=label, source_dir=source_dir)
    score_tuple = report.get("batch_machine_candidate_score")
    if not isinstance(score_tuple, list) or len(score_tuple) != len(SCORE_LABELS):
        raise _fail("load_method", "invalid_score_tuple", label=label, value=score_tuple)
    score_mode = str(evaluation.get("score_mode") or "raw")
    sequence_score = evaluation.get("sequence_score")
    if not isinstance(sequence_score, dict) or "makespan" not in sequence_score:
        raise _fail("load_method", "missing_makespan", label=label)
    makespan = _to_float(sequence_score["makespan"], "makespan", label)

    method = MethodData(
        label=label,
        source_dir=source_dir,
        jobs=jobs,
        batches=batches,
        machines_by_bay=_machines_by_bay(jobs),
        score_tuple=[float(value) for value in score_tuple],
        score_mode=score_mode,
        makespan=makespan,
    )
    _validate_method(method)
    return method


def _validate_method(method: MethodData) -> None:
    label = method.label
    # 1) 모든 assignment 작업이 정확히 한 batch에만 등장한다.
    seen: Dict[str, str] = {}
    for batch in method.batches:
        for job_id in batch.member_ids:
            if job_id not in method.jobs:
                raise _fail("_validate_method", "batch_job_not_in_assignment", label=label, job_id=job_id)
            if job_id in seen:
                raise _fail("_validate_method", "job_in_multiple_batches", label=label, job_id=job_id, batches=[seen[job_id], batch.batch_id])
            seen[job_id] = batch.batch_id
    missing = sorted(set(method.jobs) - set(seen))
    if missing:
        raise _fail("_validate_method", "unscheduled_jobs", label=label, count=len(missing), sample=missing[:3])
    # 2) batch_duration == max(member processing_time).
    for batch in method.batches:
        longest = max(method.jobs[job_id].processing_time for job_id in batch.member_ids)
        if abs(longest - batch.batch_duration) > FLOAT_TOLERANCE:
            raise _fail("_validate_method", "batch_duration_mismatch", label=label, batch_id=batch.batch_id, longest=longest, batch_duration=batch.batch_duration)
    # 3) makespan == max(finish_time).
    observed_makespan = max(batch.finish_time for batch in method.batches)
    if abs(observed_makespan - method.makespan) > FLOAT_TOLERANCE:
        raise _fail("_validate_method", "makespan_mismatch", label=label, observed=observed_makespan, reported=method.makespan)
    # 4) 각 순위 그룹의 rank가 1..n 유일.
    group_ranks: Dict[Tuple[str, str], List[int]] = {}
    for job in method.jobs.values():
        group_ranks.setdefault((job.bay_id, job.family), []).append(job.rank)
    for (bay_id, family), ranks in group_ranks.items():
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise _fail("_validate_method", "rank_not_contiguous", label=label, bay=bay_id, family=family, ranks=sorted(ranks))
    print(
        f"[VALIDATION][compare_full_flow_schedules._validate_method] passed=true label={label} "
        f"jobs={len(method.jobs)} batches={len(method.batches)} makespan={method.makespan:.3f}"
    )


# --------------------------------------------------------------------------- #
# matplotlib / 한글 폰트 (scripts/plot_multi_series_generator_heatmaps.py 패턴 복제)
# --------------------------------------------------------------------------- #
def _import_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        raise _fail("_import_plt", "matplotlib_import_failed", error=str(exc)) from exc
    return plt


def _resolve_korean_font(font_path: str | None):
    """malgun.ttf(맑은 고딕)를 등록하고 FontProperties를 반환한다. 없으면 None."""

    from matplotlib import font_manager

    candidates = []
    if font_path:
        candidates.append(Path(font_path))
    candidates.extend([Path("C:/Windows/Fonts/malgun.ttf"), Path("/mnt/c/Windows/Fonts/malgun.ttf")])
    for candidate in candidates:
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            prop = font_manager.FontProperties(fname=str(candidate))
            print(f"[CHECK][compare_full_flow_schedules._resolve_korean_font] font={candidate.name}")
            return prop
    print("[CHECK][compare_full_flow_schedules._resolve_korean_font] korean_font_not_found using_default=true")
    return None


def _apply_font(plt, font) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    if font is not None:
        plt.rcParams["font.family"] = font.get_name()


def _family_color(family: str) -> str:
    return FAMILY_COLORS.get(family, UNKNOWN_FAMILY_COLOR)


def _ordered_machines(method: MethodData) -> List[Tuple[str, str]]:
    """(bay_id, machine_id)를 Bay 순서로 정렬해 반환한다."""

    ordered: List[Tuple[str, str]] = []
    remaining_bays = [bay for bay in BAY_ORDER if bay in method.machines_by_bay]
    remaining_bays += [bay for bay in sorted(method.machines_by_bay) if bay not in BAY_ORDER]
    for bay_id in remaining_bays:
        for machine_id in sorted(method.machines_by_bay[bay_id]):
            ordered.append((bay_id, machine_id))
    return ordered


def _short_job_id(job_id: str) -> str:
    if "SYNTH_MULTI_" in job_id:
        return job_id.split("SYNTH_MULTI_")[-1]
    return job_id


def _short_member_label(job_id: str) -> str:
    """`..._WO_000014_NP_0024` -> `WO14-24` 형태의 짧은 식별자."""

    tokens = job_id.split("_")
    if "WO" in tokens:
        wo_index = tokens.index("WO")
        tail = tokens[wo_index + 1 :]
        digits = [tok for tok in tail if tok.isdigit()]
        if len(digits) >= 2:
            return f"WO{int(digits[0])}-{int(digits[-1])}"
    return _short_job_id(job_id)


# --------------------------------------------------------------------------- #
# Gantt
# --------------------------------------------------------------------------- #
def _draw_gantt_axis(ax, method: MethodData, xmax: float, font) -> None:
    machines = _ordered_machines(method)
    row_of_machine = {machine_id: index for index, (_, machine_id) in enumerate(machines)}
    n_rows = len(machines)

    # Bay 배경 밴드
    band_index = 0
    row = 0
    while row < n_rows:
        bay_id = machines[row][0]
        span = sum(1 for _, (bay, _machine) in enumerate(machines) if bay == bay_id)
        ax.axhspan(row - 0.5, row + span - 0.5, color=BAY_BAND_COLORS[band_index % 2], zorder=0)
        ax.text(
            -xmax * 0.012,
            row + span / 2 - 0.5,
            f"Bay {bay_id}",
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#333333",
            fontproperties=font,
        )
        band_index += 1
        row += span

    for batch in method.batches:
        if batch.machine_id not in row_of_machine:
            raise _fail("_draw_gantt_axis", "batch_machine_not_on_axis", label=method.label, machine_id=batch.machine_id)
        y = row_of_machine[batch.machine_id]
        start = batch.start_time
        # batch 외곽선 = batch_duration (envelope)
        ax.broken_barh(
            [(start, batch.batch_duration)],
            (y - 0.45, 0.9),
            facecolors="none",
            edgecolors="#555555",
            linewidth=0.9,
            zorder=3,
        )
        # 멤버 작업: 긴 순서로 위->아래 lane, 각자 processing_time 길이
        members = sorted(batch.member_ids, key=lambda jid: -method.jobs[jid].processing_time)
        lanes = len(members)
        lane_h = 0.9 / lanes
        for lane, job_id in enumerate(members):
            job = method.jobs[job_id]
            yb = y - 0.45 + lane * lane_h
            ax.broken_barh(
                [(start, job.processing_time)],
                (yb + lane_h * 0.08, lane_h * 0.84),
                facecolors=_family_color(job.family),
                edgecolors="white",
                linewidth=0.3,
                zorder=4,
            )
            ax.text(
                start + job.processing_time + xmax * 0.002,
                yb + lane_h * 0.5,
                f"{job.rank}/{job.group_size}",
                ha="left",
                va="center",
                fontsize=5.5,
                color="#222222",
            )

    ax.set_ylim(n_rows - 0.5, -0.5)  # 첫 설비가 위로
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([machine_id for _, machine_id in machines], fontsize=7)
    ax.set_xlim(-xmax * 0.06, xmax * 1.04)
    ax.set_xlabel("time (min)", fontproperties=font)
    ax.grid(True, axis="x", alpha=0.25, zorder=1)
    score = ", ".join(f"{label}={value:g}" for label, value in zip(SCORE_LABELS, method.score_tuple))
    ax.set_title(
        f"{method.label}  |  makespan={method.makespan:.1f}  |  batches={len(method.batches)}\nscore({method.score_mode}): {score}",
        fontsize=9,
        fontproperties=font,
    )


def _family_legend(plt, ax, families: Sequence[str], font) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=_family_color(family), edgecolor="white", label=family) for family in families]
    handles.append(Patch(facecolor="none", edgecolor="#555555", label="batch(=max)"))
    ax.legend(
        handles=handles,
        title="family / 막대=작업시간, 숫자=동종계열 내 등수",
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        fontsize=7,
        title_fontproperties=font,
        framealpha=0.95,
    )


def render_gantt(method: MethodData, out_path: Path, plt, font) -> None:
    n_rows = len(_ordered_machines(method))
    width = min(26.0, max(12.0, method.makespan / 32.0))
    fig, ax = plt.subplots(figsize=(width, max(5.0, n_rows * 0.55)))
    _draw_gantt_axis(ax, method, xmax=max(method.makespan, 1.0), font=font)
    families = [family for family in FAMILY_ORDER if any(job.family == family for job in method.jobs.values())]
    _family_legend(plt, ax, families, font)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHECK][compare_full_flow_schedules.render_gantt] wrote={out_path}")


def render_compare_gantt(method_a: MethodData, method_b: MethodData, out_path: Path, plt, font) -> None:
    xmax = max(method_a.makespan, method_b.makespan, 1.0)
    rows = max(len(_ordered_machines(method_a)), len(_ordered_machines(method_b)))
    width = min(26.0, max(12.0, xmax / 32.0))
    fig, axes = plt.subplots(2, 1, figsize=(width, max(9.0, rows * 0.9)))
    for ax, method in zip(axes, (method_a, method_b)):
        _draw_gantt_axis(ax, method, xmax=xmax, font=font)
    families = sorted(
        {job.family for method in (method_a, method_b) for job in method.jobs.values()},
        key=lambda fam: FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99,
    )
    _family_legend(plt, axes[0], families, font)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHECK][compare_full_flow_schedules.render_compare_gantt] wrote={out_path}")


# --------------------------------------------------------------------------- #
# batch 구성 표
# --------------------------------------------------------------------------- #
def _composition_rows(method: MethodData) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ordered = sorted(method.batches, key=lambda b: (b.bay_id, b.machine_id, b.start_time, b.batch_id))
    for batch in ordered:
        longest = max(method.jobs[jid].processing_time for jid in batch.member_ids)
        members = sorted(batch.member_ids, key=lambda jid: -method.jobs[jid].processing_time)
        for job_id in members:
            job = method.jobs[job_id]
            rows.append(
                {
                    "batch_id": batch.batch_id,
                    "bay_id": batch.bay_id,
                    "machine_id": batch.machine_id,
                    "start_time": round(batch.start_time, 3),
                    "finish_time": round(batch.finish_time, 3),
                    "batch_duration": round(batch.batch_duration, 3),
                    "job_id": job.job_id,
                    "family": job.family,
                    "processing_time": round(job.processing_time, 3),
                    "rank_in_group": job.rank,
                    "group_size": job.group_size,
                    "is_longest_in_batch": int(abs(job.processing_time - longest) <= FLOAT_TOLERANCE),
                }
            )
    return rows


def write_batch_composition(method: MethodData, out_dir: Path, plt, font) -> None:
    rows = _composition_rows(method)
    csv_path = out_dir / f"batch_composition_{method.label}.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CHECK][compare_full_flow_schedules.write_batch_composition] wrote={csv_path} rows={len(rows)}")

    # 배치별 요약 표 이미지 (멤버를 한 셀에 축약)
    ordered = sorted(method.batches, key=lambda b: (b.bay_id, b.machine_id, b.start_time, b.batch_id))
    table_rows: List[List[str]] = []
    for batch in ordered:
        members = sorted(batch.member_ids, key=lambda jid: -method.jobs[jid].processing_time)
        member_lines = []
        for jid in members:
            job = method.jobs[jid]
            marker = "★" if abs(job.processing_time - batch.batch_duration) <= FLOAT_TOLERANCE else " "
            member_lines.append(
                f"{marker}{_short_member_label(jid)} {job.family} #{job.rank}/{job.group_size} ({job.processing_time:.1f})"
            )
        table_rows.append(
            [
                batch.bay_id,
                batch.machine_id,
                batch.batch_id,
                f"{batch.start_time:.1f}",
                f"{batch.finish_time:.1f}",
                f"{batch.batch_duration:.1f}",
                "\n".join(member_lines),
            ]
        )
    header = ["bay", "machine", "batch", "start", "finish", "dur(max)", "members  ★=최장  [WO-seq family #rank/size (proc)]"]
    col_widths = [0.045, 0.075, 0.085, 0.06, 0.06, 0.075, 0.60]
    fig_height = max(4.0, 1.4 + 0.62 * len(table_rows))
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")
    ax.set_title(
        f"batch 구성 — {method.label}  (dur = 멤버 최장 작업시간, #rank = 동종계열(Bay·family) 내 등수)",
        fontsize=10,
        fontproperties=font,
    )
    table = ax.table(cellText=table_rows, colLabels=header, colWidths=col_widths, cellLoc="left", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 2.7)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_text_props(va="center")
        if font is not None:
            cell.get_text().set_fontproperties(font)
        if row_idx == 0:
            cell.set_facecolor("#dfe6f2")
            cell.get_text().set_fontweight("bold")
    png_path = out_dir / f"batch_composition_{method.label}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHECK][compare_full_flow_schedules.write_batch_composition] wrote={png_path}")


# --------------------------------------------------------------------------- #
# 비교 요약
# --------------------------------------------------------------------------- #
def _summary_metrics(method: MethodData) -> Dict[str, float]:
    wo_counts = [len(batch.member_ids) for batch in method.batches]
    metrics = {label: value for label, value in zip(SCORE_LABELS, method.score_tuple)}
    metrics["batch_count"] = float(len(method.batches))
    metrics["avg_batch_wo_count"] = round(sum(wo_counts) / len(wo_counts), 4)
    metrics["avg_wo_fill_ratio"] = round(sum(wo_counts) / len(wo_counts) / MAX_BATCH_WO_COUNT, 4)
    metrics["total_jobs"] = float(len(method.jobs))
    return metrics


def write_comparison_summary(method_a: MethodData, method_b: MethodData, out_dir: Path, plt, font) -> None:
    metrics_a = _summary_metrics(method_a)
    metrics_b = _summary_metrics(method_b)
    keys = list(metrics_a.keys())
    csv_path = out_dir / "comparison_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", method_a.label, method_b.label, f"delta(B-A)"])
        for key in keys:
            a_value = metrics_a[key]
            b_value = metrics_b[key]
            writer.writerow([key, a_value, b_value, round(b_value - a_value, 4)])
    print(f"[CHECK][compare_full_flow_schedules.write_comparison_summary] wrote={csv_path}")

    table_rows = [
        [key, f"{metrics_a[key]:g}", f"{metrics_b[key]:g}", f"{metrics_b[key] - metrics_a[key]:+g}"]
        for key in keys
    ]
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.5 * len(keys) + 1.2)))
    ax.axis("off")
    ax.set_title(
        f"비교 요약  A={method_a.label}  B={method_b.label}  (하위 gap/makespan은 작을수록 우수)",
        fontsize=10,
        fontproperties=font,
    )
    table = ax.table(
        cellText=table_rows,
        colLabels=["metric", method_a.label, method_b.label, "Δ(B-A)"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if font is not None:
            cell.get_text().set_fontproperties(font)
        if row_idx == 0:
            cell.set_facecolor("#dfe6f2")
            cell.get_text().set_fontweight("bold")
        elif col_idx == 3 and row_idx > 0:
            metric_key = keys[row_idx - 1]
            # score 성분 6개만 "작을수록 우수"라 색으로 방향을 표시한다.
            if metric_key in SCORE_LABELS:
                delta = metrics_b[metric_key] - metrics_a[metric_key]
                if abs(delta) > FLOAT_TOLERANCE:
                    cell.set_facecolor("#e8f5e9" if delta < 0 else "#fdecea")
    png_path = out_dir / "comparison_summary.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHECK][compare_full_flow_schedules.write_comparison_summary] wrote={png_path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="두 Phase 2 full-flow 스케줄을 Gantt/배치구성/요약으로 비교한다.")
    parser.add_argument("--method-a-dir", required=True, help="방법 A full-flow output 디렉터리")
    parser.add_argument("--method-b-dir", required=True, help="방법 B full-flow output 디렉터리")
    parser.add_argument("--label-a", default="heuristic", help="방법 A 라벨")
    parser.add_argument("--label-b", default="proposed", help="방법 B 라벨")
    parser.add_argument("--output-dir", default="output/analysis/schedule_comparison", help="비교 산출물 저장 디렉터리")
    parser.add_argument("--rank-scope", choices=("bay_family", "family", "machine"), default="bay_family", help="작업시간 순위 집합")
    parser.add_argument("--font-path", default=None, help="한글 폰트 경로(선택)")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_a = load_method(Path(args.method_a_dir), args.label_a, args.rank_scope)
    method_b = load_method(Path(args.method_b_dir), args.label_b, args.rank_scope)

    plt = _import_plt()
    font = _resolve_korean_font(args.font_path)
    _apply_font(plt, font)

    render_gantt(method_a, output_dir / f"gantt_{method_a.label}.png", plt, font)
    render_gantt(method_b, output_dir / f"gantt_{method_b.label}.png", plt, font)
    render_compare_gantt(method_a, method_b, output_dir / "gantt_compare.png", plt, font)
    write_batch_composition(method_a, output_dir, plt, font)
    write_batch_composition(method_b, output_dir, plt, font)
    write_comparison_summary(method_a, method_b, output_dir, plt, font)

    print(
        f"[VALIDATION][compare_full_flow_schedules.main] passed=true output_dir={output_dir} "
        f"a={method_a.label} b={method_b.label} rank_scope={args.rank_scope}"
    )


if __name__ == "__main__":
    main()
