"""PDF report-formula synthetic block/W/O generator.

이 모듈은 `[제출본]가공공장_중간보고.pdf`의 발표 수식을 그대로 실행하는
학습용 데이터 생성기다. `데이터분석/shipyard_data_generator.py`처럼 실적
데이터에서 계수를 다시 fit하지 않는다.

생성 흐름:
1. block seed 값 생성: LTH -> MARK_LTH -> CUT_LTH -> STL_QTY -> THK.
2. STL_QTY를 W/O 수로 보고 W/O row를 생성한다.
3. W/O 값을 다시 집계해 최종 block row를 만든다.

주의:
- Phase 1 단독 학습은 기존 block-only bootstrap generator를 유지한다.
- Merged Phase 2와 frozen-feedback full-flow는 W/O가 필요하므로 이 generator를 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from Utils.data.multi_series_cutting_data import build_block_set_id


WO_COLUMNS = (
    "PROJ_NO",
    "BLK_NO",
    "WK_ORD_NO",
    "GYEL",
    "LTH",
    "THK",
    "CUT_LTH",
    "MARK_LTH",
    "BVL_LTH",
    "STL_QTY",
    "BV_QTY",
    "TACT_TIME",
    "PTLST_QTY",
)

BLOCK_COLUMNS = (
    "PROJ_NO",
    "BLK_NO",
    "GYEL",
    "LTH",
    "THK",
    "CUT_LTH",
    "MARK_LTH",
    "BVL_LTH",
    "STL_QTY",
    "BV_QTY",
    "TACT_TIME",
    "PTLST_QTY",
)

# PDF page 10: block-level fixed formulas.
BLOCK_LENGTH_MEAN = 13380.9
BLOCK_LENGTH_STD = 5376.4
BLOCK_LENGTH_MIN = 515.0
BLOCK_LENGTH_MAX = 21995.0
BLOCK_MARK_A = 0.03425
BLOCK_MARK_B = -178.9
BLOCK_MARK_ZERO_INFLATION = 0.0127
BLOCK_MARK_GAMMA_SHAPE = 3.049
BLOCK_MARK_GAMMA_SCALE = 150.8
BLOCK_MARK_GAMMA_SHIFT = -459.9
BLOCK_CUT_A = 1.7056
BLOCK_CUT_B = 59.21
BLOCK_CUT_GAMMA_SHAPE = 3.391
BLOCK_CUT_GAMMA_SCALE = 0.292
BLOCK_STEEL_A = 0.01203
BLOCK_STEEL_B = 2.014
BLOCK_STEEL_GAMMA_SHAPE = 7.616
BLOCK_STEEL_GAMMA_SCALE = 0.128
BLOCK_THICKNESS_A = 3.7816
BLOCK_THICKNESS_B = 13.31
BLOCK_THICKNESS_STD = 4.797

# PDF pages 22-27: W/O-level fixed formulas.
WO_LENGTH_A = 0.95
WO_LENGTH_B = 0.86
WO_LENGTH_POWER = 0.89
WO_THICKNESS_BASE = 12.4
WO_THICKNESS_AMP = 6.7
WO_THICKNESS_DECAY = -2.85
WO_THICKNESS_SKEW_SHAPE = 7.09
WO_THICKNESS_SKEW_LOC = -1.15
WO_THICKNESS_SKEW_SCALE = 1.53
WO_CUT_A = 0.0058
WO_CUT_B = 12.649
WO_CUT_STD = 42.072
WO_MARK_A = 0.0047
WO_MARK_B = -8.293
WO_MARK_STD = 17.452
WO_BEVEL_A_THK = 0.827
WO_BEVEL_A_LTH = 0.0004
WO_BEVEL_A_MARK = 0.0649
WO_BEVEL_B = -12.496
WO_BEVEL_STD = 7.602
WO_BVQ_A_BVL = 0.1447
WO_BVQ_A_CUT = 0.01236
WO_BVQ_B = 1.4136
WO_BVQ_STD = 2.812
WO_PTLST_A_CUT = 0.2156
WO_PTLST_A_LTH = -0.0005
WO_PTLST_A_MARK = -0.1079
WO_PTLST_B = 5.441
WO_PTLST_STD = 7.602
TACT_A_CUT = 0.3037
TACT_A_MARK = 0.1325
TACT_A_THK = 0.4790
TACT_A_PTLST = 0.3840

# PDF는 nearest-spec rounding이라고만 명시한다. 현재 데이터가 0.5 단위 두께를
# 포함하므로 6~36mm 0.5 단위 grid를 고정 spec으로 둔다.
DEFAULT_THICKNESS_SPECS = tuple(round(value * 0.5, 1) for value in range(12, 73))


@dataclass(frozen=True)
class ReportFormulaGeneration:
    """Generated W/O and block DataFrames for one synthetic dataset."""

    wo_df: pd.DataFrame
    block_df: pd.DataFrame


def generate_report_formula_data(
    n_blocks: int,
    seed: int = 2026,
    gyel: str = "NP",
    thickness_specs: Sequence[float] = DEFAULT_THICKNESS_SPECS,
) -> ReportFormulaGeneration:
    """Generate synthetic W/O and block rows using only the PDF formulas."""

    if n_blocks <= 0:
        print(f"[ERROR][report_formula_data_generator.generate_report_formula_data] cause=invalid_n_blocks value={n_blocks}")
        raise ValueError("n_blocks must be positive")
    _validate_thickness_specs(thickness_specs)
    rng = np.random.default_rng(seed)
    wo_rows: List[Dict] = []
    block_rows: List[Dict] = []
    for block_index in range(1, n_blocks + 1):
        project_no = f"SYNTH_PROJ_{(block_index - 1) // 1000 + 1}"
        block_no = f"SYNTH_BLK_{block_index:06d}"
        seed_block = _generate_block_seed_values(rng, thickness_specs)
        block_wo_rows = _generate_work_order_rows(
            rng=rng,
            project_no=project_no,
            block_no=block_no,
            gyel=gyel,
            block_length=float(seed_block["LTH"]),
            block_thickness=float(seed_block["THK"]),
            wo_count=int(seed_block["STL_QTY"]),
            thickness_specs=thickness_specs,
        )
        wo_rows.extend(block_wo_rows)
        block_rows.append(_aggregate_block_row(project_no, block_no, gyel, block_wo_rows))

    wo_df = pd.DataFrame(wo_rows, columns=WO_COLUMNS)
    block_df = pd.DataFrame(block_rows, columns=BLOCK_COLUMNS)
    validate_report_formula_data(wo_df, block_df)
    print(
        "[VALIDATION][report_formula_data_generator.generate_report_formula_data] "
        f"passed=true blocks={len(block_df)} wos={len(wo_df)} seed={seed}"
    )
    return ReportFormulaGeneration(wo_df=wo_df, block_df=block_df)


def build_report_formula_episode_jobs(
    episode_count: int,
    min_blocks: int,
    max_blocks: int,
    seed: int = 2026,
    gyel: str = "NP",
) -> List[Dict]:
    """Merged Phase 2 학습용 가변 크기 W/O episode를 만든다."""

    if episode_count <= 0:
        print(f"[ERROR][report_formula_data_generator.build_report_formula_episode_jobs] cause=invalid_episode_count value={episode_count}")
        raise ValueError("episode_count must be positive")
    if min_blocks <= 0 or max_blocks < min_blocks:
        print(
            "[ERROR][report_formula_data_generator.build_report_formula_episode_jobs] "
            f"cause=invalid_block_range min_blocks={min_blocks} max_blocks={max_blocks}"
        )
        raise ValueError("invalid block range")
    rng = np.random.default_rng(seed)
    episodes: List[Dict] = []
    for episode_index in range(episode_count):
        episode_id = f"PDF{episode_index + 1:05d}"
        block_count = int(rng.integers(min_blocks, max_blocks + 1))
        episode_seed = int(rng.integers(1, 2_147_483_647))
        generated = generate_report_formula_data(
            n_blocks=block_count,
            seed=episode_seed,
            gyel=gyel,
        )
        episodes.append(
            {
                "episode_id": episode_id,
                "problem_id": episode_id,
                "block_count": block_count,
                "job_count": len(generated.wo_df),
                "seed": episode_seed,
                "wo_df": generated.wo_df,
                "block_df": generated.block_df,
                "jobs": jobs_from_report_formula_wo(generated.wo_df, episode_id),
            }
        )
    print(
        "[VALIDATION][report_formula_data_generator.build_report_formula_episode_jobs] "
        f"passed=true episodes={episode_count} min_blocks={min_blocks} max_blocks={max_blocks}"
    )
    return episodes


def jobs_from_report_formula_wo(wo_df: pd.DataFrame, episode_id: str) -> Dict[str, SimpleNamespace]:
    """생성한 W/O 행을 merged Phase 2가 사용하는 Job-like 객체로 변환한다."""

    _require_columns(wo_df, WO_COLUMNS, "wo_df")
    jobs: Dict[str, SimpleNamespace] = {}
    for row_index, row in wo_df.reset_index(drop=True).iterrows():
        job_id = f"{episode_id}_{str(row['WK_ORD_NO']).strip()}"
        block_set_id = build_block_set_id(row["PROJ_NO"], row["GYEL"], row["BLK_NO"])
        tact_time = _positive_float(row["TACT_TIME"], "TACT_TIME", job_id)
        jobs[job_id] = SimpleNamespace(
            job_id=job_id,
            family=str(row["GYEL"]).strip(),
            block_set_id=block_set_id,
            steel_quantity=1,
            plate_length=_positive_float(row["LTH"], "LTH", job_id),
            thickness=_positive_float(row["THK"], "THK", job_id),
            cut_length=_non_negative_float(row["CUT_LTH"], "CUT_LTH", job_id),
            marking_length=_non_negative_float(row["MARK_LTH"], "MARK_LTH", job_id),
            bevel_length=_non_negative_float(row["BVL_LTH"], "BVL_LTH", job_id),
            bevel_quantity=_non_negative_int(row["BV_QTY"], "BV_QTY", job_id),
            part_count=_positive_int(row["PTLST_QTY"], "PTLST_QTY", job_id),
            processing_time=tact_time,
            base_stage_minutes={"cut": tact_time},
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            extra={
                "source_project_no": str(row["PROJ_NO"]).strip(),
                "source_block_no": str(row["BLK_NO"]).strip(),
                "source_wk_ord_no": str(row["WK_ORD_NO"]).strip(),
                "data_role": "report_formula_synthetic_wo",
            },
        )
    if not jobs:
        print("[ERROR][report_formula_data_generator.jobs_from_report_formula_wo] cause=no_jobs")
        raise RuntimeError("generated W/O data produced no jobs")
    return jobs


def scenario_jobs_from_report_formula_jobs(jobs: Mapping[str, object]) -> List[Dict]:
    """Convert generated Job-like objects into scenario job dictionaries.

    Phase 2 full-flow consumes the same scenario contract as real Excel data:
    `scenario["jobs"]` must be a list of dictionaries.  This function is the
    single conversion point from PDF-generated W/O objects to that contract.
    """

    if not jobs:
        print("[ERROR][report_formula_data_generator.scenario_jobs_from_report_formula_jobs] cause=no_jobs")
        raise RuntimeError("scenario conversion requires non-empty generated jobs")
    rows: List[Dict] = []
    for job_id, job in sorted(jobs.items()):
        rows.append(
            {
                "job_id": _required_text(_job_field(job, "job_id"), "job_id", str(job_id)),
                "family": _required_text(_job_field(job, "family"), "family", str(job_id)),
                "block_set_id": _required_text(_job_field(job, "block_set_id"), "block_set_id", str(job_id)),
                "steel_quantity": _positive_int(_job_field(job, "steel_quantity"), "steel_quantity", str(job_id)),
                "plate_length": _positive_float(_job_field(job, "plate_length"), "plate_length", str(job_id)),
                "thickness": _positive_float(_job_field(job, "thickness"), "thickness", str(job_id)),
                "cut_length": _non_negative_float(_job_field(job, "cut_length"), "cut_length", str(job_id)),
                "marking_length": _non_negative_float(_job_field(job, "marking_length"), "marking_length", str(job_id)),
                "bevel_length": _non_negative_float(_job_field(job, "bevel_length"), "bevel_length", str(job_id)),
                "bevel_quantity": _non_negative_int(_job_field(job, "bevel_quantity"), "bevel_quantity", str(job_id)),
                "part_count": _positive_int(_job_field(job, "part_count"), "part_count", str(job_id)),
                "processing_time": _positive_float(_job_field(job, "processing_time"), "processing_time", str(job_id)),
                "base_stage_minutes": {
                    "cut": _positive_float(_job_field(job, "processing_time"), "processing_time", str(job_id))
                },
                "cut_bay": None,
                "source_cut_bay": None,
                "allowed_bay_ids": list(_job_field(job, "allowed_bay_ids")),
                "allowed_machine_ids": list(_job_field(job, "allowed_machine_ids")),
                "prohibited_machine_ids": list(_job_field(job, "prohibited_machine_ids")),
                "extra": dict(_job_field(job, "extra")),
            }
        )
    return rows


def validate_report_formula_data(wo_df: pd.DataFrame, block_df: pd.DataFrame) -> None:
    """Validate generated W/O/block rows and their aggregation relationship."""

    _require_columns(wo_df, WO_COLUMNS, "wo_df")
    _require_columns(block_df, BLOCK_COLUMNS, "block_df")
    if wo_df.empty or block_df.empty:
        print("[ERROR][report_formula_data_generator.validate_report_formula_data] cause=empty_generated_data")
        raise RuntimeError("generated W/O and block data must not be empty")
    for frame_name, frame, positive_columns, non_negative_columns in (
        ("wo_df", wo_df, ("LTH", "THK", "TACT_TIME", "PTLST_QTY", "STL_QTY"), ("CUT_LTH", "MARK_LTH", "BVL_LTH", "BV_QTY")),
        ("block_df", block_df, ("LTH", "THK", "TACT_TIME", "STL_QTY", "PTLST_QTY"), ("CUT_LTH", "MARK_LTH", "BVL_LTH", "BV_QTY")),
    ):
        if frame.isna().any().any():
            print(f"[ERROR][report_formula_data_generator.validate_report_formula_data] cause=nan_values frame={frame_name}")
            raise RuntimeError(f"NaN values in {frame_name}")
        for column in positive_columns:
            if (pd.to_numeric(frame[column], errors="coerce") <= 0).any():
                print(
                    "[ERROR][report_formula_data_generator.validate_report_formula_data] "
                    f"cause=non_positive_values frame={frame_name} column={column}"
                )
                raise RuntimeError(f"non-positive {column} in {frame_name}")
        for column in non_negative_columns:
            if (pd.to_numeric(frame[column], errors="coerce") < 0).any():
                print(
                    "[ERROR][report_formula_data_generator.validate_report_formula_data] "
                    f"cause=negative_values frame={frame_name} column={column}"
                )
                raise RuntimeError(f"negative {column} in {frame_name}")

    grouped = wo_df.groupby(["PROJ_NO", "GYEL", "BLK_NO"], sort=True)
    for _, block in block_df.iterrows():
        key = (block["PROJ_NO"], block["GYEL"], block["BLK_NO"])
        if key not in grouped.groups:
            print(f"[ERROR][report_formula_data_generator.validate_report_formula_data] cause=missing_block_wos key={key}")
            raise RuntimeError(f"missing generated W/O rows for block {key}")
        rows = grouped.get_group(key)
        _assert_close(float(block["LTH"]), float(rows["LTH"].max()), "LTH", key)
        _assert_close(float(block["THK"]), float(rows["THK"].max()), "THK", key)
        _assert_close(float(block["TACT_TIME"]), float(rows["TACT_TIME"].max()), "TACT_TIME", key)
        _assert_close(float(block["CUT_LTH"]), float(rows["CUT_LTH"].sum()), "CUT_LTH", key)
        _assert_close(float(block["MARK_LTH"]), float(rows["MARK_LTH"].sum()), "MARK_LTH", key)
        _assert_close(float(block["BVL_LTH"]), float(rows["BVL_LTH"].sum()), "BVL_LTH", key)
        _assert_close(float(block["BV_QTY"]), float(rows["BV_QTY"].sum()), "BV_QTY", key)
        _assert_close(float(block["PTLST_QTY"]), float(rows["PTLST_QTY"].sum()), "PTLST_QTY", key)
        if int(block["STL_QTY"]) != len(rows):
            print(
                "[ERROR][report_formula_data_generator.validate_report_formula_data] "
                f"cause=stl_qty_mismatch key={key} block={block['STL_QTY']} wo_count={len(rows)}"
            )
            raise RuntimeError(f"STL_QTY mismatch for block {key}")


def _generate_block_seed_values(rng: np.random.Generator, thickness_specs: Sequence[float]) -> Dict[str, float | int]:
    block_length = float(np.clip(rng.normal(BLOCK_LENGTH_MEAN, BLOCK_LENGTH_STD), BLOCK_LENGTH_MIN, BLOCK_LENGTH_MAX))
    if rng.random() < BLOCK_MARK_ZERO_INFLATION:
        block_mark = 0.0
    else:
        block_mark = max(
            0.0,
            BLOCK_MARK_A * block_length
            + BLOCK_MARK_B
            + rng.gamma(BLOCK_MARK_GAMMA_SHAPE, BLOCK_MARK_GAMMA_SCALE)
            + BLOCK_MARK_GAMMA_SHIFT,
        )
    block_cut = max(0.0, (BLOCK_CUT_A * block_mark + BLOCK_CUT_B) * rng.gamma(BLOCK_CUT_GAMMA_SHAPE, BLOCK_CUT_GAMMA_SCALE))
    steel_quantity = max(1, int(round((BLOCK_STEEL_A * block_cut + BLOCK_STEEL_B) * rng.gamma(BLOCK_STEEL_GAMMA_SHAPE, BLOCK_STEEL_GAMMA_SCALE))))
    block_thickness_raw = BLOCK_THICKNESS_A * np.log(steel_quantity) + BLOCK_THICKNESS_B + rng.normal(0.0, BLOCK_THICKNESS_STD)
    return {
        "LTH": block_length,
        "MARK_LTH": block_mark,
        "CUT_LTH": block_cut,
        "STL_QTY": steel_quantity,
        "THK": _nearest_spec(block_thickness_raw, thickness_specs),
    }


def _generate_work_order_rows(
    rng: np.random.Generator,
    project_no: str,
    block_no: str,
    gyel: str,
    block_length: float,
    block_thickness: float,
    wo_count: int,
    thickness_specs: Sequence[float],
) -> List[Dict]:
    if wo_count <= 0:
        print(f"[ERROR][report_formula_data_generator._generate_work_order_rows] cause=invalid_wo_count value={wo_count}")
        raise RuntimeError("wo_count must be positive")
    u_values = np.arange(wo_count, dtype=float) / max(wo_count, 1)
    length_factor = 1.0 - (WO_LENGTH_A - WO_LENGTH_B / wo_count) * np.power(u_values, WO_LENGTH_POWER)
    lengths = np.maximum(1.0, block_length * length_factor)
    thicknesses = []
    for length in lengths:
        ratio = float(length) / block_length
        raw_thickness = (
            WO_THICKNESS_BASE
            + WO_THICKNESS_AMP * np.exp(WO_THICKNESS_DECAY * ratio)
            + _sample_wo_thickness_noise(rng)
        )
        thicknesses.append(min(block_thickness, _nearest_spec(max(1.0, raw_thickness), thickness_specs)))
    if max(thicknesses) < block_thickness:
        thicknesses[0] = block_thickness

    rows: List[Dict] = []
    for wo_index, (length, thickness) in enumerate(zip(lengths, thicknesses), start=1):
        cut_length = max(0.0, WO_CUT_A * length + WO_CUT_B + rng.normal(0.0, WO_CUT_STD))
        mark_length = max(0.0, WO_MARK_A * length + WO_MARK_B + rng.normal(0.0, WO_MARK_STD))
        bevel_length = max(
            0.0,
            WO_BEVEL_B
            + WO_BEVEL_A_THK * thickness
            + WO_BEVEL_A_LTH * length
            + WO_BEVEL_A_MARK * mark_length
            + rng.normal(0.0, WO_BEVEL_STD),
        )
        bevel_quantity = 0
        if bevel_length > 0:
            bevel_quantity = max(1, int(round(WO_BVQ_A_BVL * bevel_length + WO_BVQ_A_CUT * cut_length + WO_BVQ_B + rng.normal(0.0, WO_BVQ_STD))))
        part_count = max(1, int(round(WO_PTLST_B + WO_PTLST_A_CUT * cut_length + WO_PTLST_A_LTH * length + WO_PTLST_A_MARK * mark_length + rng.normal(0.0, WO_PTLST_STD))))
        tact_time = TACT_A_CUT * cut_length + TACT_A_MARK * mark_length + TACT_A_THK * thickness + TACT_A_PTLST * part_count
        rows.append(
            {
                "PROJ_NO": project_no,
                "BLK_NO": block_no,
                "WK_ORD_NO": f"{project_no}NPF{block_no}WO_{wo_index}",
                "GYEL": gyel,
                "LTH": round(float(length), 6),
                "THK": round(float(thickness), 6),
                "CUT_LTH": round(float(cut_length), 6),
                "MARK_LTH": round(float(mark_length), 6),
                "BVL_LTH": round(float(bevel_length), 6),
                "STL_QTY": 1,
                "BV_QTY": int(bevel_quantity),
                "TACT_TIME": round(float(tact_time), 6),
                "PTLST_QTY": int(part_count),
            }
        )
    return rows


def _aggregate_block_row(project_no: str, block_no: str, gyel: str, wo_rows: Sequence[Mapping]) -> Dict:
    frame = pd.DataFrame(wo_rows)
    return {
        "PROJ_NO": project_no,
        "BLK_NO": block_no,
        "GYEL": gyel,
        "LTH": round(float(frame["LTH"].max()), 6),
        "THK": round(float(frame["THK"].max()), 6),
        "CUT_LTH": round(float(frame["CUT_LTH"].sum()), 6),
        "MARK_LTH": round(float(frame["MARK_LTH"].sum()), 6),
        "BVL_LTH": round(float(frame["BVL_LTH"].sum()), 6),
        "STL_QTY": int(len(frame)),
        "BV_QTY": int(frame["BV_QTY"].sum()),
        "TACT_TIME": round(float(frame["TACT_TIME"].max()), 6),
        "PTLST_QTY": int(frame["PTLST_QTY"].sum()),
    }


def _sample_wo_thickness_noise(rng: np.random.Generator) -> float:
    try:
        from scipy.stats import skewnorm
    except ImportError as exc:
        print("[ERROR][report_formula_data_generator._sample_wo_thickness_noise] cause=missing_scipy")
        raise RuntimeError("scipy is required for PDF SkewNormal thickness formula") from exc
    return float(
        skewnorm.rvs(
            WO_THICKNESS_SKEW_SHAPE,
            loc=WO_THICKNESS_SKEW_LOC,
            scale=WO_THICKNESS_SKEW_SCALE,
            random_state=rng,
        )
    )


def _nearest_spec(value: float, specs: Sequence[float]) -> float:
    spec_array = np.asarray(specs, dtype=float)
    return float(spec_array[np.argmin(np.abs(spec_array - float(value)))])


def _validate_thickness_specs(specs: Sequence[float]) -> None:
    if not specs:
        print("[ERROR][report_formula_data_generator._validate_thickness_specs] cause=no_thickness_specs")
        raise RuntimeError("thickness_specs must not be empty")
    if any(float(value) <= 0 for value in specs):
        print(f"[ERROR][report_formula_data_generator._validate_thickness_specs] cause=non_positive_spec specs={specs}")
        raise RuntimeError("thickness specs must be positive")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        print(f"[ERROR][report_formula_data_generator._require_columns] cause=missing_columns frame={frame_name} missing={missing}")
        raise RuntimeError(f"missing columns in {frame_name}: {missing}")


def _positive_float(value: object, column: str, context: str) -> float:
    parsed = _numeric_value(value, column, context)
    if parsed <= 0:
        print(f"[ERROR][report_formula_data_generator._positive_float] cause=non_positive column={column} context={context} value={value}")
        raise RuntimeError(f"non-positive {column}: {context}")
    return parsed


def _non_negative_float(value: object, column: str, context: str) -> float:
    parsed = _numeric_value(value, column, context)
    if parsed < 0:
        print(f"[ERROR][report_formula_data_generator._non_negative_float] cause=negative column={column} context={context} value={value}")
        raise RuntimeError(f"negative {column}: {context}")
    return parsed


def _positive_int(value: object, column: str, context: str) -> int:
    parsed = _positive_float(value, column, context)
    if not float(parsed).is_integer():
        print(f"[ERROR][report_formula_data_generator._positive_int] cause=non_integer column={column} context={context} value={value}")
        raise RuntimeError(f"non-integer {column}: {context}")
    return int(parsed)


def _non_negative_int(value: object, column: str, context: str) -> int:
    parsed = _non_negative_float(value, column, context)
    if not float(parsed).is_integer():
        print(f"[ERROR][report_formula_data_generator._non_negative_int] cause=non_integer column={column} context={context} value={value}")
        raise RuntimeError(f"non-integer {column}: {context}")
    return int(parsed)


def _numeric_value(value: object, column: str, context: str) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        print(f"[ERROR][report_formula_data_generator._numeric_value] cause=non_numeric column={column} context={context} value={value}")
        raise RuntimeError(f"non-numeric {column}: {context}")
    return float(parsed)


def _assert_close(actual: float, expected: float, column: str, key: tuple[str, str]) -> None:
    if abs(actual - expected) > 1e-4:
        print(
            "[ERROR][report_formula_data_generator._assert_close] "
            f"cause=aggregate_mismatch key={key} column={column} actual={actual} expected={expected}"
        )
        raise RuntimeError(f"aggregate mismatch for {key}::{column}")


def _job_field(job: object, field_name: str) -> object:
    if isinstance(job, Mapping):
        if field_name not in job:
            print(
                "[ERROR][report_formula_data_generator._job_field] "
                f"cause=missing_key field={field_name} job={job}"
            )
            raise RuntimeError(f"generated job missing field: {field_name}")
        return job[field_name]
    if not hasattr(job, field_name):
        print(
            "[ERROR][report_formula_data_generator._job_field] "
            f"cause=missing_attr field={field_name} job={job}"
        )
        raise RuntimeError(f"generated job missing field: {field_name}")
    return getattr(job, field_name)


def _required_text(value: object, column: str, context: str) -> str:
    parsed = str(value or "").strip()
    if not parsed:
        print(
            "[ERROR][report_formula_data_generator._required_text] "
            f"cause=empty_text column={column} context={context} value={value}"
        )
        raise RuntimeError(f"empty {column}: {context}")
    return parsed
