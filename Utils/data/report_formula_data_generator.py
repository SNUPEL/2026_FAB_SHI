"""PDF report-formula synthetic block/W/O generator.

이 모듈은 `[제출본]가공공장_중간보고.pdf`의 발표 수식을 그대로 실행하는
학습용 데이터 생성기다. `데이터분석/shipyard_data_generator.py`처럼 실적
데이터에서 계수를 다시 fit하지 않는다.

생성 흐름:
1. block seed 값 생성: LTH -> MARK_LTH -> CUT_LTH -> 잠재 W/O 수 -> THK.
2. 잠재 W/O 수만큼 W/O row를 생성한다.
3. W/O별 STL_QTY는 실적 조건부 분포에서 별도로 생성한다.
4. W/O 값을 다시 집계해 최종 block row를 만든다.

주의:
- NP Phase 1/Phase 2 학습은 모두 이 생성기를 사용한다.
- 계열별로 다시 적합한 다계열 분석 생성기는 이 고정 NP 학습 경로에서 사용하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from Utils.data.multi_series_cutting_data import build_block_set_id


WO_COLUMNS = (
    "PROJ_NO",
    "BLK_NO",
    "WK_ORD_NO",
    "GYEL",
    "LTH",
    "BTH",
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
    "BTH",
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

# 신규 실적에는 PDF에 없던 폭(BTH)이 있으므로, 전 계열에 같은 로그선형식 구조를
# 사용하고 계열별 계수만 실적에서 적합한다. 식의 입력은 BTH보다 먼저 생성되는
# W/O 물리·가공 특성으로 한정한다.
BTH_FORMULA_FEATURES = (
    "LTH",
    "THK",
    "MARK_LTH",
    "CUT_LTH",
    "BVL_LTH",
    "BV_QTY",
    "PTLST_QTY",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MULTI_SERIES_WO_SOURCE = REPO_ROOT / "input" / "260724_절단WO_데이터_None.xlsx"
BTH_RANDOM_STREAM_SALT = 7_142_026
STL_RANDOM_STREAM_SALT = 7_152_026
STL_LOCAL_WEIGHT = 0.75

# PDF의 nearest-spec rounding에 사용하는 과거 NP W/O 실적 두께 규격이다.
# 관측되지 않은 6.5/7.0mm 등을 임의 규격으로 추가하지 않는다.
DEFAULT_THICKNESS_SPECS = (
    6.0,
    9.0,
    10.0,
    10.5,
    11.0,
    11.5,
    12.0,
    12.5,
    13.0,
    13.5,
    14.0,
    14.5,
    15.0,
    15.5,
    16.0,
    16.5,
    17.0,
    17.5,
    18.0,
    18.5,
    19.0,
    19.5,
    20.0,
    20.5,
    21.0,
    21.5,
    22.0,
    22.5,
    23.0,
    24.0,
    24.5,
    25.0,
    26.0,
    26.5,
    27.0,
    27.5,
    28.0,
    28.5,
    29.0,
    30.0,
    31.0,
    32.0,
    34.0,
    35.0,
    36.0,
)

# 정규오차 식에서 모든 W/O 값이 0 이하인 극단 표본은 같은 식으로 다시 뽑는다.
# 이 횟수 안에도 유효 표본이 없으면 임의 분배하지 않고 데이터 생성을 실패시킨다.
MAX_REGRESSION_RESAMPLE_ATTEMPTS = 1_000


@dataclass(frozen=True)
class ReportFormulaGeneration:
    """Generated W/O and block DataFrames for one synthetic dataset."""

    wo_df: pd.DataFrame
    block_df: pd.DataFrame


@dataclass(frozen=True)
class BthFormulaProfile:
    """계열별 W/O 폭 로그선형식과 관측 폭 규격."""

    series: str
    coefficients: tuple[float, ...]
    residual_std: float
    r_squared: float
    observed_specs: tuple[float, ...]


@dataclass(frozen=True)
class StlQuantityProfile:
    """계열별 W/O 강재수량의 조건부 이웃분포."""

    series: str
    features: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    normalized_actual_features: np.ndarray
    actual_values: np.ndarray
    classes: tuple[int, ...]
    priors: tuple[float, ...]


def fit_bth_formula(frame: pd.DataFrame, series: str) -> BthFormulaProfile:
    """실적 W/O에서 공통 형태의 계열별 BTH 로그선형식을 적합한다."""

    normalized_series = str(series or "").strip().upper()
    required = ("GYEL", "BTH", *BTH_FORMULA_FEATURES)
    _require_columns(frame, required, "bth_formula_source")
    selected = frame.loc[
        frame["GYEL"].astype(str).str.strip().str.upper() == normalized_series,
        ["BTH", *BTH_FORMULA_FEATURES],
    ].apply(pd.to_numeric, errors="coerce")
    minimum_rows = len(BTH_FORMULA_FEATURES) + 2
    if len(selected) < minimum_rows:
        print(
            "[ERROR][report_formula_data_generator.fit_bth_formula] "
            f"cause=insufficient_series_rows series={normalized_series} "
            f"rows={len(selected)} required={minimum_rows}"
        )
        raise RuntimeError(f"insufficient BTH formula rows for {normalized_series}")
    if selected.isna().any(axis=None):
        print(
            "[ERROR][report_formula_data_generator.fit_bth_formula] "
            f"cause=non_numeric_or_missing series={normalized_series}"
        )
        raise RuntimeError(f"invalid BTH formula source for {normalized_series}")
    if (selected["BTH"] <= 0).any() or (selected[list(BTH_FORMULA_FEATURES)] < 0).any(axis=None):
        print(
            "[ERROR][report_formula_data_generator.fit_bth_formula] "
            f"cause=out_of_domain_value series={normalized_series}"
        )
        raise RuntimeError(f"out-of-domain BTH formula source for {normalized_series}")

    design = np.column_stack(
        [
            np.ones(len(selected), dtype=float),
            np.log1p(selected[list(BTH_FORMULA_FEATURES)].to_numpy(dtype=float)),
        ]
    )
    target = np.log(selected["BTH"].to_numpy(dtype=float))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank != design.shape[1]:
        print(
            "[ERROR][report_formula_data_generator.fit_bth_formula] "
            f"cause=rank_deficient series={normalized_series} rank={rank} expected={design.shape[1]}"
        )
        raise RuntimeError(f"rank-deficient BTH formula for {normalized_series}")
    residuals = target - design @ coefficients
    residual_std = float(np.std(residuals, ddof=1))
    if not np.isfinite(residual_std) or residual_std < 0:
        print(
            "[ERROR][report_formula_data_generator.fit_bth_formula] "
            f"cause=invalid_residual_std series={normalized_series} value={residual_std}"
        )
        raise RuntimeError(f"invalid BTH residual for {normalized_series}")
    total_variation = float(np.sum((target - target.mean()) ** 2))
    if total_variation <= 0:
        print(
            "[ERROR][report_formula_data_generator.fit_bth_formula] "
            f"cause=constant_target series={normalized_series}"
        )
        raise RuntimeError(f"constant BTH target for {normalized_series}")
    r_squared = float(1.0 - np.sum(residuals**2) / total_variation)
    observed_specs = tuple(sorted(float(value) for value in selected["BTH"].unique()))
    return BthFormulaProfile(
        series=normalized_series,
        coefficients=tuple(float(value) for value in coefficients),
        residual_std=residual_std,
        r_squared=r_squared,
        observed_specs=observed_specs,
    )


def sample_bth_formula(
    features: pd.DataFrame,
    profile: BthFormulaProfile,
    rng: np.random.Generator,
) -> np.ndarray:
    """계열별 식으로 BTH를 생성하고 실제 관측 폭 규격에 스냅한다."""

    _require_columns(features, BTH_FORMULA_FEATURES, "bth_formula_features")
    numeric = features[list(BTH_FORMULA_FEATURES)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any(axis=None) or (numeric < 0).any(axis=None):
        print(
            "[ERROR][report_formula_data_generator.sample_bth_formula] "
            f"cause=invalid_features series={profile.series}"
        )
        raise RuntimeError(f"invalid BTH formula features for {profile.series}")
    coefficients = np.asarray(profile.coefficients, dtype=float)
    expected_size = len(BTH_FORMULA_FEATURES) + 1
    if coefficients.shape != (expected_size,) or not np.isfinite(coefficients).all():
        print(
            "[ERROR][report_formula_data_generator.sample_bth_formula] "
            f"cause=invalid_coefficients series={profile.series} shape={coefficients.shape}"
        )
        raise RuntimeError(f"invalid BTH coefficients for {profile.series}")
    specs = np.asarray(profile.observed_specs, dtype=float)
    if specs.ndim != 1 or len(specs) == 0 or not np.isfinite(specs).all() or (specs <= 0).any():
        print(
            "[ERROR][report_formula_data_generator.sample_bth_formula] "
            f"cause=invalid_observed_specs series={profile.series}"
        )
        raise RuntimeError(f"invalid BTH specs for {profile.series}")

    design = np.column_stack(
        [np.ones(len(numeric), dtype=float), np.log1p(numeric.to_numpy(dtype=float))]
    )
    noise = (
        np.zeros(len(numeric), dtype=float)
        if profile.residual_std == 0.0
        else rng.normal(0.0, profile.residual_std, len(numeric))
    )
    continuous = np.clip(np.exp(design @ coefficients + noise), specs[0], specs[-1])
    insertion = np.searchsorted(specs, continuous, side="left")
    insertion = np.clip(insertion, 0, len(specs) - 1)
    lower = np.maximum(insertion - 1, 0)
    use_lower = np.abs(continuous - specs[lower]) <= np.abs(specs[insertion] - continuous)
    return specs[np.where(use_lower, lower, insertion)]


def fit_stl_quantity_profile(frame: pd.DataFrame, series: str) -> StlQuantityProfile:
    """실적 W/O에서 핵심 가공특성 조건부 STL_QTY 분포를 적합한다."""

    normalized_series = str(series or "").strip().upper()
    required = ("GYEL", "STL_QTY", *BTH_FORMULA_FEATURES)
    _require_columns(frame, required, "stl_quantity_source")
    selected = frame.loc[
        frame["GYEL"].astype(str).str.strip().str.upper().eq(normalized_series),
        ["STL_QTY", *BTH_FORMULA_FEATURES],
    ].apply(pd.to_numeric, errors="coerce")
    if len(selected) < 3 or selected.isna().any(axis=None):
        print(
            "[ERROR][report_formula_data_generator.fit_stl_quantity_profile] "
            f"cause=invalid_series_rows series={normalized_series} rows={len(selected)}"
        )
        raise RuntimeError(f"invalid STL_QTY profile source for {normalized_series}")
    if (selected["STL_QTY"] < 0).any() or not np.allclose(
        selected["STL_QTY"], np.round(selected["STL_QTY"])
    ):
        print(
            "[ERROR][report_formula_data_generator.fit_stl_quantity_profile] "
            f"cause=invalid_target series={normalized_series}"
        )
        raise RuntimeError(f"invalid STL_QTY target for {normalized_series}")

    feature_values = selected[list(BTH_FORMULA_FEATURES)].to_numpy(dtype=float)
    means = feature_values.mean(axis=0)
    scales = feature_values.std(axis=0, ddof=0)
    active = scales > 0
    if not active.any():
        print(
            "[ERROR][report_formula_data_generator.fit_stl_quantity_profile] "
            f"cause=all_features_constant series={normalized_series}"
        )
        raise RuntimeError(f"constant STL_QTY conditioning features for {normalized_series}")
    features = tuple(np.asarray(BTH_FORMULA_FEATURES)[active].tolist())
    normalized = (feature_values[:, active] - means[active]) / scales[active]
    actual_values = selected["STL_QTY"].to_numpy(dtype=int)
    classes = np.unique(actual_values)
    priors = np.asarray([(actual_values == value).mean() for value in classes], dtype=float)
    return StlQuantityProfile(
        series=normalized_series,
        features=features,
        feature_mean=tuple(float(value) for value in means[active]),
        feature_scale=tuple(float(value) for value in scales[active]),
        normalized_actual_features=normalized,
        actual_values=actual_values,
        classes=tuple(int(value) for value in classes),
        priors=tuple(float(value) for value in priors),
    )


def sample_stl_quantity(
    features: pd.DataFrame,
    profile: StlQuantityProfile,
    rng: np.random.Generator,
    neighbor_count: int = 32,
) -> np.ndarray:
    """조건부 이웃확률과 실적 주변분포를 함께 지켜 STL_QTY를 생성한다."""

    if neighbor_count < 1:
        print(
            "[ERROR][report_formula_data_generator.sample_stl_quantity] "
            f"cause=invalid_neighbor_count value={neighbor_count}"
        )
        raise ValueError("neighbor_count must be positive")
    _require_columns(features, profile.features, "stl_quantity_features")
    numeric = features[list(profile.features)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any(axis=None):
        print(
            "[ERROR][report_formula_data_generator.sample_stl_quantity] "
            f"cause=invalid_features series={profile.series}"
        )
        raise RuntimeError(f"invalid STL_QTY features for {profile.series}")
    means = np.asarray(profile.feature_mean, dtype=float)
    scales = np.asarray(profile.feature_scale, dtype=float)
    actual_features = np.asarray(profile.normalized_actual_features, dtype=float)
    actual_values = np.asarray(profile.actual_values, dtype=int)
    classes = np.asarray(profile.classes, dtype=int)
    priors = np.asarray(profile.priors, dtype=float)
    if (
        actual_features.ndim != 2
        or actual_features.shape[0] != len(actual_values)
        or actual_features.shape[1] != len(profile.features)
        or means.shape != scales.shape
        or means.shape != (len(profile.features),)
        or (scales <= 0).any()
    ):
        print(
            "[ERROR][report_formula_data_generator.sample_stl_quantity] "
            f"cause=invalid_profile series={profile.series}"
        )
        raise RuntimeError(f"invalid STL_QTY profile for {profile.series}")

    target = (numeric.to_numpy(dtype=float) - means) / scales
    k = min(int(neighbor_count), len(actual_values))
    _, nearest = cKDTree(actual_features).query(target, k=k)
    nearest = np.asarray(nearest, dtype=int)
    if nearest.ndim == 1:
        nearest = nearest.reshape(-1, 1)
    local = np.column_stack([(actual_values[nearest] == value).mean(axis=1) for value in classes])
    probabilities = STL_LOCAL_WEIGHT * local + (1.0 - STL_LOCAL_WEIGHT) * priors

    expected_counts = priors * len(target)
    target_counts = np.floor(expected_counts).astype(int)
    remainder = len(target) - int(target_counts.sum())
    if remainder:
        order = np.argsort(-(expected_counts - target_counts), kind="stable")
        target_counts[order[:remainder]] += 1
    majority = int(np.argmax(target_counts))
    class_indices = np.full(len(target), majority, dtype=int)
    available = np.ones(len(target), dtype=bool)
    for class_index in np.argsort(target_counts):
        count = int(target_counts[class_index])
        if class_index == majority or count == 0:
            continue
        candidates = np.flatnonzero(available)
        scores = (
            np.log(probabilities[candidates, class_index] + 1e-12)
            - np.log(probabilities[candidates, majority] + 1e-12)
            + rng.gumbel(size=len(candidates))
        )
        selected = candidates[np.argpartition(scores, -count)[-count:]]
        class_indices[selected] = int(class_index)
        available[selected] = False
    return classes[class_indices]


@lru_cache(maxsize=16)
def load_bth_formula_profile(source_path: str, series: str) -> BthFormulaProfile:
    """같은 프로세스에서 계열별 BTH 식을 한 번만 적합한다."""

    path = Path(source_path)
    if not path.exists():
        print(
            "[ERROR][report_formula_data_generator.load_bth_formula_profile] "
            f"cause=source_not_found path={path} series={series}"
        )
        raise FileNotFoundError(path)
    frame = pd.read_excel(path, sheet_name="Sheet1")
    profile = fit_bth_formula(frame, series=series)
    print(
        "[CHECK][report_formula_data_generator.load_bth_formula_profile] "
        f"series={profile.series} rows={len(frame)} specs={len(profile.observed_specs)} path={path}"
    )
    return profile


@lru_cache(maxsize=16)
def load_stl_quantity_profile(source_path: str, series: str) -> StlQuantityProfile:
    """같은 프로세스에서 계열별 STL_QTY 조건부분포를 한 번만 적합한다."""

    path = Path(source_path)
    if not path.exists():
        print(
            "[ERROR][report_formula_data_generator.load_stl_quantity_profile] "
            f"cause=source_not_found path={path} series={series}"
        )
        raise FileNotFoundError(path)
    frame = pd.read_excel(path, sheet_name="Sheet1")
    profile = fit_stl_quantity_profile(frame, series=series)
    print(
        "[CHECK][report_formula_data_generator.load_stl_quantity_profile] "
        f"series={profile.series} rows={len(frame)} classes={profile.classes} path={path}"
    )
    return profile


def generate_report_formula_data(
    n_blocks: int,
    seed: int = 2026,
    gyel: str = "NP",
    thickness_specs: Sequence[float] = DEFAULT_THICKNESS_SPECS,
    bth_source_path: str | Path = DEFAULT_MULTI_SERIES_WO_SOURCE,
    bth_profile: BthFormulaProfile | None = None,
    stl_quantity_profile: StlQuantityProfile | None = None,
    wo_counts: Sequence[int] | None = None,
    block_seeds: Sequence[int] | None = None,
) -> ReportFormulaGeneration:
    """Generate synthetic W/O and block rows using only the PDF formulas."""

    if n_blocks <= 0:
        print(f"[ERROR][report_formula_data_generator.generate_report_formula_data] cause=invalid_n_blocks value={n_blocks}")
        raise ValueError("n_blocks must be positive")
    normalized_gyel = str(gyel).strip().upper()
    if normalized_gyel != "NP":
        print(
            "[ERROR][report_formula_data_generator.generate_report_formula_data] "
            f"cause=unsupported_fixed_formula_series gyel={gyel} supported=NP"
        )
        raise RuntimeError("PDF fixed-formula generator supports NP only")
    _validate_thickness_specs(thickness_specs)
    resolved_wo_counts = _validate_optional_positive_integer_sequence(
        values=wo_counts,
        expected_length=n_blocks,
        field_name="wo_counts",
    )
    resolved_block_seeds = _validate_optional_positive_integer_sequence(
        values=block_seeds,
        expected_length=n_blocks,
        field_name="block_seeds",
    )
    rng = np.random.default_rng(seed)
    wo_rows: List[Dict] = []
    block_rows: List[Dict] = []
    for block_index in range(1, n_blocks + 1):
        project_no = f"SYNTH_PROJ_{(block_index - 1) // 1000 + 1}"
        block_no = f"SYNTH_BLK_{block_index:06d}"
        block_rng = (
            np.random.default_rng(resolved_block_seeds[block_index - 1])
            if resolved_block_seeds is not None
            else rng
        )
        requested_wo_count = (
            resolved_wo_counts[block_index - 1]
            if resolved_wo_counts is not None
            else None
        )
        seed_block = _generate_block_seed_values(
            block_rng,
            thickness_specs,
            wo_count_override=requested_wo_count,
        )
        block_wo_rows = _generate_work_order_rows(
            rng=block_rng,
            project_no=project_no,
            block_no=block_no,
            gyel=normalized_gyel,
            block_length=float(seed_block["LTH"]),
            block_thickness=float(seed_block["THK"]),
            block_mark_length=float(seed_block["MARK_LTH"]),
            block_cut_length=float(seed_block["CUT_LTH"]),
            wo_count=int(seed_block["WO_QTY"]),
            thickness_specs=thickness_specs,
        )
        wo_rows.extend(block_wo_rows)
        block_rows.append(_aggregate_block_row(project_no, block_no, normalized_gyel, block_wo_rows))

    wo_df = pd.DataFrame(wo_rows)
    if (bth_profile is None) != (stl_quantity_profile is None):
        print(
            "[ERROR][report_formula_data_generator.generate_report_formula_data] "
            "cause=incomplete_fixed_profile bth_and_stl_must_be_supplied_together"
        )
        raise RuntimeError("BTH and STL profiles must be supplied together")
    if bth_profile is None:
        resolved_source = str(Path(bth_source_path).resolve())
        bth_profile = load_bth_formula_profile(resolved_source, normalized_gyel)
        stl_quantity_profile = load_stl_quantity_profile(resolved_source, normalized_gyel)
    if bth_profile.series != normalized_gyel or stl_quantity_profile.series != normalized_gyel:
        print(
            "[ERROR][report_formula_data_generator.generate_report_formula_data] "
            f"cause=profile_series_mismatch gyel={normalized_gyel} "
            f"bth={bth_profile.series} stl={stl_quantity_profile.series}"
        )
        raise RuntimeError("fixed profile series does not match generated series")
    bth_rng = np.random.default_rng(np.random.SeedSequence([int(seed), BTH_RANDOM_STREAM_SALT]))
    wo_df["BTH"] = sample_bth_formula(wo_df, bth_profile, bth_rng)
    stl_rng = np.random.default_rng(np.random.SeedSequence([int(seed), STL_RANDOM_STREAM_SALT]))
    wo_df["STL_QTY"] = sample_stl_quantity(wo_df, stl_quantity_profile, stl_rng)
    wo_df = wo_df[list(WO_COLUMNS)]
    bth_by_block = wo_df.groupby(["PROJ_NO", "GYEL", "BLK_NO"], sort=True)["BTH"].max()
    stl_by_block = wo_df.groupby(["PROJ_NO", "GYEL", "BLK_NO"], sort=True)["STL_QTY"].sum()
    for block_row in block_rows:
        key = (block_row["PROJ_NO"], block_row["GYEL"], block_row["BLK_NO"])
        block_row["BTH"] = float(bth_by_block.loc[key])
        block_row["STL_QTY"] = int(stl_by_block.loc[key])
    block_df = pd.DataFrame(block_rows, columns=BLOCK_COLUMNS)
    validate_report_formula_data(wo_df, block_df)
    print(
        "[VALIDATION][report_formula_data_generator.generate_report_formula_data] "
        f"passed=true blocks={len(block_df)} wos={len(wo_df)} seed={seed}"
    )
    return ReportFormulaGeneration(wo_df=wo_df, block_df=block_df)


def generate_report_formula_block_seeds(
    n_blocks: int,
    seed: int,
    thickness_specs: Sequence[float] = DEFAULT_THICKNESS_SPECS,
) -> pd.DataFrame:
    """PPT 블록 수식으로 물리 블록 목표와 전체 W/O 수를 먼저 생성한다."""

    if n_blocks <= 0:
        print(
            "[ERROR][report_formula_data_generator.generate_report_formula_block_seeds] "
            f"cause=invalid_n_blocks value={n_blocks}"
        )
        raise ValueError("n_blocks must be positive")
    _validate_thickness_specs(thickness_specs)
    rng = np.random.default_rng(seed)
    rows = []
    for physical_index in range(1, n_blocks + 1):
        values = _generate_block_seed_values(rng, thickness_specs)
        rows.append({"physical_index": physical_index, **values})
    return pd.DataFrame(rows)


def build_report_formula_episode_jobs(
    episode_count: int,
    min_blocks: int,
    max_blocks: int,
    seed: int = 2026,
    gyel: str = "NP",
) -> List[Dict]:
    """Phase 1/2가 공유하는 가변 크기 W/O episode를 만든다.

    ``NP``는 발표자료 고정 산식을 사용하고, ``MIXED``는 실적 계열 조합과
    NP/FN/FL/NC 계열별 수식을 사용하는 공용 생성기로 명시적으로 분기한다.
    """

    if episode_count <= 0:
        print(f"[ERROR][report_formula_data_generator.build_report_formula_episode_jobs] cause=invalid_episode_count value={episode_count}")
        raise ValueError("episode_count must be positive")
    if min_blocks <= 0 or max_blocks < min_blocks:
        print(
            "[ERROR][report_formula_data_generator.build_report_formula_episode_jobs] "
            f"cause=invalid_block_range min_blocks={min_blocks} max_blocks={max_blocks}"
        )
        raise ValueError("invalid block range")
    normalized_gyel = str(gyel or "").strip().upper()
    if normalized_gyel == "MIXED":
        # 순환 import를 피하면서 NP와 다계열 생성기의 public episode 진입점만 공유한다.
        from Utils.data.multi_series_formula_data_generator import (
            build_multi_series_formula_episode_jobs,
        )

        return build_multi_series_formula_episode_jobs(
            episode_count=episode_count,
            min_physical_blocks=min_blocks,
            max_physical_blocks=max_blocks,
            seed=seed,
        )
    if normalized_gyel != "NP":
        print(
            "[ERROR][report_formula_data_generator.build_report_formula_episode_jobs] "
            f"cause=unsupported_episode_series gyel={gyel} supported=NP,MIXED"
        )
        raise RuntimeError(f"unsupported formula episode series: {gyel}")
    rng = np.random.default_rng(seed)
    episodes: List[Dict] = []
    for episode_index in range(episode_count):
        episode_id = f"PDF{episode_index + 1:05d}"
        block_count = int(rng.integers(min_blocks, max_blocks + 1))
        episode_seed = int(rng.integers(1, 2_147_483_647))
        generated = generate_report_formula_data(
            n_blocks=block_count,
            seed=episode_seed,
            gyel=normalized_gyel,
        )
        episodes.append(
            {
                "episode_id": episode_id,
                "problem_id": episode_id,
                "physical_block_count": block_count,
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
            plate_width=_positive_float(row["BTH"], "BTH", job_id),
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
                "plate_width": _positive_float(_job_field(job, "plate_width"), "plate_width", str(job_id)),
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
        ("wo_df", wo_df, ("LTH", "BTH", "THK", "TACT_TIME", "PTLST_QTY"), ("CUT_LTH", "MARK_LTH", "BVL_LTH", "STL_QTY", "BV_QTY")),
        ("block_df", block_df, ("LTH", "BTH", "THK", "TACT_TIME", "PTLST_QTY"), ("CUT_LTH", "MARK_LTH", "BVL_LTH", "STL_QTY", "BV_QTY")),
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
        stl_quantity = pd.to_numeric(frame["STL_QTY"], errors="coerce")
        if not np.allclose(stl_quantity, np.round(stl_quantity)):
            print(
                "[ERROR][report_formula_data_generator.validate_report_formula_data] "
                f"cause=non_integer_values frame={frame_name} column=STL_QTY"
            )
            raise RuntimeError(f"non-integer STL_QTY in {frame_name}")

    grouped = wo_df.groupby(["PROJ_NO", "GYEL", "BLK_NO"], sort=True)
    for _, block in block_df.iterrows():
        key = (block["PROJ_NO"], block["GYEL"], block["BLK_NO"])
        if key not in grouped.groups:
            print(f"[ERROR][report_formula_data_generator.validate_report_formula_data] cause=missing_block_wos key={key}")
            raise RuntimeError(f"missing generated W/O rows for block {key}")
        rows = grouped.get_group(key)
        _assert_close(float(block["LTH"]), float(rows["LTH"].max()), "LTH", key)
        _assert_close(float(block["BTH"]), float(rows["BTH"].max()), "BTH", key)
        _assert_close(float(block["THK"]), float(rows["THK"].max()), "THK", key)
        _assert_close(float(block["TACT_TIME"]), float(rows["TACT_TIME"].max()), "TACT_TIME", key)
        _assert_close(float(block["CUT_LTH"]), float(rows["CUT_LTH"].sum()), "CUT_LTH", key)
        _assert_close(float(block["MARK_LTH"]), float(rows["MARK_LTH"].sum()), "MARK_LTH", key)
        _assert_close(float(block["BVL_LTH"]), float(rows["BVL_LTH"].sum()), "BVL_LTH", key)
        _assert_close(float(block["BV_QTY"]), float(rows["BV_QTY"].sum()), "BV_QTY", key)
        _assert_close(float(block["PTLST_QTY"]), float(rows["PTLST_QTY"].sum()), "PTLST_QTY", key)
        expected_stl_quantity = int(rows["STL_QTY"].sum())
        if int(block["STL_QTY"]) != expected_stl_quantity:
            print(
                "[ERROR][report_formula_data_generator.validate_report_formula_data] "
                f"cause=stl_qty_mismatch key={key} block={block['STL_QTY']} wo_sum={expected_stl_quantity}"
            )
            raise RuntimeError(f"STL_QTY mismatch for block {key}")


def _generate_block_seed_values(
    rng: np.random.Generator,
    thickness_specs: Sequence[float],
    wo_count_override: int | None = None,
) -> Dict[str, float | int]:
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
    formula_wo_count = max(1, int(round((BLOCK_STEEL_A * block_cut + BLOCK_STEEL_B) * rng.gamma(BLOCK_STEEL_GAMMA_SHAPE, BLOCK_STEEL_GAMMA_SCALE))))
    wo_count = int(wo_count_override) if wo_count_override is not None else formula_wo_count
    if wo_count <= 0:
        print(
            "[ERROR][report_formula_data_generator._generate_block_seed_values] "
            f"cause=invalid_wo_count_override value={wo_count_override}"
        )
        raise RuntimeError("wo_count_override must be positive")
    block_thickness_raw = BLOCK_THICKNESS_A * np.log(wo_count) + BLOCK_THICKNESS_B + rng.normal(0.0, BLOCK_THICKNESS_STD)
    return {
        "LTH": block_length,
        "MARK_LTH": block_mark,
        "CUT_LTH": block_cut,
        "WO_QTY": wo_count,
        "THK": _nearest_spec(block_thickness_raw, thickness_specs),
    }


def _generate_work_order_rows(
    rng: np.random.Generator,
    project_no: str,
    block_no: str,
    gyel: str,
    block_length: float,
    block_thickness: float,
    block_mark_length: float,
    block_cut_length: float,
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

    # PDF 23쪽의 W/O 식으로 상대적인 W/O 크기를 만든 뒤, PDF 21쪽의
    # block 제약에 따라 합계가 block MARK/CUT seed와 정확히 같도록 맞춘다.
    raw_cut_lengths = _sample_linked_non_negative_regression_values(
        rng=rng,
        expected_values=WO_CUT_A * lengths + WO_CUT_B,
        residual_std=WO_CUT_STD,
        target_total=block_cut_length,
        field_name="CUT_LTH",
    )
    raw_mark_lengths = _sample_linked_non_negative_regression_values(
        rng=rng,
        expected_values=WO_MARK_A * lengths + WO_MARK_B,
        residual_std=WO_MARK_STD,
        target_total=block_mark_length,
        field_name="MARK_LTH",
    )
    cut_lengths = _scale_non_negative_values_to_total(
        raw_values=raw_cut_lengths,
        total=block_cut_length,
        field_name="CUT_LTH",
    )
    mark_lengths = _scale_non_negative_values_to_total(
        raw_values=raw_mark_lengths,
        total=block_mark_length,
        field_name="MARK_LTH",
    )

    rows: List[Dict] = []
    for wo_index, (length, thickness, cut_length, mark_length) in enumerate(
        zip(lengths, thicknesses, cut_lengths, mark_lengths),
        start=1,
    ):
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
                "CUT_LTH": float(cut_length),
                "MARK_LTH": float(mark_length),
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


def _sample_linked_non_negative_regression_values(
    rng: np.random.Generator,
    expected_values: Sequence[float],
    residual_std: float,
    target_total: float,
    field_name: str,
) -> np.ndarray:
    """PPT 정규오차 회귀식에서 block 합계와 연결 가능한 W/O 표본을 뽑는다."""

    expected = np.asarray(expected_values, dtype=float)
    target = float(target_total)
    sigma = float(residual_std)
    if expected.ndim != 1 or len(expected) == 0:
        print(
            "[ERROR][report_formula_data_generator._sample_linked_non_negative_regression_values] "
            f"cause=invalid_expected_shape field={field_name} shape={expected.shape}"
        )
        raise RuntimeError(f"invalid W/O regression input for {field_name}")
    if not np.isfinite(expected).all() or not np.isfinite(target) or not np.isfinite(sigma):
        print(
            "[ERROR][report_formula_data_generator._sample_linked_non_negative_regression_values] "
            f"cause=non_finite_input field={field_name} target={target_total} residual_std={residual_std}"
        )
        raise RuntimeError(f"non-finite W/O regression input for {field_name}")
    if target < 0 or sigma <= 0:
        print(
            "[ERROR][report_formula_data_generator._sample_linked_non_negative_regression_values] "
            f"cause=out_of_domain field={field_name} target={target} residual_std={sigma}"
        )
        raise RuntimeError(f"out-of-domain W/O regression input for {field_name}")
    if target == 0:
        return np.zeros(len(expected), dtype=float)

    for _ in range(MAX_REGRESSION_RESAMPLE_ATTEMPTS):
        sampled = np.maximum(0.0, expected + rng.normal(0.0, sigma, len(expected)))
        if float(sampled.sum()) > 0:
            return sampled

    print(
        "[ERROR][report_formula_data_generator._sample_linked_non_negative_regression_values] "
        f"cause=unable_to_sample_positive_formula_values field={field_name} "
        f"attempts={MAX_REGRESSION_RESAMPLE_ATTEMPTS}"
    )
    raise RuntimeError(f"unable to sample positive W/O formula values for {field_name}")


def _scale_non_negative_values_to_total(
    raw_values: Sequence[float],
    total: float,
    field_name: str,
) -> np.ndarray:
    """W/O 비율을 유지하면서 반올림 후 합계까지 block seed에 맞춘다."""

    raw = np.asarray(raw_values, dtype=float)
    target = float(total)
    if raw.ndim != 1 or len(raw) == 0:
        print(
            "[ERROR][report_formula_data_generator._scale_non_negative_values_to_total] "
            f"cause=invalid_shape field={field_name} raw_shape={raw.shape}"
        )
        raise RuntimeError(f"invalid W/O scaling input for {field_name}")
    if not np.isfinite(raw).all() or not np.isfinite(target):
        print(
            "[ERROR][report_formula_data_generator._scale_non_negative_values_to_total] "
            f"cause=non_finite_input field={field_name} total={total}"
        )
        raise RuntimeError(f"non-finite W/O scaling input for {field_name}")
    if (raw < 0).any() or target < 0:
        print(
            "[ERROR][report_formula_data_generator._scale_non_negative_values_to_total] "
            f"cause=out_of_domain field={field_name} total={target}"
        )
        raise RuntimeError(f"out-of-domain W/O scaling input for {field_name}")
    if target == 0:
        return np.zeros(len(raw), dtype=float)

    raw_total = float(raw.sum())
    if raw_total <= 0:
        print(
            "[ERROR][report_formula_data_generator._scale_non_negative_values_to_total] "
            f"cause=non_positive_formula_total field={field_name} total={target}"
        )
        raise RuntimeError(f"positive W/O formula total required for {field_name}")
    scaled = raw / raw_total * target
    rounded = np.round(scaled, 6)
    rounded_target = round(target, 6)
    residual = round(rounded_target - float(rounded.sum()), 6)
    if residual:
        adjust_index = int(np.argmax(rounded))
        rounded[adjust_index] = round(float(rounded[adjust_index]) + residual, 6)
    if (rounded < 0).any() or abs(float(rounded.sum()) - rounded_target) > 1e-6:
        print(
            "[ERROR][report_formula_data_generator._scale_non_negative_values_to_total] "
            f"cause=total_identity_failed field={field_name} expected={rounded_target} actual={rounded.sum()}"
        )
        raise RuntimeError(f"W/O total identity failed for {field_name}")
    return rounded


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


def _validate_optional_positive_integer_sequence(
    values: Sequence[int] | None,
    expected_length: int,
    field_name: str,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    raw_values = tuple(values)
    resolved = []
    invalid_value = None
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)):
            invalid_value = value
            break
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid_value = value
            break
        if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            invalid_value = value
            break
        resolved.append(int(numeric))
    if len(raw_values) != expected_length or invalid_value is not None:
        print(
            "[ERROR][report_formula_data_generator._validate_optional_positive_integer_sequence] "
            f"cause=invalid_sequence field={field_name} expected={expected_length} "
            f"actual={len(raw_values)} invalid_value={invalid_value} values={raw_values[:10]}"
        )
        raise RuntimeError(f"invalid {field_name}")
    return tuple(resolved)


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
