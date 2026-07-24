"""실적 계열 조합과 계열별 수식을 사용하는 공용 합성데이터 생성기."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from Utils.data import report_formula_data_generator as report_formula
from Utils.data.report_formula_data_generator import (
    BthFormulaProfile,
    StlQuantityProfile,
    DEFAULT_MULTI_SERIES_WO_SOURCE,
    fit_bth_formula,
    fit_stl_quantity_profile,
    generate_report_formula_block_seeds,
    jobs_from_report_formula_wo,
    generate_report_formula_data,
)


SUPPORTED_SERIES = frozenset({"NP", "FN", "FL", "NC"})
SERIES_BLOCK_FEATURES = ("LTH", "THK", "BTH", "CUT_LTH", "BV_QTY")
PHYSICAL_CONDITION_FEATURES = ("LTH", "THK", "CUT_LTH")
CONDITIONAL_NEIGHBOR_COUNT = 8
SERIES_RANDOM_STREAM_INDEX = {"FL": 1, "FN": 2, "NC": 3, "NP": 4}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MULTI_SERIES_BLOCK_SOURCE = REPO_ROOT / "변경사항" / "절단블록_데이터.xlsx"
MULTI_SERIES_GENERATION_PROFILE_SCHEMA = "multi_series_generation_profile_v1"
DEFAULT_MULTI_SERIES_GENERATION_PROFILE = Path(__file__).with_name(
    "multi_series_generation_profile.json"
)
NP_FIXED_FORMULA_CONSTANT_NAMES = (
    "BLOCK_LENGTH_MEAN",
    "BLOCK_LENGTH_STD",
    "BLOCK_LENGTH_MIN",
    "BLOCK_LENGTH_MAX",
    "BLOCK_MARK_A",
    "BLOCK_MARK_B",
    "BLOCK_MARK_ZERO_INFLATION",
    "BLOCK_MARK_GAMMA_SHAPE",
    "BLOCK_MARK_GAMMA_SCALE",
    "BLOCK_MARK_GAMMA_SHIFT",
    "BLOCK_CUT_A",
    "BLOCK_CUT_B",
    "BLOCK_CUT_GAMMA_SHAPE",
    "BLOCK_CUT_GAMMA_SCALE",
    "BLOCK_STEEL_A",
    "BLOCK_STEEL_B",
    "BLOCK_STEEL_GAMMA_SHAPE",
    "BLOCK_STEEL_GAMMA_SCALE",
    "BLOCK_THICKNESS_A",
    "BLOCK_THICKNESS_B",
    "BLOCK_THICKNESS_STD",
    "WO_LENGTH_A",
    "WO_LENGTH_B",
    "WO_LENGTH_POWER",
    "WO_THICKNESS_BASE",
    "WO_THICKNESS_AMP",
    "WO_THICKNESS_DECAY",
    "WO_THICKNESS_SKEW_SHAPE",
    "WO_THICKNESS_SKEW_LOC",
    "WO_THICKNESS_SKEW_SCALE",
    "WO_CUT_A",
    "WO_CUT_B",
    "WO_CUT_STD",
    "WO_MARK_A",
    "WO_MARK_B",
    "WO_MARK_STD",
    "WO_BEVEL_A_THK",
    "WO_BEVEL_A_LTH",
    "WO_BEVEL_A_MARK",
    "WO_BEVEL_B",
    "WO_BEVEL_STD",
    "WO_BVQ_A_BVL",
    "WO_BVQ_A_CUT",
    "WO_BVQ_B",
    "WO_BVQ_STD",
    "WO_PTLST_A_CUT",
    "WO_PTLST_A_LTH",
    "WO_PTLST_A_MARK",
    "WO_PTLST_B",
    "WO_PTLST_STD",
    "TACT_A_CUT",
    "TACT_A_MARK",
    "TACT_A_THK",
    "TACT_A_PTLST",
)
MULTI_SERIES_WO_COLUMNS = (
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
MULTI_SERIES_BLOCK_COLUMNS = (
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
    "WO_QTY",
    "BV_QTY",
    "TACT_TIME",
    "PTLST_QTY",
)


@dataclass(frozen=True)
class PhysicalBlockJointProfile:
    """물리 블록별 전체 W/O 수와 계열별 count vector의 실적 profile."""

    rows: pd.DataFrame
    physical_rows: pd.DataFrame
    physical_keys: tuple[tuple[str, str], ...]
    combinations: tuple[tuple[str, ...], ...]
    probabilities: tuple[float, ...]
    counts: tuple[int, ...]
    physical_block_count: int


@dataclass(frozen=True)
class MultiSeriesFormulaGeneration:
    """한 번에 생성한 다계열 W/O와 series-block 데이터."""

    wo_df: pd.DataFrame
    block_df: pd.DataFrame
    physical_block_count: int
    series_combinations: tuple[tuple[str, ...], ...]
    allocation_df: pd.DataFrame


@dataclass(frozen=True)
class MultiSeriesGenerationProfile:
    """학습 중 Excel 없이 사용하는 검증 완료 고정 profile."""

    physical: PhysicalBlockJointProfile
    np_bth: BthFormulaProfile
    np_stl: StlQuantityProfile
    empirical_generators: Mapping[str, object]
    sources: Mapping[str, Mapping[str, object]]


def _json_value(value: Any) -> Any:
    """numpy/pandas 값을 표준 JSON 값으로 변환한다."""

    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _require_profile_keys(mapping: Mapping[str, object], required: Sequence[str], context: str) -> None:
    missing = sorted(set(required) - set(mapping))
    if missing:
        print(
            "[ERROR][multi_series_formula_data_generator._require_profile_keys] "
            f"cause=missing_profile_keys context={context} keys={missing}"
        )
        raise RuntimeError(f"missing fixed profile keys: {context}: {missing}")


def _frame_payload(frame: pd.DataFrame) -> dict:
    return {
        "columns": list(frame.columns),
        "records": _json_value(frame.to_dict(orient="records")),
    }


def _frame_from_payload(payload: Mapping[str, object], context: str) -> pd.DataFrame:
    _require_profile_keys(payload, ("columns", "records"), context)
    columns = payload["columns"]
    records = payload["records"]
    if not isinstance(columns, list) or not isinstance(records, list) or len(set(columns)) != len(columns):
        print(
            "[ERROR][multi_series_formula_data_generator._frame_from_payload] "
            f"cause=invalid_frame_payload context={context}"
        )
        raise RuntimeError(f"invalid fixed profile frame: {context}")
    return pd.DataFrame.from_records(records, columns=columns)


def _serialize_physical_profile(profile: PhysicalBlockJointProfile) -> dict:
    return {
        "rows": _frame_payload(profile.rows),
        "physical_rows": _frame_payload(profile.physical_rows),
        "physical_keys": _json_value(profile.physical_keys),
        "combinations": _json_value(profile.combinations),
        "probabilities": list(profile.probabilities),
        "counts": list(profile.counts),
        "physical_block_count": profile.physical_block_count,
    }


def _deserialize_physical_profile(payload: Mapping[str, object]) -> PhysicalBlockJointProfile:
    _require_profile_keys(
        payload,
        ("rows", "physical_rows", "physical_keys", "combinations", "probabilities", "counts", "physical_block_count"),
        "physical_block_joint",
    )
    rows = _frame_from_payload(payload["rows"], "physical_block_joint.rows")
    physical_rows = _frame_from_payload(payload["physical_rows"], "physical_block_joint.physical_rows")
    if "SERIES_COMBINATION" in physical_rows:
        physical_rows["SERIES_COMBINATION"] = physical_rows["SERIES_COMBINATION"].map(tuple)
    profile = PhysicalBlockJointProfile(
        rows=rows,
        physical_rows=physical_rows,
        physical_keys=tuple(tuple(str(value) for value in key) for key in payload["physical_keys"]),
        combinations=tuple(tuple(str(value) for value in combination) for combination in payload["combinations"]),
        probabilities=tuple(float(value) for value in payload["probabilities"]),
        counts=tuple(int(value) for value in payload["counts"]),
        physical_block_count=int(payload["physical_block_count"]),
    )
    if (
        profile.physical_block_count <= 0
        or len(profile.physical_rows) != profile.physical_block_count
        or len(profile.combinations) != len(profile.probabilities)
        or len(profile.combinations) != len(profile.counts)
        or sum(profile.counts) != profile.physical_block_count
        or not np.isclose(sum(profile.probabilities), 1.0)
    ):
        print(
            "[ERROR][multi_series_formula_data_generator._deserialize_physical_profile] "
            "cause=invalid_physical_profile_counts"
        )
        raise RuntimeError("invalid fixed physical-block profile")
    return profile


def _serialize_bth_profile(profile: BthFormulaProfile) -> dict:
    return {
        "series": profile.series,
        "coefficients": list(profile.coefficients),
        "residual_std": profile.residual_std,
        "r_squared": profile.r_squared,
        "observed_specs": list(profile.observed_specs),
    }


def _deserialize_bth_profile(payload: Mapping[str, object]) -> BthFormulaProfile:
    _require_profile_keys(
        payload,
        ("series", "coefficients", "residual_std", "r_squared", "observed_specs"),
        "np.bth",
    )
    return BthFormulaProfile(
        series=str(payload["series"]).strip().upper(),
        coefficients=tuple(float(value) for value in payload["coefficients"]),
        residual_std=float(payload["residual_std"]),
        r_squared=float(payload["r_squared"]),
        observed_specs=tuple(float(value) for value in payload["observed_specs"]),
    )


def _serialize_stl_profile(profile: StlQuantityProfile) -> dict:
    return {
        "series": profile.series,
        "features": list(profile.features),
        "feature_mean": list(profile.feature_mean),
        "feature_scale": list(profile.feature_scale),
        "normalized_actual_features": _json_value(profile.normalized_actual_features),
        "actual_values": _json_value(profile.actual_values),
        "classes": list(profile.classes),
        "priors": list(profile.priors),
    }


def _deserialize_stl_profile(payload: Mapping[str, object]) -> StlQuantityProfile:
    _require_profile_keys(
        payload,
        ("series", "features", "feature_mean", "feature_scale", "normalized_actual_features", "actual_values", "classes", "priors"),
        "np.stl_quantity",
    )
    profile = StlQuantityProfile(
        series=str(payload["series"]).strip().upper(),
        features=tuple(str(value) for value in payload["features"]),
        feature_mean=tuple(float(value) for value in payload["feature_mean"]),
        feature_scale=tuple(float(value) for value in payload["feature_scale"]),
        normalized_actual_features=np.asarray(payload["normalized_actual_features"], dtype=float),
        actual_values=np.asarray(payload["actual_values"], dtype=int),
        classes=tuple(int(value) for value in payload["classes"]),
        priors=tuple(float(value) for value in payload["priors"]),
    )
    if (
        profile.normalized_actual_features.shape != (len(profile.actual_values), len(profile.features))
        or len(profile.feature_mean) != len(profile.features)
        or len(profile.feature_scale) != len(profile.features)
        or len(profile.classes) != len(profile.priors)
        or not np.isclose(sum(profile.priors), 1.0)
    ):
        print(
            "[ERROR][multi_series_formula_data_generator._deserialize_stl_profile] "
            "cause=invalid_np_stl_profile"
        )
        raise RuntimeError("invalid fixed NP STL_QTY profile")
    return profile


def _np_fixed_formula_contract() -> dict:
    """JSON profile과 코드의 발표자료 고정식이 같은지 확인할 계약을 만든다."""

    return {
        "version": "np_ppt_fixed_formula_v1",
        "constants": {
            name: _json_value(getattr(report_formula, name))
            for name in NP_FIXED_FORMULA_CONSTANT_NAMES
        },
        "thickness_specs": list(report_formula.DEFAULT_THICKNESS_SPECS),
        "bth_features": list(report_formula.BTH_FORMULA_FEATURES),
        "bth_random_stream_salt": report_formula.BTH_RANDOM_STREAM_SALT,
        "stl_random_stream_salt": report_formula.STL_RANDOM_STREAM_SALT,
        "stl_local_weight": report_formula.STL_LOCAL_WEIGHT,
    }


def _source_metadata(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        display_path = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(path.resolve())
    return {
        "path": display_path,
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _python_int_tuple(values: Iterable[object]) -> tuple[int, ...]:
    """uint32 seed를 Windows signed int로 축소하지 않고 Python int로 변환한다."""

    return tuple(int(value) for value in values)


def fit_physical_block_joint_profile(
    work_orders: pd.DataFrame,
    blocks: pd.DataFrame,
) -> PhysicalBlockJointProfile:
    """실적 물리 블록의 계열 조합과 계열별 핵심 특성 공동순위를 적합한다."""

    identity = ("PROJ_NO", "BLK_NO", "GYEL")
    missing_wo = [column for column in identity if column not in work_orders.columns]
    missing_block = [column for column in identity + SERIES_BLOCK_FEATURES if column not in blocks.columns]
    if missing_wo or missing_block:
        print(
            "[ERROR][multi_series_formula_data_generator.fit_physical_block_joint_profile] "
            f"cause=missing_columns wo={missing_wo} block={missing_block}"
        )
        raise RuntimeError(
            f"missing physical-block joint columns: wo={missing_wo}, block={missing_block}"
        )
    if work_orders.empty or blocks.empty:
        print(
            "[ERROR][multi_series_formula_data_generator.fit_physical_block_joint_profile] "
            f"cause=empty_source wo_rows={len(work_orders)} block_rows={len(blocks)}"
        )
        raise RuntimeError("physical-block joint source is empty")

    normalized_wo = work_orders[list(identity)].copy()
    normalized_blocks = blocks[list(identity + SERIES_BLOCK_FEATURES)].copy()
    for frame_name, frame in (("wo", normalized_wo), ("block", normalized_blocks)):
        for column in ("PROJ_NO", "BLK_NO"):
            frame[column] = frame[column].astype(str).str.strip()
            if frame[column].eq("").any():
                print(
                    "[ERROR][multi_series_formula_data_generator.fit_physical_block_joint_profile] "
                    f"cause=empty_identity source={frame_name} column={column}"
                )
                raise RuntimeError(f"empty physical block identity: {frame_name}.{column}")
        frame["GYEL"] = frame["GYEL"].astype(str).str.strip().str.upper()
        unknown = sorted(set(frame["GYEL"]) - SUPPORTED_SERIES)
        if unknown:
            print(
                "[ERROR][multi_series_formula_data_generator.fit_physical_block_joint_profile] "
                f"cause=unsupported_series source={frame_name} values={unknown}"
            )
            raise RuntimeError(f"unsupported series in joint source: {unknown}")

    duplicates = normalized_blocks.duplicated(list(identity), keep=False)
    if duplicates.any():
        examples = normalized_blocks.loc[duplicates, list(identity)].head(5).to_dict(orient="records")
        print(
            "[ERROR][multi_series_formula_data_generator.fit_physical_block_joint_profile] "
            f"cause=duplicate_series_block rows={int(duplicates.sum())} examples={examples}"
        )
        raise RuntimeError("duplicate series row in physical block")

    wo_counts = (
        normalized_wo.groupby(list(identity), sort=True)
        .size()
        .rename("SERIES_WO_QTY")
        .reset_index()
    )
    rows = normalized_blocks.merge(
        wo_counts,
        on=list(identity),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = rows["_merge"].ne("both")
    if unmatched.any():
        examples = rows.loc[unmatched, list(identity) + ["_merge"]].head(5).to_dict(orient="records")
        print(
            "[ERROR][multi_series_formula_data_generator.fit_physical_block_joint_profile] "
            f"cause=wo_block_identity_mismatch rows={int(unmatched.sum())} examples={examples}"
        )
        raise RuntimeError("W/O and block identities do not match for joint profile")
    rows = rows.drop(columns="_merge")
    rows[list(SERIES_BLOCK_FEATURES) + ["SERIES_WO_QTY"]] = rows[
        list(SERIES_BLOCK_FEATURES) + ["SERIES_WO_QTY"]
    ].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = rows[list(SERIES_BLOCK_FEATURES) + ["SERIES_WO_QTY"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (rows[["SERIES_WO_QTY", "LTH", "THK", "BTH"]] <= 0).any(axis=None):
        print(
            "[ERROR][multi_series_formula_data_generator.fit_physical_block_joint_profile] "
            "cause=invalid_joint_features"
        )
        raise RuntimeError("invalid physical-block joint features")
    if (rows[["CUT_LTH", "BV_QTY"]] < 0).any(axis=None):
        print(
            "[ERROR][multi_series_formula_data_generator.fit_physical_block_joint_profile] "
            "cause=negative_joint_features"
        )
        raise RuntimeError("negative physical-block joint features")

    rows = rows.sort_values(list(identity), kind="stable").reset_index(drop=True)
    physical_grouped = rows.groupby(["PROJ_NO", "BLK_NO"], sort=True)
    physical_rows = physical_grouped.agg(
        TOTAL_WO_QTY=("SERIES_WO_QTY", "sum"),
        LTH=("LTH", "max"),
        THK=("THK", "max"),
        BTH=("BTH", "max"),
        CUT_LTH=("CUT_LTH", "sum"),
        BV_QTY=("BV_QTY", "sum"),
    ).reset_index()
    physical_combinations = physical_grouped["GYEL"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    physical_rows["SERIES_COMBINATION"] = [
        physical_combinations.loc[(project, block)]
        for project, block in zip(physical_rows["PROJ_NO"], physical_rows["BLK_NO"])
    ]
    for feature in PHYSICAL_CONDITION_FEATURES:
        physical_rows[f"{feature}__rank"] = physical_rows[feature].rank(
            method="average", pct=True
        )
    physical_rows["TOTAL_WO_QTY"] = physical_rows["TOTAL_WO_QTY"].astype(int)
    physical_rows["TOTAL_WO_QTY__rank"] = physical_rows["TOTAL_WO_QTY"].rank(
        method="average", pct=True
    )
    counts = physical_combinations.value_counts(sort=False).sort_index()
    total = int(counts.sum())
    probabilities = counts.to_numpy(dtype=float) / total
    return PhysicalBlockJointProfile(
        rows=rows,
        physical_rows=physical_rows,
        physical_keys=tuple((str(project), str(block)) for project, block in physical_combinations.index),
        combinations=tuple(counts.index.tolist()),
        probabilities=tuple(float(value) for value in probabilities),
        counts=tuple(int(value) for value in counts.to_numpy(dtype=int)),
        physical_block_count=total,
    )


def select_physical_block_series_allocations(
    profile: PhysicalBlockJointProfile,
    block_seeds: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """전체 W/O 수와 물리 특성에 조건부인 실적 계열 count vector를 선택한다."""

    required = (
        "physical_index",
        "LTH",
        "THK",
        "CUT_LTH",
        "WO_QTY",
        "PHYSICAL_RANDOM_SEED",
    )
    missing = [column for column in required if column not in block_seeds.columns]
    if missing:
        print(
            "[ERROR][multi_series_formula_data_generator.select_physical_block_series_allocations] "
            f"cause=missing_block_seed_columns columns={missing}"
        )
        raise RuntimeError(f"missing physical block seed columns: {missing}")
    if block_seeds.empty or block_seeds["physical_index"].duplicated().any():
        print(
            "[ERROR][multi_series_formula_data_generator.select_physical_block_series_allocations] "
            "cause=empty_or_duplicate_block_seeds"
        )
        raise RuntimeError("physical block seeds must be non-empty and unique")
    numeric = block_seeds[
        ["physical_index", "LTH", "THK", "CUT_LTH", "WO_QTY", "PHYSICAL_RANDOM_SEED"]
    ].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (
        numeric[["physical_index", "LTH", "THK", "WO_QTY", "PHYSICAL_RANDOM_SEED"]] <= 0
    ).any(axis=None) or (numeric["CUT_LTH"] < 0).any():
        print(
            "[ERROR][multi_series_formula_data_generator.select_physical_block_series_allocations] "
            "cause=invalid_block_seed_values"
        )
        raise RuntimeError("invalid physical block seed values")
    integer_seed_values = numeric[
        ["physical_index", "WO_QTY", "PHYSICAL_RANDOM_SEED"]
    ].to_numpy(dtype=float)
    if not np.equal(integer_seed_values, np.floor(integer_seed_values)).all():
        print(
            "[ERROR][multi_series_formula_data_generator.select_physical_block_series_allocations] "
            "cause=non_integer_block_seed_values"
        )
        raise RuntimeError("invalid physical block seed values")
    if profile.physical_rows.empty or profile.rows.empty:
        print(
            "[ERROR][multi_series_formula_data_generator.select_physical_block_series_allocations] "
            "cause=empty_profile"
        )
        raise RuntimeError("physical block joint profile is empty")

    observed_values = {
        feature: np.sort(profile.physical_rows[feature].to_numpy(dtype=float))
        for feature in PHYSICAL_CONDITION_FEATURES
    }
    observed_total_values = np.sort(
        profile.physical_rows["TOTAL_WO_QTY"].to_numpy(dtype=float)
    )
    observed_series_max = profile.rows.groupby("GYEL", sort=True)["SERIES_WO_QTY"].max()
    allocation_rows = []
    ordered_seeds = block_seeds.sort_values("physical_index", kind="stable")
    for seed_row in ordered_seeds.itertuples(index=False):
        formula_count = int(seed_row.WO_QTY)
        feasible_indices = [
            index
            for index, combination in enumerate(profile.combinations)
            if len(combination) <= formula_count
            and (
                "NP" in combination
                or sum(int(observed_series_max.loc[series]) for series in combination)
                >= formula_count
            )
        ]
        if not feasible_indices:
            print(
                "[ERROR][multi_series_formula_data_generator.select_physical_block_series_allocations] "
                f"cause=no_feasible_series_combination total_wo_qty={formula_count}"
            )
            raise RuntimeError("no feasible series combination for generated total W/O count")
        feasible_probabilities = np.asarray(
            [profile.probabilities[index] for index in feasible_indices], dtype=float
        )
        feasible_probabilities /= feasible_probabilities.sum()
        selected_combination = profile.combinations[
            feasible_indices[int(rng.choice(len(feasible_indices), p=feasible_probabilities))]
        ]
        candidates = profile.physical_rows.loc[
            profile.physical_rows["SERIES_COMBINATION"].map(
                lambda combination: combination == selected_combination
            )
        ].reset_index(drop=True)
        if candidates.empty:
            print(
                "[ERROR][multi_series_formula_data_generator.select_physical_block_series_allocations] "
                f"cause=no_profile_for_sampled_combination combination={selected_combination}"
            )
            raise RuntimeError("no physical profile for sampled series combination")

        physical_ranks = np.array(
            [
                np.searchsorted(observed_values[feature], float(getattr(seed_row, feature)), side="right")
                / len(observed_values[feature])
                for feature in PHYSICAL_CONDITION_FEATURES
            ],
            dtype=float,
        )
        total_rank = np.searchsorted(
            observed_total_values, formula_count, side="right"
        ) / len(observed_total_values)
        generated_ranks = np.append(physical_ranks, total_rank)
        candidate_ranks = candidates[
            [
                *(f"{feature}__rank" for feature in PHYSICAL_CONDITION_FEATURES),
                "TOTAL_WO_QTY__rank",
            ]
        ].to_numpy(dtype=float)
        distances = np.square(candidate_ranks - generated_ranks).mean(axis=1)
        neighbor_count = min(CONDITIONAL_NEIGHBOR_COUNT, len(candidates))
        nearest = np.argsort(distances, kind="stable")[:neighbor_count]
        nearest_distances = distances[nearest]
        zero_distance = np.isclose(nearest_distances, 0.0)
        if zero_distance.any():
            probabilities = zero_distance.astype(float) / int(zero_distance.sum())
        else:
            inverse = 1.0 / nearest_distances
            probabilities = inverse / inverse.sum()
        selected_index = int(rng.choice(nearest, p=probabilities))
        selected = candidates.iloc[selected_index]
        profile_rows = profile.rows.loc[
            profile.rows["PROJ_NO"].eq(selected["PROJ_NO"])
            & profile.rows["BLK_NO"].eq(selected["BLK_NO"])
        ].sort_values("GYEL", kind="stable")
        profile_total = int(profile_rows["SERIES_WO_QTY"].sum())
        if profile_rows.empty or profile_total != int(selected["TOTAL_WO_QTY"]):
            print(
                "[ERROR][multi_series_formula_data_generator.select_physical_block_series_allocations] "
                f"cause=invalid_profile_count_vector project={selected['PROJ_NO']} "
                f"block={selected['BLK_NO']} expected={selected['TOTAL_WO_QTY']}"
            )
            raise RuntimeError("invalid physical profile count vector")
        allocated_counts = _apportion_empirical_count_vector(
            profile_rows["SERIES_WO_QTY"].to_numpy(dtype=int),
            formula_count,
            np.asarray(
                [
                    formula_count
                    if str(series) == "NP"
                    else int(observed_series_max.loc[str(series)])
                    for series in profile_rows["GYEL"]
                ],
                dtype=int,
            ),
        )

        physical_index = int(seed_row.physical_index)
        project_no = f"SYNTH_MULTI_PROJ_{(physical_index - 1) // 1000 + 1}"
        block_no = f"SYNTH_MULTI_BLK_{physical_index:06d}"
        for profile_row, allocated_count in zip(
            profile_rows.itertuples(index=False), allocated_counts
        ):
            physical_random_seed = int(seed_row.PHYSICAL_RANDOM_SEED)
            series_random_seed = int(
                np.random.SeedSequence(
                    [physical_random_seed, SERIES_RANDOM_STREAM_INDEX[str(profile_row.GYEL)]]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            allocation_rows.append(
                {
                    "physical_index": physical_index,
                    "PROJ_NO": project_no,
                    "BLK_NO": block_no,
                    "FORMULA_WO_QTY": formula_count,
                    "TOTAL_WO_QTY": formula_count,
                    "GYEL": str(profile_row.GYEL),
                    "SERIES_WO_QTY": int(allocated_count),
                    "PROFILE_PROJ_NO": str(selected["PROJ_NO"]),
                    "PROFILE_BLK_NO": str(selected["BLK_NO"]),
                    "PROFILE_TOTAL_WO_QTY": profile_total,
                    "PHYSICAL_RANDOM_SEED": physical_random_seed,
                    "SERIES_RANDOM_SEED": series_random_seed,
                }
            )
    allocations = pd.DataFrame(allocation_rows)
    if allocations.empty:
        print(
            "[ERROR][multi_series_formula_data_generator.select_physical_block_series_allocations] "
            "cause=no_allocations"
        )
        raise RuntimeError("no physical block series allocations")
    return allocations.sort_values(["physical_index", "GYEL"], kind="stable").reset_index(drop=True)


def _apportion_empirical_count_vector(
    empirical_counts: np.ndarray,
    target_total: int,
    maximum_counts: np.ndarray,
) -> np.ndarray:
    """실적 count vector 비율을 유지하면서 양의 정수 합을 목표 총수에 맞춘다."""

    counts = np.asarray(empirical_counts, dtype=int)
    maxima = np.asarray(maximum_counts, dtype=int)
    if (
        counts.ndim != 1
        or len(counts) == 0
        or maxima.shape != counts.shape
        or (counts <= 0).any()
        or (maxima <= 0).any()
        or target_total < len(counts)
        or target_total > int(maxima.sum())
    ):
        print(
            "[ERROR][multi_series_formula_data_generator._apportion_empirical_count_vector] "
            f"cause=invalid_count_vector counts={counts.tolist()} maxima={maxima.tolist()} "
            f"target_total={target_total}"
        )
        raise RuntimeError("invalid empirical series count vector")
    ideal = counts.astype(float) * (float(target_total) / float(counts.sum()))
    allocated = np.clip(np.floor(ideal).astype(int), 1, maxima)
    while int(allocated.sum()) > target_total:
        reducible = np.flatnonzero(allocated > 1)
        if len(reducible) == 0:
            print(
                "[ERROR][multi_series_formula_data_generator._apportion_empirical_count_vector] "
                f"cause=cannot_reduce counts={counts.tolist()} target_total={target_total}"
            )
            raise RuntimeError("cannot reduce empirical series count allocation")
        decrement_cost = -2.0 * (allocated[reducible] - ideal[reducible]) + 1.0
        allocated[int(reducible[np.argmin(decrement_cost)])] -= 1
    while int(allocated.sum()) < target_total:
        expandable = np.flatnonzero(allocated < maxima)
        if len(expandable) == 0:
            print(
                "[ERROR][multi_series_formula_data_generator._apportion_empirical_count_vector] "
                f"cause=cannot_expand counts={counts.tolist()} maxima={maxima.tolist()} "
                f"target_total={target_total}"
            )
            raise RuntimeError("cannot expand empirical series count allocation")
        increment_cost = 2.0 * (allocated[expandable] - ideal[expandable]) + 1.0
        allocated[int(expandable[np.argmin(increment_cost)])] += 1
    if (
        (allocated <= 0).any()
        or (allocated > maxima).any()
        or int(allocated.sum()) != target_total
    ):
        print(
            "[ERROR][multi_series_formula_data_generator._apportion_empirical_count_vector] "
            f"cause=invalid_result allocated={allocated.tolist()} target_total={target_total}"
        )
        raise RuntimeError("invalid empirical series count allocation")
    return allocated


def build_multi_series_generation_profile(
    wo_source_path: str | Path = DEFAULT_MULTI_SERIES_WO_SOURCE,
    block_source_path: str | Path = DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
    fl_mark_method: str = "chain",
) -> dict:
    """실적 Excel을 한 번 적합해 재현 가능한 고정 JSON payload를 만든다.

    fl_mark_method: FL 계열 MARK_LTH 생성 방법('chain' 또는 'dirichlet').
        다른 계열엔 영향이 없다. 자세한 내용은 shipyard_data_generator.FL_MARK_METHODS.
    """

    wo_path = Path(wo_source_path).resolve()
    block_path = Path(block_source_path).resolve()
    missing = [path for path in (wo_path, block_path) if not path.is_file()]
    if missing:
        print(
            "[ERROR][multi_series_formula_data_generator.build_multi_series_generation_profile] "
            f"cause=source_not_found paths={missing}"
        )
        raise FileNotFoundError(missing[0])
    work_orders = pd.read_excel(wo_path, sheet_name="Sheet1")
    blocks = pd.read_excel(block_path, sheet_name="Sheet1")
    physical = fit_physical_block_joint_profile(work_orders, blocks)
    np_bth = fit_bth_formula(work_orders, series="NP")
    np_stl = fit_stl_quantity_profile(work_orders, series="NP")

    from 데이터분석.shipyard_data_generator import ShipyardGenerator

    empirical = {}
    for series in sorted(SUPPORTED_SERIES - {"NP"}):
        # fl_mark_method는 FL에만 영향을 준다(다른 계열은 무시).
        empirical[series] = ShipyardGenerator(
            wo_path,
            block_path,
            mode="pearson",
            series=series,
            fl_mark_method=fl_mark_method,
        ).fit().to_generation_profile()
    payload = {
        "schema": MULTI_SERIES_GENERATION_PROFILE_SCHEMA,
        "sources": {
            "work_orders": _source_metadata(wo_path),
            "blocks": _source_metadata(block_path),
        },
        "np_fixed_formula": _np_fixed_formula_contract(),
        "physical_block_joint": _serialize_physical_profile(physical),
        "np": {
            "bth": _serialize_bth_profile(np_bth),
            "stl_quantity": _serialize_stl_profile(np_stl),
        },
        "empirical_series": empirical,
    }
    print(
        "[VALIDATION][multi_series_formula_data_generator.build_multi_series_generation_profile] "
        f"passed=true physical_blocks={physical.physical_block_count} "
        f"series={','.join(sorted(SUPPORTED_SERIES))}"
    )
    return payload


def write_multi_series_generation_profile(
    output_path: str | Path = DEFAULT_MULTI_SERIES_GENERATION_PROFILE,
    wo_source_path: str | Path = DEFAULT_MULTI_SERIES_WO_SOURCE,
    block_source_path: str | Path = DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
    fl_mark_method: str = "chain",
) -> Path:
    """오프라인 적합 결과를 표준 JSON으로 원자적으로 저장한다.

    fl_mark_method: FL 계열 MARK_LTH 생성 방법('chain' 또는 'dirichlet').
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_multi_series_generation_profile(
        wo_source_path, block_source_path, fl_mark_method=fl_mark_method
    )
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)
    _load_multi_series_generation_profile.cache_clear()
    load_multi_series_generation_profile(path)
    print(
        "[CHECK][multi_series_formula_data_generator.write_multi_series_generation_profile] "
        f"output={path} bytes={path.stat().st_size}"
    )
    return path


def load_multi_series_generation_profile(
    profile_path: str | Path = DEFAULT_MULTI_SERIES_GENERATION_PROFILE,
) -> MultiSeriesGenerationProfile:
    """고정 profile을 읽는다. 누락·손상 시 Excel 재적합 없이 실패한다."""

    return _load_multi_series_generation_profile(str(Path(profile_path).resolve()))


@lru_cache(maxsize=4)
def _load_multi_series_generation_profile(profile_path: str) -> MultiSeriesGenerationProfile:
    path = Path(profile_path)
    if not path.is_file():
        print(
            "[ERROR][multi_series_formula_data_generator._load_multi_series_generation_profile] "
            f"cause=profile_not_found path={path}"
        )
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(
            "[ERROR][multi_series_formula_data_generator._load_multi_series_generation_profile] "
            f"cause=invalid_json path={path} error={exc}"
        )
        raise RuntimeError(f"invalid multi-series generation profile: {path}") from exc
    if not isinstance(payload, dict):
        print(
            "[ERROR][multi_series_formula_data_generator._load_multi_series_generation_profile] "
            f"cause=invalid_root_type path={path} type={type(payload).__name__}"
        )
        raise RuntimeError("multi-series generation profile root must be an object")
    _require_profile_keys(
        payload,
        ("schema", "sources", "np_fixed_formula", "physical_block_joint", "np", "empirical_series"),
        "root",
    )
    if payload["schema"] != MULTI_SERIES_GENERATION_PROFILE_SCHEMA:
        print(
            "[ERROR][multi_series_formula_data_generator._load_multi_series_generation_profile] "
            f"cause=schema_mismatch actual={payload['schema']} "
            f"expected={MULTI_SERIES_GENERATION_PROFILE_SCHEMA}"
        )
        raise RuntimeError("multi-series generation profile schema mismatch")
    if payload["np_fixed_formula"] != _np_fixed_formula_contract():
        print(
            "[ERROR][multi_series_formula_data_generator._load_multi_series_generation_profile] "
            "cause=np_fixed_formula_contract_mismatch"
        )
        raise RuntimeError("NP fixed formula contract does not match the profile")

    sources = payload["sources"]
    _require_profile_keys(sources, ("work_orders", "blocks"), "sources")
    for name in ("work_orders", "blocks"):
        metadata = sources[name]
        _require_profile_keys(metadata, ("path", "sha256", "size_bytes"), f"sources.{name}")
        digest = str(metadata["sha256"])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            print(
                "[ERROR][multi_series_formula_data_generator._load_multi_series_generation_profile] "
                f"cause=invalid_source_sha256 source={name} value={digest}"
            )
            raise RuntimeError(f"invalid source SHA256 in fixed profile: {name}")

    np_payload = payload["np"]
    _require_profile_keys(np_payload, ("bth", "stl_quantity"), "np")
    np_bth = _deserialize_bth_profile(np_payload["bth"])
    np_stl = _deserialize_stl_profile(np_payload["stl_quantity"])
    if np_bth.series != "NP" or np_stl.series != "NP":
        print(
            "[ERROR][multi_series_formula_data_generator._load_multi_series_generation_profile] "
            f"cause=np_profile_series_mismatch bth={np_bth.series} stl={np_stl.series}"
        )
        raise RuntimeError("fixed NP profile has the wrong series")

    empirical_payload = payload["empirical_series"]
    expected_empirical = SUPPORTED_SERIES - {"NP"}
    if set(empirical_payload) != expected_empirical:
        print(
            "[ERROR][multi_series_formula_data_generator._load_multi_series_generation_profile] "
            f"cause=empirical_series_mismatch actual={sorted(empirical_payload)} "
            f"expected={sorted(expected_empirical)}"
        )
        raise RuntimeError("fixed profile empirical series mismatch")
    from 데이터분석.shipyard_data_generator import ShipyardGenerator

    empirical_generators = {
        series: ShipyardGenerator.from_generation_profile(empirical_payload[series])
        for series in sorted(expected_empirical)
    }
    profile = MultiSeriesGenerationProfile(
        physical=_deserialize_physical_profile(payload["physical_block_joint"]),
        np_bth=np_bth,
        np_stl=np_stl,
        empirical_generators=empirical_generators,
        sources=sources,
    )
    print(
        "[CHECK][multi_series_formula_data_generator._load_multi_series_generation_profile] "
        f"path={path} physical_blocks={profile.physical.physical_block_count} "
        f"series={','.join(sorted(SUPPORTED_SERIES))}"
    )
    return profile


def generate_multi_series_formula_data(
    n_physical_blocks: int,
    seed: int = 2026,
    profile_path: str | Path = DEFAULT_MULTI_SERIES_GENERATION_PROFILE,
) -> MultiSeriesFormulaGeneration:
    """전체 W/O 수를 먼저 생성한 뒤 계열별 count vector와 W/O를 생성한다."""

    if n_physical_blocks <= 0:
        print(
            "[ERROR][multi_series_formula_data_generator.generate_multi_series_formula_data] "
            f"cause=invalid_n_physical_blocks value={n_physical_blocks}"
        )
        raise ValueError("n_physical_blocks must be positive")
    generation_profile = load_multi_series_generation_profile(profile_path)
    profile = generation_profile.physical
    seed_sequence = np.random.SeedSequence(int(seed))
    block_seed_child, allocation_child, physical_child, *series_children = seed_sequence.spawn(
        3 + len(SUPPORTED_SERIES)
    )
    block_seed = int(block_seed_child.generate_state(1, dtype=np.uint32)[0])
    block_seeds = generate_report_formula_block_seeds(n_physical_blocks, block_seed)
    physical_rng = np.random.default_rng(physical_child)
    block_seeds["PHYSICAL_RANDOM_SEED"] = physical_rng.integers(
        1,
        np.iinfo(np.int32).max,
        size=n_physical_blocks,
        dtype=np.int64,
    )
    allocations = select_physical_block_series_allocations(
        profile,
        block_seeds,
        np.random.default_rng(allocation_child),
    )
    combinations = tuple(
        allocations.groupby("physical_index", sort=True)["GYEL"].agg(
            lambda values: tuple(sorted(set(values)))
        )
    )
    series_seed_map = {
        series: int(child.generate_state(1, dtype=np.uint32)[0])
        for series, child in zip(sorted(SUPPORTED_SERIES), series_children)
    }

    wo_frames = []
    for series in sorted(SUPPORTED_SERIES):
        series_allocations = allocations.loc[allocations["GYEL"].eq(series)].sort_values(
            "physical_index", kind="stable"
        )
        if series_allocations.empty:
            continue
        raw_wos = _generate_one_series(
            series=series,
            block_count=len(series_allocations),
            seed=series_seed_map[series],
            generation_profile=generation_profile,
            wo_counts=tuple(series_allocations["SERIES_WO_QTY"].astype(int)),
            block_seeds=_python_int_tuple(series_allocations["SERIES_RANDOM_SEED"]),
        )
        wo_frames.append(
            _assign_physical_block_identity_in_order(
                raw_wos,
                series,
                tuple(series_allocations["physical_index"].astype(int)),
            )
        )

    if not wo_frames:
        print(
            "[ERROR][multi_series_formula_data_generator.generate_multi_series_formula_data] "
            "cause=no_generated_series_rows"
        )
        raise RuntimeError("multi-series generation produced no W/O rows")
    wo_df = pd.concat(wo_frames, ignore_index=True)[list(MULTI_SERIES_WO_COLUMNS)]
    wo_df = wo_df.sort_values(
        ["PROJ_NO", "BLK_NO", "GYEL", "WK_ORD_NO"], kind="stable"
    ).reset_index(drop=True)
    block_df = _aggregate_multi_series_blocks(wo_df)
    validate_multi_series_formula_data(
        wo_df,
        block_df,
        n_physical_blocks,
        expected_series_combinations=combinations,
        expected_allocations=allocations,
    )
    print(
        "[VALIDATION][multi_series_formula_data_generator.generate_multi_series_formula_data] "
        f"passed=true physical_blocks={n_physical_blocks} series_blocks={len(block_df)} "
        f"wos={len(wo_df)} seed={seed}"
    )
    return MultiSeriesFormulaGeneration(
        wo_df=wo_df,
        block_df=block_df,
        physical_block_count=n_physical_blocks,
        series_combinations=combinations,
        allocation_df=allocations,
    )


def build_multi_series_formula_episode_jobs(
    episode_count: int,
    min_physical_blocks: int,
    max_physical_blocks: int,
    seed: int = 2026,
    profile_path: str | Path = DEFAULT_MULTI_SERIES_GENERATION_PROFILE,
) -> list[Dict]:
    """Phase 1/2가 공유하는 가변 크기 다계열 episode를 생성한다."""

    if episode_count <= 0 or min_physical_blocks <= 0 or max_physical_blocks < min_physical_blocks:
        print(
            "[ERROR][multi_series_formula_data_generator.build_multi_series_formula_episode_jobs] "
            f"cause=invalid_args episodes={episode_count} min={min_physical_blocks} max={max_physical_blocks}"
        )
        raise ValueError("invalid multi-series episode arguments")
    rng = np.random.default_rng(seed)
    episodes = []
    for episode_index in range(episode_count):
        episode_id = f"MULTI{episode_index + 1:05d}"
        physical_count = int(rng.integers(min_physical_blocks, max_physical_blocks + 1))
        episode_seed = int(rng.integers(1, 2_147_483_647))
        generated = generate_multi_series_formula_data(
            n_physical_blocks=physical_count,
            seed=episode_seed,
            profile_path=profile_path,
        )
        episodes.append(
            {
                "episode_id": episode_id,
                "problem_id": episode_id,
                "physical_block_count": physical_count,
                "block_count": len(generated.block_df),
                "job_count": len(generated.wo_df),
                "seed": episode_seed,
                "wo_df": generated.wo_df,
                "block_df": generated.block_df,
                "jobs": jobs_from_report_formula_wo(generated.wo_df, episode_id),
                "series_combinations": generated.series_combinations,
                "allocation_df": generated.allocation_df,
            }
        )
    return episodes


def load_physical_block_joint_profile(
    profile_path: str | Path = DEFAULT_MULTI_SERIES_GENERATION_PROFILE,
) -> PhysicalBlockJointProfile:
    """고정 JSON에서 실적 물리 블록 공동분포를 반환한다."""

    return load_multi_series_generation_profile(profile_path).physical


def _generate_one_series(
    series: str,
    block_count: int,
    seed: int,
    generation_profile: MultiSeriesGenerationProfile,
    wo_counts: Sequence[int],
    block_seeds: Sequence[int],
) -> pd.DataFrame:
    if series == "NP":
        return generate_report_formula_data(
            n_blocks=block_count,
            seed=seed,
            gyel="NP",
            bth_profile=generation_profile.np_bth,
            stl_quantity_profile=generation_profile.np_stl,
            wo_counts=wo_counts,
            block_seeds=block_seeds,
        ).wo_df.copy()
    if series not in generation_profile.empirical_generators:
        print(
            "[ERROR][multi_series_formula_data_generator._generate_one_series] "
            f"cause=missing_empirical_generator series={series}"
        )
        raise RuntimeError(f"missing fixed empirical generator for {series}")
    generator = generation_profile.empirical_generators[series]
    work_orders, _ = generator.generate(
        n_blocks=block_count,
        seed=seed,
        wo_counts=wo_counts,
        block_seeds=block_seeds,
    )
    return work_orders.copy()


def _source_block_ids(work_orders: pd.DataFrame) -> pd.Series:
    """계열별 생성기가 만든 원본 block identity를 안정적인 문자열로 변환한다."""

    if "BLK_ID" in work_orders.columns:
        return work_orders["BLK_ID"].astype(str)
    required = ("PROJ_NO", "BLK_NO")
    missing = [column for column in required if column not in work_orders.columns]
    if missing:
        print(
            "[ERROR][multi_series_formula_data_generator._source_block_ids] "
            f"cause=missing_source_identity columns={missing}"
        )
        raise RuntimeError(f"missing generated source identity: {missing}")
    return work_orders["PROJ_NO"].astype(str) + "\x1f" + work_orders["BLK_NO"].astype(str)


def _assign_physical_block_identity_in_order(
    work_orders: pd.DataFrame,
    series: str,
    physical_indices: Sequence[int],
) -> pd.DataFrame:
    frame = work_orders.copy()
    source_block_ids = _source_block_ids(frame)
    unique_ids = tuple(dict.fromkeys(source_block_ids.tolist()))
    resolved_indices = tuple(int(index) for index in physical_indices)
    if (
        len(unique_ids) != len(resolved_indices)
        or len(set(resolved_indices)) != len(resolved_indices)
        or any(index <= 0 for index in resolved_indices)
    ):
        print(
            "[ERROR][multi_series_formula_data_generator._assign_physical_block_identity_in_order] "
            f"cause=identity_count_mismatch series={series} generated={len(unique_ids)} "
            f"physical_indices={len(resolved_indices)}"
        )
        raise RuntimeError(f"generated source identity count mismatch for {series}")
    source_to_physical = dict(zip(unique_ids, resolved_indices))
    mapped = source_block_ids.map(source_to_physical)
    if mapped.isna().any():
        print(
            "[ERROR][multi_series_formula_data_generator._assign_physical_block_identity_in_order] "
            f"cause=unmapped_source series={series} rows={int(mapped.isna().sum())}"
        )
        raise RuntimeError(f"unmapped generated source block for {series}")
    mapped = mapped.astype(int)
    frame["PROJ_NO"] = mapped.map(lambda index: f"SYNTH_MULTI_PROJ_{(int(index) - 1) // 1000 + 1}")
    frame["BLK_NO"] = mapped.map(lambda index: f"SYNTH_MULTI_BLK_{int(index):06d}")
    frame["GYEL"] = series
    frame["WK_ORD_NO"] = [
        f"SYNTH_MULTI_WO_{int(block_index):06d}_{series}_{local_index:04d}"
        for block_index, local_index in zip(
            mapped,
            frame.groupby(mapped, sort=False).cumcount() + 1,
        )
    ]
    missing_features = [column for column in MULTI_SERIES_WO_COLUMNS if column not in frame.columns]
    if missing_features:
        print(
            "[ERROR][multi_series_formula_data_generator._assign_physical_block_identity_in_order] "
            f"cause=missing_generated_features series={series} columns={missing_features}"
        )
        raise RuntimeError(f"missing generated W/O features for {series}: {missing_features}")
    return frame


def _aggregate_multi_series_blocks(work_orders: pd.DataFrame) -> pd.DataFrame:
    grouped = work_orders.groupby(["PROJ_NO", "BLK_NO", "GYEL"], sort=True)
    blocks = grouped.agg(
        LTH=("LTH", "max"),
        BTH=("BTH", "max"),
        THK=("THK", "max"),
        CUT_LTH=("CUT_LTH", "sum"),
        MARK_LTH=("MARK_LTH", "max"),
        BVL_LTH=("BVL_LTH", "sum"),
        STL_QTY=("STL_QTY", "sum"),
        WO_QTY=("WK_ORD_NO", "size"),
        BV_QTY=("BV_QTY", "sum"),
        TACT_TIME=("TACT_TIME", "max"),
        PTLST_QTY=("PTLST_QTY", "sum"),
    ).reset_index()
    return blocks[list(MULTI_SERIES_BLOCK_COLUMNS)]


def validate_multi_series_formula_data(
    work_orders: pd.DataFrame,
    blocks: pd.DataFrame,
    expected_physical_blocks: int,
    expected_series_combinations: Sequence[tuple[str, ...]],
    expected_allocations: pd.DataFrame | None = None,
) -> None:
    """다계열 생성값의 identity, 집계, 범위 계약을 검사한다."""

    if work_orders.empty or blocks.empty:
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            "cause=empty_generated_data"
        )
        raise RuntimeError("empty multi-series generated data")
    if work_orders.isna().any(axis=None) or blocks.isna().any(axis=None):
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            "cause=missing_values"
        )
        raise RuntimeError("missing values in multi-series generated data")
    physical_count = blocks.groupby(["PROJ_NO", "BLK_NO"]).ngroups
    if physical_count != expected_physical_blocks:
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            f"cause=physical_block_count_mismatch actual={physical_count} expected={expected_physical_blocks}"
        )
        raise RuntimeError("physical block count mismatch")
    combinations_by_block = blocks.groupby(["PROJ_NO", "BLK_NO"], sort=False)["GYEL"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    indexed_combinations = []
    for (_, block_no), combination in combinations_by_block.items():
        block_text = str(block_no)
        prefix = "SYNTH_MULTI_BLK_"
        suffix = block_text[len(prefix) :] if block_text.startswith(prefix) else ""
        if not suffix.isdigit() or int(suffix) <= 0:
            print(
                "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
                f"cause=invalid_generated_block_id block_no={block_text}"
            )
            raise RuntimeError(f"invalid generated physical block id: {block_text}")
        indexed_combinations.append((int(suffix), combination))
    physical_indices = [index for index, _ in indexed_combinations]
    if len(set(physical_indices)) != len(physical_indices):
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            "cause=duplicate_generated_physical_index"
        )
        raise RuntimeError("duplicate generated physical block index")
    actual_combinations = tuple(
        combination for _, combination in sorted(indexed_combinations, key=lambda item: item[0])
    )
    if actual_combinations != tuple(expected_series_combinations):
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            "cause=series_combination_mismatch"
        )
        raise RuntimeError("generated physical-block series combinations do not match sampled combinations")
    if blocks.duplicated(["PROJ_NO", "GYEL", "BLK_NO"]).any():
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            "cause=duplicate_series_block"
        )
        raise RuntimeError("duplicate generated series block")
    if work_orders["WK_ORD_NO"].duplicated().any():
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            "cause=duplicate_work_order"
        )
        raise RuntimeError("duplicate generated work order")
    if expected_allocations is not None:
        allocation_columns = (
            "PROJ_NO",
            "BLK_NO",
            "GYEL",
            "FORMULA_WO_QTY",
            "TOTAL_WO_QTY",
            "SERIES_WO_QTY",
            "PHYSICAL_RANDOM_SEED",
            "SERIES_RANDOM_SEED",
        )
        missing_allocations = [
            column for column in allocation_columns if column not in expected_allocations.columns
        ]
        if missing_allocations or expected_allocations.empty:
            print(
                "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
                f"cause=invalid_allocation_table missing={missing_allocations} "
                f"rows={len(expected_allocations)}"
            )
            raise RuntimeError("invalid expected series allocation table")
        allocation = expected_allocations[list(allocation_columns)].copy()
        numeric_allocation_columns = [
            "FORMULA_WO_QTY",
            "TOTAL_WO_QTY",
            "SERIES_WO_QTY",
            "PHYSICAL_RANDOM_SEED",
            "SERIES_RANDOM_SEED",
        ]
        allocation[numeric_allocation_columns] = allocation[
            numeric_allocation_columns
        ].apply(pd.to_numeric, errors="coerce")
        allocation_numeric = allocation[numeric_allocation_columns].to_numpy(dtype=float)
        if (
            allocation.isna().any(axis=None)
            or allocation.duplicated(["PROJ_NO", "BLK_NO", "GYEL"]).any()
            or (allocation[numeric_allocation_columns] <= 0).any(axis=None)
            or not np.equal(allocation_numeric, np.floor(allocation_numeric)).all()
            or not allocation["FORMULA_WO_QTY"].eq(allocation["TOTAL_WO_QTY"]).all()
        ):
            print(
                "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
                "cause=invalid_allocation_values"
            )
            raise RuntimeError("invalid expected series allocation values")
        expected_series_counts = allocation.set_index(
            ["PROJ_NO", "BLK_NO", "GYEL"]
        )["SERIES_WO_QTY"].astype("int64").sort_index()
        actual_series_counts = work_orders.groupby(
            ["PROJ_NO", "BLK_NO", "GYEL"], sort=True
        ).size().astype("int64").sort_index()
        if not expected_series_counts.equals(actual_series_counts):
            print(
                "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
                "cause=series_wo_count_mismatch"
            )
            raise RuntimeError("generated series W/O counts do not match allocations")
        physical_allocation = allocation.groupby(
            ["PROJ_NO", "BLK_NO"], sort=True
        ).agg(
            ALLOCATED_WO_QTY=("SERIES_WO_QTY", "sum"),
            FORMULA_WO_QTY=("FORMULA_WO_QTY", "first"),
            FORMULA_WO_QTY_UNIQUE=("FORMULA_WO_QTY", "nunique"),
        )
        actual_physical_counts = work_orders.groupby(
            ["PROJ_NO", "BLK_NO"], sort=True
        ).size()
        if (
            not physical_allocation["ALLOCATED_WO_QTY"].astype("int64").equals(
                physical_allocation["FORMULA_WO_QTY"].astype("int64")
            )
            or not physical_allocation["ALLOCATED_WO_QTY"].astype("int64").equals(
                actual_physical_counts.astype("int64")
            )
            or not physical_allocation["FORMULA_WO_QTY_UNIQUE"].eq(1).all()
        ):
            print(
                "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
                "cause=physical_wo_count_mismatch"
            )
            raise RuntimeError("physical W/O total does not match original formula count")
    numeric = work_orders[
        ["LTH", "BTH", "THK", "CUT_LTH", "MARK_LTH", "BVL_LTH", "STL_QTY", "BV_QTY", "TACT_TIME", "PTLST_QTY"]
    ].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any(axis=None) or (numeric[["LTH", "BTH", "THK", "TACT_TIME", "PTLST_QTY"]] <= 0).any(axis=None):
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            "cause=invalid_positive_feature"
        )
        raise RuntimeError("invalid positive multi-series feature")
    if (numeric[["CUT_LTH", "MARK_LTH", "BVL_LTH", "STL_QTY", "BV_QTY"]] < 0).any(axis=None):
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            "cause=invalid_non_negative_feature"
        )
        raise RuntimeError("invalid non-negative multi-series feature")
    recomputed = _aggregate_multi_series_blocks(work_orders)
    left = blocks.sort_values(["PROJ_NO", "BLK_NO", "GYEL"]).reset_index(drop=True)
    right = recomputed.sort_values(["PROJ_NO", "BLK_NO", "GYEL"]).reset_index(drop=True)
    if not left.equals(right):
        print(
            "[ERROR][multi_series_formula_data_generator.validate_multi_series_formula_data] "
            "cause=block_aggregation_mismatch"
        )
        raise RuntimeError("multi-series block aggregation mismatch")
