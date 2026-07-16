"""실적 계열 조합과 계열별 수식을 사용하는 공용 합성데이터 생성기."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from Utils.data.report_formula_data_generator import (
    DEFAULT_MULTI_SERIES_WO_SOURCE,
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


def generate_multi_series_formula_data(
    n_physical_blocks: int,
    seed: int = 2026,
    wo_source_path: str | Path = DEFAULT_MULTI_SERIES_WO_SOURCE,
    block_source_path: str | Path = DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
) -> MultiSeriesFormulaGeneration:
    """전체 W/O 수를 먼저 생성한 뒤 계열별 count vector와 W/O를 생성한다."""

    if n_physical_blocks <= 0:
        print(
            "[ERROR][multi_series_formula_data_generator.generate_multi_series_formula_data] "
            f"cause=invalid_n_physical_blocks value={n_physical_blocks}"
        )
        raise ValueError("n_physical_blocks must be positive")
    wo_path = Path(wo_source_path).resolve()
    block_path = Path(block_source_path).resolve()
    profile = load_physical_block_joint_profile(wo_path, block_path)
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
            wo_source_path=wo_path,
            block_source_path=block_path,
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
    wo_source_path: str | Path = DEFAULT_MULTI_SERIES_WO_SOURCE,
    block_source_path: str | Path = DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
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
            wo_source_path=wo_source_path,
            block_source_path=block_source_path,
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
    wo_source_path: str | Path = DEFAULT_MULTI_SERIES_WO_SOURCE,
    block_source_path: str | Path = DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
) -> PhysicalBlockJointProfile:
    """실적 물리 블록 공동분포를 캐시해 반환한다."""

    return _load_physical_block_joint_profile(
        str(Path(wo_source_path).resolve()),
        str(Path(block_source_path).resolve()),
    )


@lru_cache(maxsize=8)
def _load_physical_block_joint_profile(
    wo_source_path: str,
    block_source_path: str,
) -> PhysicalBlockJointProfile:
    wo_path = Path(wo_source_path)
    block_path = Path(block_source_path)
    missing = [path for path in (wo_path, block_path) if not path.exists()]
    if missing:
        print(
            "[ERROR][multi_series_formula_data_generator._load_physical_block_joint_profile] "
            f"cause=source_not_found paths={missing}"
        )
        raise FileNotFoundError(missing[0])
    work_orders = pd.read_excel(wo_path, sheet_name="Sheet1")
    blocks = pd.read_excel(block_path, sheet_name="Sheet1")
    profile = fit_physical_block_joint_profile(work_orders, blocks)
    print(
        "[CHECK][multi_series_formula_data_generator._load_physical_block_joint_profile] "
        f"physical_blocks={profile.physical_block_count} combinations={len(profile.combinations)} "
        f"wo_path={wo_path} block_path={block_path}"
    )
    return profile


@lru_cache(maxsize=12)
def _load_empirical_series_generator(
    wo_source_path: str,
    block_source_path: str,
    series: str,
):
    # 기존 계열별 식 적합 구현을 분석 CLI와 공용 실행 경로가 함께 사용한다.
    from 데이터분석.shipyard_data_generator import ShipyardGenerator

    return ShipyardGenerator(
        wo_source_path,
        block_source_path,
        mode="spearman",
        series=series,
    ).fit()


def _generate_one_series(
    series: str,
    block_count: int,
    seed: int,
    wo_source_path: Path,
    block_source_path: Path,
    wo_counts: Sequence[int],
    block_seeds: Sequence[int],
) -> pd.DataFrame:
    if series == "NP":
        return generate_report_formula_data(
            n_blocks=block_count,
            seed=seed,
            gyel="NP",
            bth_source_path=wo_source_path,
            wo_counts=wo_counts,
            block_seeds=block_seeds,
        ).wo_df.copy()
    generator = _load_empirical_series_generator(
        str(wo_source_path), str(block_source_path), series
    )
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
