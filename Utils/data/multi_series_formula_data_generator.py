"""실적 계열 조합과 계열별 수식을 사용하는 공용 합성데이터 생성기."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from Utils.data.report_formula_data_generator import (
    DEFAULT_MULTI_SERIES_WO_SOURCE,
    jobs_from_report_formula_wo,
    generate_report_formula_data,
)


SUPPORTED_SERIES = frozenset({"NP", "FN", "FL", "NC"})
JOINT_BLOCK_FEATURES = ("WO_QTY", "LTH", "THK", "BTH", "CUT_LTH", "BV_QTY")
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
    """물리 블록별 계열 조합과 계열별 공동 percentile profile."""

    rows: pd.DataFrame
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


def fit_physical_block_joint_profile(
    work_orders: pd.DataFrame,
    blocks: pd.DataFrame,
) -> PhysicalBlockJointProfile:
    """실적 물리 블록의 계열 조합과 계열별 핵심 특성 공동순위를 적합한다."""

    identity = ("PROJ_NO", "BLK_NO", "GYEL")
    missing_wo = [column for column in identity if column not in work_orders.columns]
    missing_block = [column for column in identity + JOINT_BLOCK_FEATURES[1:] if column not in blocks.columns]
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
    normalized_blocks = blocks[list(identity + JOINT_BLOCK_FEATURES[1:])].copy()
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
        .rename("WO_QTY")
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
    rows[list(JOINT_BLOCK_FEATURES)] = rows[list(JOINT_BLOCK_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = rows[list(JOINT_BLOCK_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (rows[["WO_QTY", "LTH", "THK", "BTH"]] <= 0).any(axis=None):
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

    for feature in JOINT_BLOCK_FEATURES:
        rows[f"{feature}__rank"] = rows.groupby("GYEL", sort=False)[feature].rank(
            method="average", pct=True
        )
    rows = rows.sort_values(list(identity), kind="stable").reset_index(drop=True)
    physical_combinations = rows.groupby(["PROJ_NO", "BLK_NO"], sort=True)["GYEL"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    counts = physical_combinations.value_counts(sort=False).sort_index()
    total = int(counts.sum())
    probabilities = counts.to_numpy(dtype=float) / total
    return PhysicalBlockJointProfile(
        rows=rows,
        physical_keys=tuple((str(project), str(block)) for project, block in physical_combinations.index),
        combinations=tuple(counts.index.tolist()),
        probabilities=tuple(float(value) for value in probabilities),
        counts=tuple(int(value) for value in counts.to_numpy(dtype=int)),
        physical_block_count=total,
    )


def sample_physical_block_joint_targets(
    profile: PhysicalBlockJointProfile,
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """실적 물리 블록 donor를 뽑아 계열 조합과 공동순위를 함께 반환한다."""

    if count <= 0:
        print(
            "[ERROR][multi_series_formula_data_generator.sample_physical_block_joint_targets] "
            f"cause=invalid_count value={count}"
        )
        raise ValueError("series combination sample count must be positive")
    if not profile.physical_keys or len(profile.physical_keys) != profile.physical_block_count:
        print(
            "[ERROR][multi_series_formula_data_generator.sample_physical_block_joint_targets] "
            "cause=invalid_profile"
        )
        raise RuntimeError("invalid physical-block joint profile")
    donor_indices = rng.integers(0, len(profile.physical_keys), size=count)
    selected_keys = [profile.physical_keys[int(index)] for index in donor_indices]
    selected = pd.DataFrame(
        {
            "physical_index": np.arange(1, count + 1, dtype=int),
            "DONOR_PROJ_NO": [key[0] for key in selected_keys],
            "DONOR_BLK_NO": [key[1] for key in selected_keys],
        }
    )
    sampled = selected.merge(
        profile.rows,
        left_on=["DONOR_PROJ_NO", "DONOR_BLK_NO"],
        right_on=["PROJ_NO", "BLK_NO"],
        how="left",
        sort=False,
        validate="many_to_many",
    )
    if sampled["GYEL"].isna().any():
        print(
            "[ERROR][multi_series_formula_data_generator.sample_physical_block_joint_targets] "
            f"cause=donor_join_failed rows={int(sampled['GYEL'].isna().sum())}"
        )
        raise RuntimeError("sampled physical-block donor could not be joined")
    return sampled.sort_values(["physical_index", "GYEL"], kind="stable").reset_index(drop=True)


def generate_multi_series_formula_data(
    n_physical_blocks: int,
    seed: int = 2026,
    wo_source_path: str | Path = DEFAULT_MULTI_SERIES_WO_SOURCE,
    block_source_path: str | Path = DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
) -> MultiSeriesFormulaGeneration:
    """실적 계열 조합을 표본화하고 계열별 수식으로 block/W/O를 생성한다."""

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
    combination_seed, *series_seeds = seed_sequence.spawn(1 + len(SUPPORTED_SERIES))
    joint_targets = sample_physical_block_joint_targets(
        profile,
        n_physical_blocks,
        np.random.default_rng(combination_seed),
    )
    combinations = tuple(
        joint_targets.groupby("physical_index", sort=True)["GYEL"].agg(
            lambda values: tuple(sorted(set(values)))
        )
    )
    series_seed_map = {
        series: int(child.generate_state(1, dtype=np.uint32)[0])
        for series, child in zip(sorted(SUPPORTED_SERIES), series_seeds)
    }

    wo_frames = []
    for series in sorted(SUPPORTED_SERIES):
        series_targets = joint_targets.loc[joint_targets["GYEL"].eq(series)].copy()
        if series_targets.empty:
            continue
        raw_wos = _generate_one_series(
            series=series,
            block_count=len(series_targets),
            seed=series_seed_map[series],
            wo_source_path=wo_path,
            block_source_path=block_path,
        )
        generated_blocks = _aggregate_generated_joint_blocks(raw_wos)
        source_to_physical = match_generated_blocks_to_joint_targets(
            generated_blocks,
            series_targets,
        )
        wo_frames.append(
            _assign_physical_block_identity(raw_wos, series, source_to_physical)
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
) -> pd.DataFrame:
    if series == "NP":
        return generate_report_formula_data(
            n_blocks=block_count,
            seed=seed,
            gyel="NP",
            bth_source_path=wo_source_path,
        ).wo_df.copy()
    generator = _load_empirical_series_generator(
        str(wo_source_path), str(block_source_path), series
    )
    work_orders, _ = generator.generate(n_blocks=block_count, seed=seed)
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


def _aggregate_generated_joint_blocks(work_orders: pd.DataFrame) -> pd.DataFrame:
    """수식 생성 W/O를 공동순위 매칭에 필요한 series-block 특성으로 집계한다."""

    required = JOINT_BLOCK_FEATURES[1:]
    missing = [column for column in required if column not in work_orders.columns]
    if missing:
        print(
            "[ERROR][multi_series_formula_data_generator._aggregate_generated_joint_blocks] "
            f"cause=missing_features columns={missing}"
        )
        raise RuntimeError(f"missing generated joint features: {missing}")
    frame = work_orders.copy()
    frame["source_block_id"] = _source_block_ids(frame)
    grouped = frame.groupby("source_block_id", sort=False)
    return grouped.agg(
        WO_QTY=("source_block_id", "size"),
        LTH=("LTH", "max"),
        THK=("THK", "max"),
        BTH=("BTH", "max"),
        CUT_LTH=("CUT_LTH", "sum"),
        BV_QTY=("BV_QTY", "sum"),
    ).reset_index()


def match_generated_blocks_to_joint_targets(
    generated_blocks: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> dict[str, int]:
    """계열별 생성 block을 donor 공동순위와 최소비용 일대일 매칭한다."""

    generated_required = ("source_block_id",) + JOINT_BLOCK_FEATURES
    target_required = ("physical_index", "GYEL") + tuple(
        f"{feature}__rank" for feature in JOINT_BLOCK_FEATURES
    )
    missing_generated = [column for column in generated_required if column not in generated_blocks.columns]
    missing_target = [column for column in target_required if column not in target_rows.columns]
    if missing_generated or missing_target:
        print(
            "[ERROR][multi_series_formula_data_generator.match_generated_blocks_to_joint_targets] "
            f"cause=missing_columns generated={missing_generated} target={missing_target}"
        )
        raise RuntimeError(
            f"missing joint matching columns: generated={missing_generated}, target={missing_target}"
        )
    if len(generated_blocks) != len(target_rows) or len(generated_blocks) == 0:
        print(
            "[ERROR][multi_series_formula_data_generator.match_generated_blocks_to_joint_targets] "
            f"cause=count_mismatch generated={len(generated_blocks)} target={len(target_rows)}"
        )
        raise RuntimeError("generated and target block counts must match and be positive")
    if generated_blocks["source_block_id"].duplicated().any() or target_rows["physical_index"].duplicated().any():
        print(
            "[ERROR][multi_series_formula_data_generator.match_generated_blocks_to_joint_targets] "
            "cause=duplicate_identity"
        )
        raise RuntimeError("joint matching identities must be unique")

    generated = generated_blocks.reset_index(drop=True).copy()
    targets = target_rows.sort_values("physical_index", kind="stable").reset_index(drop=True).copy()
    generated_rank_columns = []
    target_rank_columns = []
    for feature in JOINT_BLOCK_FEATURES:
        numeric = pd.to_numeric(generated[feature], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            print(
                "[ERROR][multi_series_formula_data_generator.match_generated_blocks_to_joint_targets] "
                f"cause=invalid_generated_feature feature={feature}"
            )
            raise RuntimeError(f"invalid generated matching feature: {feature}")
        rank_column = f"{feature}__rank"
        generated[rank_column] = numeric.rank(method="average", pct=True)
        generated_rank_columns.append(rank_column)
        target_rank_columns.append(rank_column)

    generated_ranks = generated[generated_rank_columns].to_numpy(dtype=float)
    target_ranks = targets[target_rank_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(target_ranks).all()
        or (target_ranks <= 0.0).any()
        or (target_ranks > 1.0).any()
    ):
        print(
            "[ERROR][multi_series_formula_data_generator.match_generated_blocks_to_joint_targets] "
            "cause=invalid_target_ranks"
        )
        raise RuntimeError("invalid donor target ranks")

    differences = generated_ranks[:, None, :] - target_ranks[None, :, :]
    costs = np.square(differences).mean(axis=2)
    generated_indices, target_indices = linear_sum_assignment(costs)
    if len(generated_indices) != len(generated):
        print(
            "[ERROR][multi_series_formula_data_generator.match_generated_blocks_to_joint_targets] "
            f"cause=incomplete_assignment assigned={len(generated_indices)} expected={len(generated)}"
        )
        raise RuntimeError("incomplete physical-block joint assignment")
    return {
        str(generated.iloc[int(generated_index)]["source_block_id"]): int(
            targets.iloc[int(target_index)]["physical_index"]
        )
        for generated_index, target_index in zip(generated_indices, target_indices)
    }


def _assign_physical_block_identity(
    work_orders: pd.DataFrame,
    series: str,
    source_to_physical: Mapping[str, int],
) -> pd.DataFrame:
    frame = work_orders.copy()
    source_block_ids = _source_block_ids(frame)
    unique_ids = tuple(dict.fromkeys(source_block_ids.tolist()))
    if set(unique_ids) != set(source_to_physical):
        print(
            "[ERROR][multi_series_formula_data_generator._assign_physical_block_identity] "
            f"cause=source_mapping_mismatch series={series} generated={len(unique_ids)} mapped={len(source_to_physical)}"
        )
        raise RuntimeError(f"generated source mapping mismatch for {series}")
    mapped = source_block_ids.map(source_to_physical)
    if mapped.isna().any():
        print(
            "[ERROR][multi_series_formula_data_generator._assign_physical_block_identity] "
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
            "[ERROR][multi_series_formula_data_generator._assign_physical_block_identity] "
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
