"""MIXED 실적·생성 데이터의 계열별·계열 간 상관관계 히트맵을 생성한다."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from itertools import combinations
from pathlib import Path
from types import ModuleType
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.data.multi_series_formula_data_generator import (
    DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
    DEFAULT_MULTI_SERIES_WO_SOURCE,
    fit_physical_block_joint_profile,
    generate_multi_series_formula_data,
)

SERIES = ("NP", "FN", "FL", "NC")
BLOCK_KEYS = ("PROJ_NO", "GYEL", "BLK_NO")
W_O_FEATURES = (
    "LTH",
    "BTH",
    "THK",
    "MARK_LTH",
    "CUT_LTH",
    "BVL_LTH",
    "BV_QTY",
    "PTLST_QTY",
    "STL_QTY",
    "TACT_TIME",
)
BLOCK_FEATURES = W_O_FEATURES[:-1] + ("WO_QTY", "TACT_TIME")
JOINT_FEATURES = ("WO_QTY", "LTH", "THK", "BTH", "CUT_LTH", "BV_QTY")
FEATURE_LABELS = {
    "LTH": "길이",
    "BTH": "폭",
    "THK": "두께",
    "MARK_LTH": "마킹길이",
    "CUT_LTH": "절단길이",
    "BVL_LTH": "베벨길이",
    "BV_QTY": "베벨수량",
    "PTLST_QTY": "부재수량",
    "STL_QTY": "강재수량",
    "WO_QTY": "W/O수",
    "TACT_TIME": "택트타임",
}


def _load_generator_module() -> ModuleType:
    """확정 TACT_TIME 식을 중복 작성하지 않고 생성기에서 직접 읽는다."""

    source = REPO_ROOT / "데이터분석" / "shipyard_data_generator.py"
    spec = importlib.util.spec_from_file_location("shipyard_data_generator", source)
    if spec is None or spec.loader is None:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._load_generator_module] "
            f"cause=module_spec_failed input={source}"
        )
        raise RuntimeError(f"generator_module_load_failed: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    """분석 계약에 필요한 컬럼이 하나라도 없으면 즉시 실패시킨다."""

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._require_columns] "
            f"cause=missing_columns key={label} input={missing}"
        )
        raise RuntimeError(f"missing_columns[{label}]: {missing}")


def _numeric_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> pd.DataFrame:
    """숫자 특성을 strict 변환하고 결측·무한값을 허용하지 않는다."""

    _require_columns(frame, columns, label)
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
    if invalid.any():
        locations = np.argwhere(invalid)[:5]
        examples = [
            f"row={int(row)},column={columns[int(column)]}"
            for row, column in locations
        ]
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._numeric_columns] "
            f"cause=non_numeric_or_non_finite key={label} input={examples}"
        )
        raise RuntimeError(f"invalid_numeric_features[{label}]: {examples}")
    return numeric


def _filter_family(frame: pd.DataFrame, family: str, label: str) -> pd.DataFrame:
    """명시한 계열만 선택하며 빈 결과는 허용하지 않는다."""

    _require_columns(frame, ("GYEL",), label)
    normalized = frame["GYEL"].astype(str).str.strip().str.upper()
    selected = frame.loc[normalized.eq(family)].copy()
    selected["GYEL"] = family
    if selected.empty:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._filter_family] "
            f"cause=empty_series key={label} input={family}"
        )
        raise RuntimeError(f"empty_series[{label}]: {family}")
    return selected


def _prepare_actual_family_data(
    actual_wos: pd.DataFrame,
    actual_blocks: pd.DataFrame,
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """실적 W/O TACT_TIME과 블록 WO_QTY/TACT_TIME을 strict 파생한다."""

    family = family.strip().upper()
    if family not in SERIES:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._prepare_actual_family_data] "
            f"cause=unsupported_series input={family}"
        )
        raise RuntimeError(f"unsupported_series: {family}")

    wo_required = BLOCK_KEYS + W_O_FEATURES[:-1]
    block_required = BLOCK_KEYS + BLOCK_FEATURES[:-2]
    _require_columns(actual_wos, wo_required, f"{family}_actual_wo")
    _require_columns(actual_blocks, block_required, f"{family}_actual_block")
    wos = _filter_family(actual_wos, family, f"{family}_actual_wo")
    blocks = _filter_family(actual_blocks, family, f"{family}_actual_block")

    wo_numeric = _numeric_columns(wos, W_O_FEATURES[:-1], f"{family}_actual_wo")
    wos.loc[:, W_O_FEATURES[:-1]] = wo_numeric
    generator = _load_generator_module()
    wos["TACT_TIME"] = generator._calculate_tact_time(
        wos["CUT_LTH"].to_numpy(dtype=float),
        wos["MARK_LTH"].to_numpy(dtype=float),
        wos["THK"].to_numpy(dtype=float),
        wos["PTLST_QTY"].to_numpy(dtype=float),
        wos["BV_QTY"].to_numpy(dtype=float),
    )

    duplicate_blocks = blocks.duplicated(list(BLOCK_KEYS), keep=False)
    if duplicate_blocks.any():
        examples = blocks.loc[duplicate_blocks, list(BLOCK_KEYS)].head(5).to_dict("records")
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._prepare_actual_family_data] "
            f"cause=duplicate_actual_blocks key={family} input={examples}"
        )
        raise RuntimeError(f"duplicate_actual_blocks[{family}]: {examples}")

    block_numeric = _numeric_columns(blocks, BLOCK_FEATURES[:-2], f"{family}_actual_block")
    blocks.loc[:, BLOCK_FEATURES[:-2]] = block_numeric
    wo_aggregates = (
        wos.groupby(list(BLOCK_KEYS), as_index=False, sort=False)
        .agg(WO_QTY=("BLK_NO", "size"), TACT_TIME=("TACT_TIME", "max"))
    )
    blocks = blocks.merge(
        wo_aggregates,
        on=list(BLOCK_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = blocks["_merge"].ne("both")
    if unmatched.any():
        examples = blocks.loc[unmatched, list(BLOCK_KEYS)].head(5).to_dict("records")
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._prepare_actual_family_data] "
            f"cause=actual_block_without_wo key={family} input={examples}"
        )
        raise RuntimeError(f"actual_block_without_wo[{family}]: {examples}")
    blocks = blocks.drop(columns="_merge")
    blocks["WO_QTY"] = blocks["WO_QTY"].astype(int)
    _numeric_columns(blocks, BLOCK_FEATURES, f"{family}_actual_block_prepared")
    return wos, blocks


def _prepare_generated_family_data(
    generated_wos: pd.DataFrame,
    generated_blocks: pd.DataFrame,
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """하나의 MIXED 생성 결과에서 계열별 W/O와 block을 분리한다."""

    wos = generated_wos.copy()
    blocks = generated_blocks.copy()
    wos = _filter_family(wos, family, f"{family}_generated_wo")
    blocks = _filter_family(blocks, family, f"{family}_generated_block")
    _numeric_columns(wos, W_O_FEATURES, f"{family}_generated_wo")
    _numeric_columns(blocks, BLOCK_FEATURES, f"{family}_generated_block")
    return wos, blocks


def _correlation_matrix(
    frame: pd.DataFrame,
    features: Sequence[str],
    label: str,
) -> pd.DataFrame:
    """Pearson 상관행렬을 계산하고 렌더링 전에 수학적 계약을 검증한다."""

    numeric = _numeric_columns(frame, features, label)
    if len(numeric) < 3:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._correlation_matrix] "
            f"cause=insufficient_rows key={label} input={len(numeric)}"
        )
        raise RuntimeError(f"insufficient_correlation_rows[{label}]: {len(numeric)}")
    constant = [column for column in features if numeric[column].nunique(dropna=False) <= 1]
    if constant:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._correlation_matrix] "
            f"cause=constant_features key={label} input={constant}"
        )
        raise RuntimeError(f"constant_correlation_features[{label}]: {constant}")

    matrix = numeric.corr(method="pearson")
    values = matrix.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or not np.allclose(values, values.T, atol=1e-12)
        or not np.allclose(np.diag(values), 1.0, atol=1e-12)
    ):
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._correlation_matrix] "
            f"cause=invalid_correlation_matrix key={label}"
        )
        raise RuntimeError(f"invalid_correlation_matrix[{label}]")
    return matrix


def _physical_block_cross_series_metrics(
    blocks: pd.DataFrame,
    min_pairs: int = 3,
) -> pd.DataFrame:
    """물리 블록에서 서로 다른 계열의 모든 핵심 특성 쌍 상관을 계산한다."""

    if min_pairs < 3:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._physical_block_cross_series_metrics] "
            f"cause=invalid_min_pairs input={min_pairs}"
        )
        raise ValueError("min_pairs must be at least 3")
    required = BLOCK_KEYS + JOINT_FEATURES
    _require_columns(blocks, required, "physical_block_joint")
    frame = blocks.loc[:, required].copy()
    frame["GYEL"] = frame["GYEL"].astype(str).str.strip().str.upper()
    unknown = sorted(set(frame["GYEL"]) - set(SERIES))
    if unknown:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._physical_block_cross_series_metrics] "
            f"cause=unsupported_series input={unknown}"
        )
        raise RuntimeError(f"unsupported_series[physical_block_joint]: {unknown}")
    duplicates = frame.duplicated(list(BLOCK_KEYS), keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, list(BLOCK_KEYS)].head(5).to_dict("records")
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._physical_block_cross_series_metrics] "
            f"cause=duplicate_series_block input={examples}"
        )
        raise RuntimeError(f"duplicate_series_block[physical_block_joint]: {examples}")
    frame.loc[:, JOINT_FEATURES] = _numeric_columns(
        frame, JOINT_FEATURES, "physical_block_joint"
    )

    by_series = {
        series: frame.loc[frame["GYEL"].eq(series), ["PROJ_NO", "BLK_NO", *JOINT_FEATURES]]
        .set_index(["PROJ_NO", "BLK_NO"])
        for series in SERIES
    }
    rows = []
    for series_a, series_b in combinations(SERIES, 2):
        paired = by_series[series_a].join(
            by_series[series_b],
            how="inner",
            lsuffix="__a",
            rsuffix="__b",
        )
        for feature_a in JOINT_FEATURES:
            for feature_b in JOINT_FEATURES:
                values_a = paired[f"{feature_a}__a"]
                values_b = paired[f"{feature_b}__b"]
                pair_count = len(paired)
                if pair_count < min_pairs:
                    status = "insufficient_pairs"
                    correlation = np.nan
                elif values_a.nunique(dropna=False) <= 1 or values_b.nunique(dropna=False) <= 1:
                    status = "constant_feature"
                    correlation = np.nan
                else:
                    status = "ok"
                    correlation = float(values_a.corr(values_b, method="pearson"))
                rows.append(
                    {
                        "series_a": series_a,
                        "feature_a": feature_a,
                        "series_b": series_b,
                        "feature_b": feature_b,
                        "pair_count": pair_count,
                        "status": status,
                        "correlation": correlation,
                    }
                )
    result = pd.DataFrame(rows)
    expected_rows = len(tuple(combinations(SERIES, 2))) * len(JOINT_FEATURES) ** 2
    if len(result) != expected_rows:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._physical_block_cross_series_metrics] "
            f"cause=relation_count_mismatch actual={len(result)} expected={expected_rows}"
        )
        raise RuntimeError("physical-block cross-series relation count mismatch")
    return result


def _resolve_korean_font(font_path: Path | None) -> font_manager.FontProperties:
    """Windows와 WSL에서 같은 맑은 고딕 파일을 명시적으로 찾는다."""

    candidates = [font_path] if font_path is not None else [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/mnt/c/Windows/Fonts/malgun.ttf"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            return font_manager.FontProperties(fname=str(candidate))
    print(
        "[ERROR][plot_multi_series_generator_heatmaps._resolve_korean_font] "
        f"cause=korean_font_not_found input={[str(path) for path in candidates]}"
    )
    raise RuntimeError("korean_font_not_found")


def _plot_comparison(
    actual_matrix: pd.DataFrame,
    generated_matrix: pd.DataFrame,
    family: str,
    grain: str,
    output_path: Path,
    font: font_manager.FontProperties,
) -> None:
    """실적과 생성 상관행렬을 같은 축·색 범위로 나란히 렌더링한다."""

    labels = [FEATURE_LABELS[column] for column in actual_matrix.columns]
    actual_display = actual_matrix.copy()
    generated_display = generated_matrix.copy()
    actual_display.index = actual_display.columns = labels
    generated_display.index = generated_display.columns = labels

    sns.set_theme(style="white")
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 2, figsize=(24, 10), constrained_layout=True)
    common = {
        "annot": True,
        "fmt": ".2f",
        "cmap": "rocket",
        "vmin": -1.0,
        "vmax": 1.0,
        "square": True,
        "linewidths": 0.35,
        "linecolor": "white",
        "annot_kws": {"size": 8},
    }
    sns.heatmap(actual_display, ax=axes[0], cbar=False, **common)
    sns.heatmap(
        generated_display,
        ax=axes[1],
        cbar=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.85},
        **common,
    )
    axes[0].set_title(f"{family} {grain} 실적 데이터 상관관계", fontproperties=font, fontsize=16)
    axes[1].set_title(f"{family} {grain} 생성 데이터 상관관계", fontproperties=font, fontsize=16)
    for axis in axes:
        axis.tick_params(axis="x", rotation=45, labelsize=9)
        axis.tick_params(axis="y", rotation=0, labelsize=9)
    figure.suptitle(
        f"{family} {grain} 실적·생성 데이터 Pearson 상관관계 비교",
        fontproperties=font,
        fontsize=20,
    )
    figure.text(
        0.5,
        0.005,
        "TACT_TIME은 전 계열 공통 확정식으로 산출",
        ha="center",
        fontproperties=font,
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _merge_cross_series_metrics(
    actual_blocks: pd.DataFrame,
    generated_blocks: pd.DataFrame,
) -> pd.DataFrame:
    """실적·생성 물리 블록 계열 간 상관을 같은 관계 key로 결합한다."""

    keys = ["series_a", "feature_a", "series_b", "feature_b"]
    actual = _physical_block_cross_series_metrics(actual_blocks).rename(
        columns={
            "pair_count": "actual_pair_count",
            "status": "actual_status",
            "correlation": "actual_correlation",
        }
    )
    generated = _physical_block_cross_series_metrics(generated_blocks).rename(
        columns={
            "pair_count": "generated_pair_count",
            "status": "generated_status",
            "correlation": "generated_correlation",
        }
    )
    comparison = actual.merge(generated, on=keys, how="outer", validate="one_to_one", indicator=True)
    unmatched = comparison["_merge"].ne("both")
    if unmatched.any():
        examples = comparison.loc[unmatched, keys + ["_merge"]].head(5).to_dict("records")
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._merge_cross_series_metrics] "
            f"cause=relation_key_mismatch input={examples}"
        )
        raise RuntimeError("actual/generated cross-series relation keys do not match")
    comparison = comparison.drop(columns="_merge")
    valid = comparison["actual_status"].eq("ok") & comparison["generated_status"].eq("ok")
    comparison["absolute_error"] = np.nan
    comparison.loc[valid, "absolute_error"] = (
        comparison.loc[valid, "actual_correlation"]
        - comparison.loc[valid, "generated_correlation"]
    ).abs()
    return comparison


def _cross_series_matrix(metrics: pd.DataFrame, correlation_column: str) -> pd.DataFrame:
    """관계형 metric을 24×24 계열-특성 heatmap 행렬로 변환한다."""

    labels = [f"{series}:{feature}" for series in SERIES for feature in JOINT_FEATURES]
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for label in labels:
        matrix.loc[label, label] = 1.0
    for row in metrics.itertuples(index=False):
        value = getattr(row, correlation_column)
        status_column = correlation_column.replace("correlation", "status")
        if getattr(row, status_column) != "ok" or not np.isfinite(float(value)):
            continue
        left = f"{row.series_a}:{row.feature_a}"
        right = f"{row.series_b}:{row.feature_b}"
        matrix.loc[left, right] = float(value)
        matrix.loc[right, left] = float(value)
    return matrix


def _series_pair_matrix(
    metrics: pd.DataFrame,
    series_a: str,
    series_b: str,
    correlation_column: str,
) -> pd.DataFrame:
    """한 계열 쌍의 특성 간 상관을 읽기 쉬운 6×6 행렬로 변환한다."""

    required = {
        "series_a",
        "feature_a",
        "series_b",
        "feature_b",
        correlation_column,
    }
    status_column = correlation_column.replace("correlation", "status")
    required.add(status_column)
    missing = sorted(required - set(metrics.columns))
    if missing:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._series_pair_matrix] "
            f"cause=missing_columns key={series_a}-{series_b} input={missing}"
        )
        raise RuntimeError(f"missing_series_pair_columns[{series_a}-{series_b}]: {missing}")

    selected = metrics.loc[
        metrics["series_a"].eq(series_a) & metrics["series_b"].eq(series_b)
    ]
    if selected.empty:
        print(
            "[ERROR][plot_multi_series_generator_heatmaps._series_pair_matrix] "
            f"cause=missing_series_pair input={series_a}-{series_b}"
        )
        raise RuntimeError(f"missing_series_pair: {series_a}-{series_b}")

    matrix = pd.DataFrame(np.nan, index=JOINT_FEATURES, columns=JOINT_FEATURES, dtype=float)
    for row in selected.itertuples(index=False):
        value = getattr(row, correlation_column)
        if getattr(row, status_column) != "ok" or not np.isfinite(float(value)):
            continue
        matrix.loc[row.feature_a, row.feature_b] = float(value)
    return matrix


def _cross_series_absolute_error_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    """실적·생성 계열 간 상관계수의 절대오차를 대칭 행렬로 만든다."""

    actual = _cross_series_matrix(metrics, "actual_correlation")
    generated = _cross_series_matrix(metrics, "generated_correlation")
    return (actual - generated).abs()


def _plot_cross_series_comparison(
    metrics: pd.DataFrame,
    output_path: Path,
    font: font_manager.FontProperties,
) -> None:
    """물리 블록의 계열 간 공동 상관행렬을 실적·생성으로 비교한다."""

    actual = _cross_series_matrix(metrics, "actual_correlation")
    generated = _cross_series_matrix(metrics, "generated_correlation")
    sns.set_theme(style="white")
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 2, figsize=(30, 14), constrained_layout=True)
    common = {
        "cmap": "rocket",
        "vmin": -1.0,
        "vmax": 1.0,
        "square": True,
        "linewidths": 0.25,
        "linecolor": "white",
        "mask": None,
    }
    sns.heatmap(actual, ax=axes[0], cbar=False, mask=actual.isna(), **{k: v for k, v in common.items() if k != "mask"})
    sns.heatmap(
        generated,
        ax=axes[1],
        cbar=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        mask=generated.isna(),
        **{k: v for k, v in common.items() if k != "mask"},
    )
    axes[0].set_title("물리 블록 계열 간 실적 공동 상관", fontproperties=font, fontsize=16)
    axes[1].set_title("물리 블록 계열 간 생성 공동 상관", fontproperties=font, fontsize=16)
    for axis in axes:
        axis.tick_params(axis="x", rotation=90, labelsize=7)
        axis.tick_params(axis="y", rotation=0, labelsize=7)
    figure.suptitle(
        "동일 물리 블록 내 NP/FN/FL/NC 공동분포 비교",
        fontproperties=font,
        fontsize=20,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_series_pair_comparison(
    metrics: pd.DataFrame,
    series_a: str,
    series_b: str,
    output_path: Path,
    font: font_manager.FontProperties,
) -> None:
    """한 계열 쌍의 실적·생성 공동 상관을 6×6 확대 그림으로 저장한다."""

    actual = _series_pair_matrix(metrics, series_a, series_b, "actual_correlation")
    generated = _series_pair_matrix(metrics, series_a, series_b, "generated_correlation")
    labels = [FEATURE_LABELS[feature] for feature in JOINT_FEATURES]
    actual.index = generated.index = [f"{series_a} {label}" for label in labels]
    actual.columns = generated.columns = [f"{series_b} {label}" for label in labels]

    pair_rows = metrics.loc[
        metrics["series_a"].eq(series_a) & metrics["series_b"].eq(series_b)
    ]
    actual_pair_count = int(pair_rows["actual_pair_count"].iloc[0])
    generated_pair_count = int(pair_rows["generated_pair_count"].iloc[0])

    sns.set_theme(style="white")
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    common = {
        "annot": True,
        "fmt": ".2f",
        "cmap": "rocket",
        "vmin": -1.0,
        "vmax": 1.0,
        "square": True,
        "linewidths": 0.35,
        "linecolor": "white",
        "annot_kws": {"size": 8},
    }
    sns.heatmap(actual, ax=axes[0], cbar=False, mask=actual.isna(), **common)
    sns.heatmap(
        generated,
        ax=axes[1],
        cbar=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.85},
        mask=generated.isna(),
        **common,
    )
    axes[0].set_title(
        f"실적 동일 블록 ({actual_pair_count}쌍)",
        fontproperties=font,
        fontsize=14,
    )
    axes[1].set_title(
        f"생성 동일 블록 ({generated_pair_count}쌍)",
        fontproperties=font,
        fontsize=14,
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=45, labelsize=9)
        axis.tick_params(axis="y", rotation=0, labelsize=9)
    figure.suptitle(
        f"동일 물리 블록 내 {series_a}-{series_b} 계열 공동분포",
        fontproperties=font,
        fontsize=18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_cross_series_absolute_error(
    metrics: pd.DataFrame,
    output_path: Path,
    font: font_manager.FontProperties,
) -> None:
    """계열 간 상관계수 절대오차가 큰 위치를 한 장에 표시한다."""

    matrix = _cross_series_absolute_error_matrix(metrics)
    sns.set_theme(style="white")
    plt.rcParams["font.family"] = font.get_name()
    figure, axis = plt.subplots(figsize=(17, 15), constrained_layout=True)
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.25,
        linecolor="white",
        mask=matrix.isna(),
        cbar_kws={"label": "|실적 r - 생성 r|", "shrink": 0.8},
    )
    axis.set_title(
        "동일 물리 블록 계열 간 상관계수 절대오차",
        fontproperties=font,
        fontsize=18,
    )
    axis.tick_params(axis="x", rotation=90, labelsize=7)
    axis.tick_params(axis="y", rotation=0, labelsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_heatmaps(
    actual_wo_path: Path,
    actual_block_path: Path,
    output_dir: Path,
    seed: int,
    font_path: Path | None,
) -> list[Path]:
    """최신 MIXED 생성값으로 계열 내부·계열 간 heatmap 16개를 만든다."""

    for path in (actual_wo_path, actual_block_path):
        if not path.is_file():
            print(
                "[ERROR][plot_multi_series_generator_heatmaps.generate_heatmaps] "
                f"cause=missing_actual_file input={path}"
            )
            raise FileNotFoundError(path)
    actual_wos = pd.read_excel(actual_wo_path)
    actual_blocks = pd.read_excel(actual_block_path)
    physical_block_count = actual_blocks.groupby(["PROJ_NO", "BLK_NO"]).ngroups
    generated = generate_multi_series_formula_data(
        n_physical_blocks=physical_block_count,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generated.wo_df.to_csv(output_dir / f"mixed_wo_seed{seed}.csv", index=False, encoding="utf-8-sig")
    generated.block_df.to_csv(
        output_dir / f"mixed_block_seed{seed}.csv", index=False, encoding="utf-8-sig"
    )
    font = _resolve_korean_font(font_path)
    outputs: list[Path] = []
    error_summary_rows: list[dict] = []
    for family in SERIES:
        actual_wo, actual_block = _prepare_actual_family_data(
            actual_wos,
            actual_blocks,
            family,
        )
        generated_wo, generated_block = _prepare_generated_family_data(
            generated.wo_df,
            generated.block_df,
            family,
        )
        for grain, actual, generated_frame, features in (
            ("W/O", actual_wo, generated_wo, W_O_FEATURES),
            ("블록", actual_block, generated_block, BLOCK_FEATURES),
        ):
            key = "wo" if grain == "W/O" else "block"
            actual_matrix = _correlation_matrix(actual, features, f"{family}_{key}_actual")
            generated_matrix = _correlation_matrix(
                generated_frame,
                features,
                f"{family}_{key}_generated",
            )
            upper = np.triu_indices(len(features), k=1)
            errors = np.abs(
                actual_matrix.to_numpy(dtype=float)[upper]
                - generated_matrix.to_numpy(dtype=float)[upper]
            )
            error_summary_rows.append(
                {
                    "scope": "within_series",
                    "series_a": family,
                    "series_b": family,
                    "grain": key,
                    "relation_count": int(len(errors)),
                    "actual_pair_count": int(len(actual)),
                    "generated_pair_count": int(len(generated_frame)),
                    "mean_absolute_correlation_error": float(errors.mean()),
                    "max_absolute_correlation_error": float(errors.max()),
                }
            )
            output_path = output_dir / f"{family.lower()}_{key}_correlation_heatmap.png"
            _plot_comparison(
                actual_matrix,
                generated_matrix,
                family,
                grain,
                output_path,
                font,
            )
            outputs.append(output_path)
            print(
                "[CHECK][plot_multi_series_generator_heatmaps.generate_heatmaps] "
                f"series={family} grain={key} actual_rows={len(actual)} "
                f"generated_rows={len(generated_frame)} output={output_path}"
            )

    actual_joint = fit_physical_block_joint_profile(actual_wos, actual_blocks).rows.rename(
        columns={"SERIES_WO_QTY": "WO_QTY"}
    )
    joint_metrics = _merge_cross_series_metrics(actual_joint, generated.block_df)
    metrics_path = output_dir / "physical_block_cross_series_correlations.csv"
    joint_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    joint_path = output_dir / "physical_block_cross_series_correlation_heatmap.png"
    _plot_cross_series_comparison(joint_metrics, joint_path, font)
    outputs.append(joint_path)

    for series_a, series_b in combinations(SERIES, 2):
        pair_rows = joint_metrics.loc[
            joint_metrics["series_a"].eq(series_a)
            & joint_metrics["series_b"].eq(series_b)
            & joint_metrics["absolute_error"].notna()
        ]
        if len(pair_rows) != len(JOINT_FEATURES) ** 2:
            print(
                "[ERROR][plot_multi_series_generator_heatmaps.generate_heatmaps] "
                f"cause=incomplete_pair_metrics key={series_a}-{series_b} input={len(pair_rows)}"
            )
            raise RuntimeError(
                f"incomplete_pair_metrics[{series_a}-{series_b}]: {len(pair_rows)}"
            )
        pair_path = output_dir / f"physical_block_{series_a.lower()}_{series_b.lower()}_heatmap.png"
        _plot_series_pair_comparison(joint_metrics, series_a, series_b, pair_path, font)
        outputs.append(pair_path)
        error_summary_rows.append(
            {
                "scope": "cross_series_physical_block",
                "series_a": series_a,
                "series_b": series_b,
                "grain": "block",
                "relation_count": int(len(pair_rows)),
                "actual_pair_count": int(pair_rows["actual_pair_count"].iloc[0]),
                "generated_pair_count": int(pair_rows["generated_pair_count"].iloc[0]),
                "mean_absolute_correlation_error": float(pair_rows["absolute_error"].mean()),
                "max_absolute_correlation_error": float(pair_rows["absolute_error"].max()),
            }
        )

    error_path = output_dir / "physical_block_cross_series_absolute_error_heatmap.png"
    _plot_cross_series_absolute_error(joint_metrics, error_path, font)
    outputs.append(error_path)
    error_summary_path = output_dir / "correlation_error_summary.csv"
    pd.DataFrame(error_summary_rows).to_csv(
        error_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    valid_errors = joint_metrics["absolute_error"].dropna()
    print(
        "[CHECK][plot_multi_series_generator_heatmaps.generate_heatmaps] "
        f"grain=physical_block_cross_series relations={len(joint_metrics)} "
        f"valid_correlations={len(valid_errors)} mean_abs_error={float(valid_errors.mean()):.6f} "
        f"metrics={metrics_path} output={joint_path}"
    )

    if (
        len(outputs) != 16
        or any(not path.is_file() for path in outputs)
        or not metrics_path.is_file()
        or not error_summary_path.is_file()
    ):
        print(
            "[ERROR][plot_multi_series_generator_heatmaps.generate_heatmaps] "
            f"cause=output_count_mismatch input={len(outputs)}"
        )
        raise RuntimeError(f"heatmap_output_count_mismatch: {len(outputs)}")
    print(
        "[VALIDATION][plot_multi_series_generator_heatmaps.generate_heatmaps] "
        f"passed=true files={len(outputs)} summary={error_summary_path} output={output_dir}"
    )
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIXED 생성값의 계열별·계열 쌍별 상관관계 heatmap 16개 생성",
    )
    parser.add_argument(
        "--actual-wo-xlsx",
        type=Path,
        default=DEFAULT_MULTI_SERIES_WO_SOURCE,
    )
    parser.add_argument(
        "--actual-block-xlsx",
        type=Path,
        default=DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "analysis" / "multi_series_generator_heatmaps",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--font-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        "[CHECK][plot_multi_series_generator_heatmaps.main] "
        f"actual_wo={args.actual_wo_xlsx} actual_block={args.actual_block_xlsx} "
        f"output_dir={args.output_dir} seed={args.seed}"
    )
    generate_heatmaps(
        actual_wo_path=args.actual_wo_xlsx,
        actual_block_path=args.actual_block_xlsx,
        output_dir=args.output_dir,
        seed=args.seed,
        font_path=args.font_path,
    )


if __name__ == "__main__":
    main()
