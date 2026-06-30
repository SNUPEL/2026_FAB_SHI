"""Phase 1 block-only synthetic data generator.

This module is intentionally block-only. It does not create W/O rows and does
not backfill missing Phase 1 objectives. Missing `BV_QTY` is a hard error
because Phase 1 balancing uses it as one of the objective features.

입출력 요약:
- 입력: 실제 블록 데이터 Excel/CSV. 필수 컬럼은 `LTH, THK, CUT_LTH, MARK_LTH, STL_QTY, BV_QTY`.
- 처리: 실제 row를 bootstrap으로 뽑고, 수치형 컬럼에 작은 noise를 더해 synthetic block row 생성.
- 출력: block-only synthetic DataFrame 또는 CSV/JSON validation package.
- 금지: `BV_QTY` 결측을 0으로 대체하지 않는다. 결측/비수치 값은 즉시 실패한다.
"""

# LINE-BY-LINE: 미래 타입 힌트를 문자열로 늦게 평가합니다. 사용: Python annotation 호환성 유지.
from __future__ import annotations

# LINE-BY-LINE: summary.json 저장에 사용합니다.
import json
# LINE-BY-LINE: 입력/출력 파일 경로를 OS 독립적으로 다루기 위해 사용합니다.
from pathlib import Path
# LINE-BY-LINE: 함수 입력 타입을 명확히 표시하기 위해 사용합니다.
from typing import Mapping

# LINE-BY-LINE: 난수 generator와 numeric array 연산에 사용합니다.
import numpy as np
# LINE-BY-LINE: Excel/CSV 로딩, DataFrame validation, CSV 저장에 사용합니다.
import pandas as pd


# LINE-BY-LINE: Phase 1 block-only generator가 반드시 요구하는 실제 컬럼 목록입니다. 하나라도 없으면 실패합니다.
PHASE1_BLOCK_FEATURE_COLUMNS = (
    # LINE-BY-LINE: `LTH`는 블록/판 길이입니다. network state의 길이 feature로도 사용됩니다.
    "LTH",
    # LINE-BY-LINE: `THK`는 두께입니다. network state의 두께 feature로도 사용됩니다.
    "THK",
    # LINE-BY-LINE: `CUT_LTH`는 절단장입니다. Phase 1 목적함수 2순위 평준화 대상입니다.
    "CUT_LTH",
    # LINE-BY-LINE: `MARK_LTH`는 마킹장입니다. 현재 Phase 1 목적함수에는 직접 쓰지 않지만 block 분포 보존용입니다.
    "MARK_LTH",
    # LINE-BY-LINE: `STL_QTY`는 강재수량입니다. Phase 1 목적함수 1순위 평준화 대상입니다.
    "STL_QTY",
    # LINE-BY-LINE: `BV_QTY`는 베벨/개선 수량입니다. Phase 1 목적함수 3순위 평준화 대상입니다.
    "BV_QTY",
)
# LINE-BY-LINE: 0 이하가 나오면 물리적으로 이상한 컬럼입니다. 검증에서 즉시 실패합니다.
POSITIVE_COLUMNS = ("LTH", "THK", "STL_QTY")
# LINE-BY-LINE: 0은 허용하지만 음수는 허용하지 않는 컬럼입니다.
NON_NEGATIVE_COLUMNS = ("CUT_LTH", "MARK_LTH", "BV_QTY")
# LINE-BY-LINE: 산출물 summary에 남기는 generator 버전명입니다. 실험 재현성을 위해 고정합니다.
GENERATOR_NAME = "block_only_empirical_bootstrap_v1"


# LINE-BY-LINE: 실제 블록 Excel/CSV를 읽고 Phase 1 필수 컬럼을 검증하는 entry function입니다.
def load_phase1_actual_blocks(path: str | Path, gyel: str | None = "NP") -> pd.DataFrame:
    """Load actual block rows for Phase 1 generator fitting."""

    # LINE-BY-LINE: 문자열 path를 Path 객체로 변환합니다.
    source_path = Path(path)
    # LINE-BY-LINE: 어떤 파일을 읽는지 로그로 남깁니다.
    print(f"[CHECK][phase1_block_data_generator.load_phase1_actual_blocks] path={source_path}")
    # LINE-BY-LINE: 파일이 없으면 fallback 없이 실패합니다.
    if not source_path.exists():
        print(
            "[ERROR][phase1_block_data_generator.load_phase1_actual_blocks] "
            f"cause=missing_source_file path={source_path}"
        )
        raise RuntimeError(f"missing_source_file: {source_path}")

    # LINE-BY-LINE: Excel 파일이면 pandas read_excel로 읽습니다.
    if source_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(source_path)
    # LINE-BY-LINE: CSV 파일이면 pandas read_csv로 읽습니다.
    elif source_path.suffix.lower() == ".csv":
        df = pd.read_csv(source_path)
    # LINE-BY-LINE: 지원하지 않는 확장자는 데이터 의미를 알 수 없으므로 실패합니다.
    else:
        print(
            "[ERROR][phase1_block_data_generator.load_phase1_actual_blocks] "
            f"cause=unsupported_source_suffix suffix={source_path.suffix}"
        )
        raise RuntimeError(f"unsupported_source_suffix: {source_path.suffix}")

    # LINE-BY-LINE: gyel이 지정되면 NP 등 계열 필터를 적용합니다. 기본은 NP입니다.
    if gyel is not None:
        # LINE-BY-LINE: 계열 필터를 하려면 GYEL 컬럼이 반드시 있어야 합니다.
        if "GYEL" not in df.columns:
            print(
                "[ERROR][phase1_block_data_generator.load_phase1_actual_blocks] "
                f"cause=missing_required_columns missing=['GYEL'] path={source_path}"
            )
            raise RuntimeError("missing_required_columns: GYEL")
        # LINE-BY-LINE: 필터 전 row 수를 로그용으로 보존합니다.
        before = len(df)
        # LINE-BY-LINE: GYEL을 문자열로 비교해 NP row만 남깁니다.
        df = df[df["GYEL"].astype(str).eq(str(gyel))].copy()
        # LINE-BY-LINE: 필터 전후 row 수를 출력합니다.
        print(
            "[CHECK][phase1_block_data_generator.load_phase1_actual_blocks] "
            f"gyel={gyel} rows_before={before} rows_after={len(df)}"
        )
        # LINE-BY-LINE: 필터 결과가 비면 학습 데이터가 없으므로 실패합니다.
        if df.empty:
            print(
                "[ERROR][phase1_block_data_generator.load_phase1_actual_blocks] "
                f"cause=no_rows_after_gyel_filter gyel={gyel} path={source_path}"
            )
            raise RuntimeError(f"no_rows_after_gyel_filter: {gyel}")

    # LINE-BY-LINE: 필수 컬럼/수치/범위 검증을 통과한 DataFrame만 반환합니다.
    return validate_phase1_block_data(df, source_name=str(source_path))


# LINE-BY-LINE: Phase 1 block DataFrame이 학습/생성에 사용할 수 있는지 검증합니다.
def validate_phase1_block_data(blocks: pd.DataFrame, source_name: str = "phase1_blocks", verbose: bool = True) -> pd.DataFrame:
    """Validate required block-level Phase 1 objective features."""

    # LINE-BY-LINE: 필수 컬럼 누락 여부를 먼저 검사합니다.
    missing = [column for column in PHASE1_BLOCK_FEATURE_COLUMNS if column not in blocks.columns]
    # LINE-BY-LINE: 누락 컬럼이 있으면 0 대체 없이 실패합니다.
    if missing:
        print(
            "[ERROR][phase1_block_data_generator.validate_phase1_block_data] "
            f"cause=missing_required_columns missing={missing} source={source_name}"
        )
        raise RuntimeError(f"missing_required_columns: {','.join(missing)}")

    # LINE-BY-LINE: 원본 DataFrame을 직접 변경하지 않기 위해 복사본에서 검증/변환합니다.
    result = blocks.copy()
    # LINE-BY-LINE: 필수 컬럼들을 모두 numeric으로 변환하고, 변환 실패/결측을 검사합니다.
    for column in PHASE1_BLOCK_FEATURE_COLUMNS:
        # LINE-BY-LINE: 숫자로 변환 불가능한 값은 NaN이 됩니다.
        result[column] = pd.to_numeric(result[column], errors="coerce")
        # LINE-BY-LINE: 변환 실패 또는 결측 개수를 셉니다.
        missing_count = int(result[column].isna().sum())
        # LINE-BY-LINE: 하나라도 있으면 데이터 품질 오류이므로 실패합니다.
        if missing_count:
            print(
                "[ERROR][phase1_block_data_generator.validate_phase1_block_data] "
                f"cause=non_numeric_or_missing column={column} count={missing_count} source={source_name}"
            )
            raise RuntimeError(f"non_numeric_or_missing: {column}")

    # LINE-BY-LINE: 양수여야 하는 컬럼이 0 이하인지 검사합니다.
    for column in POSITIVE_COLUMNS:
        # LINE-BY-LINE: invalid_count는 해당 컬럼에서 0 이하인 row 개수입니다.
        invalid_count = int((result[column] <= 0).sum())
        # LINE-BY-LINE: 물리적으로 불가능한 값이 있으면 실패합니다.
        if invalid_count:
            print(
                "[ERROR][phase1_block_data_generator.validate_phase1_block_data] "
                f"cause=non_positive_value column={column} count={invalid_count} source={source_name}"
            )
            raise RuntimeError(f"non_positive_value: {column}")

    # LINE-BY-LINE: 음수만 금지되는 컬럼을 검사합니다. 예: BV_QTY=0은 가능하지만 -1은 불가능합니다.
    for column in NON_NEGATIVE_COLUMNS:
        # LINE-BY-LINE: invalid_count는 해당 컬럼에서 음수인 row 개수입니다.
        invalid_count = int((result[column] < 0).sum())
        # LINE-BY-LINE: 음수 값이 있으면 실패합니다.
        if invalid_count:
            print(
                "[ERROR][phase1_block_data_generator.validate_phase1_block_data] "
                f"cause=negative_value column={column} count={invalid_count} source={source_name}"
            )
            raise RuntimeError(f"negative_value: {column}")

    # LINE-BY-LINE: 검증 통과 row 수를 출력합니다. 학습 루프에서는 verbose=False로 정상 반복 로그만 줄입니다.
    if verbose:
        print(
            "[VALIDATION][phase1_block_data_generator.validate_phase1_block_data] "
            f"passed=true rows={len(result)} source={source_name}"
        )
    # LINE-BY-LINE: numeric 변환까지 끝난 DataFrame을 반환합니다.
    return result


# LINE-BY-LINE: 실제 block 분포에서 synthetic block-only 데이터를 생성합니다.
def generate_phase1_block_data(
    actual_blocks: pd.DataFrame,
    n_blocks: int,
    seed: int = 2026,
    noise_ratio: float = 0.03,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate synthetic Phase 1 block rows from block-level actual data only."""

    # LINE-BY-LINE: 입력 actual block data를 먼저 검증합니다.
    actual = validate_phase1_block_data(actual_blocks, source_name="actual_blocks", verbose=verbose)
    # LINE-BY-LINE: 생성할 block 수가 0 이하이면 의미가 없으므로 실패합니다.
    if n_blocks <= 0:
        print(
            "[ERROR][phase1_block_data_generator.generate_phase1_block_data] "
            f"cause=invalid_n_blocks n_blocks={n_blocks}"
        )
        raise RuntimeError(f"invalid_n_blocks: {n_blocks}")
    # LINE-BY-LINE: noise_ratio가 음수이면 분산 scale이 잘못되므로 실패합니다.
    if noise_ratio < 0:
        print(
            "[ERROR][phase1_block_data_generator.generate_phase1_block_data] "
            f"cause=invalid_noise_ratio noise_ratio={noise_ratio}"
        )
        raise RuntimeError(f"invalid_noise_ratio: {noise_ratio}")

    # LINE-BY-LINE: seed 기반 numpy random generator를 생성합니다.
    rng = np.random.default_rng(seed)
    # LINE-BY-LINE: 실제 block row를 복원추출 bootstrap으로 n_blocks개 뽑습니다.
    base = actual.iloc[rng.integers(0, len(actual), size=n_blocks)].reset_index(drop=True)
    # LINE-BY-LINE: 새 synthetic DataFrame 틀을 만듭니다.
    generated = pd.DataFrame(index=range(n_blocks))
    # LINE-BY-LINE: synthetic block id를 순번 기반으로 생성합니다.
    generated["BLK_ID"] = [f"SYNTH_BLK_{idx + 1:06d}" for idx in range(n_blocks)]
    # LINE-BY-LINE: synthetic project 번호는 실제 project가 아니므로 `SYNTH`로 고정합니다.
    generated["PROJ_NO"] = "SYNTH"
    # LINE-BY-LINE: synthetic block 번호는 BLK_ID와 동일하게 둡니다.
    generated["BLK_NO"] = generated["BLK_ID"]
    # LINE-BY-LINE: 현재 Phase 1 범위가 NP이므로 synthetic 계열도 NP로 둡니다.
    generated["GYEL"] = "NP"

    # LINE-BY-LINE: 필수 feature 컬럼마다 bootstrap 값에 noise를 적용해 synthetic 값을 만듭니다.
    for column in PHASE1_BLOCK_FEATURE_COLUMNS:
        generated[column] = _jitter_column(base[column], actual[column], column, rng, noise_ratio)

    # LINE-BY-LINE: 산출물 식별용 role column입니다. 학습에는 직접 쓰지 않습니다.
    generated["DATA_ROLE"] = "synthetic_phase1_block_only"
    # LINE-BY-LINE: 생성 설정을 로그로 남깁니다. 학습 루프에서는 verbose=False로 episode별 중복 로그를 줄입니다.
    if verbose:
        print(
            "[CHECK][phase1_block_data_generator.generate_phase1_block_data] "
            f"generator={GENERATOR_NAME} rows={len(generated)} seed={seed} noise_ratio={noise_ratio}"
        )
    # LINE-BY-LINE: 생성 결과도 동일한 validator로 다시 검증하고 반환합니다.
    return validate_phase1_block_data(generated, source_name="generated_blocks", verbose=verbose)


def compare_phase1_block_data(
    actual_blocks: pd.DataFrame,
    generated_blocks: pd.DataFrame,
    method: str = "pearson",
) -> dict[str, object]:
    """Compare actual and generated block features with correlations and summaries."""

    actual = validate_phase1_block_data(actual_blocks, source_name="actual_blocks")
    generated = validate_phase1_block_data(generated_blocks, source_name="generated_blocks")
    actual_corr = actual[list(PHASE1_BLOCK_FEATURE_COLUMNS)].corr(method=method)
    generated_corr = generated[list(PHASE1_BLOCK_FEATURE_COLUMNS)].corr(method=method)
    delta_corr = generated_corr - actual_corr
    correlation_mae = _upper_triangle_mae(actual_corr, generated_corr)
    distribution = _distribution_summary(actual, generated)
    summary = {
        "generator": GENERATOR_NAME,
        "correlation_method": method,
        "correlation_mae": correlation_mae,
        "source_row_count": int(len(actual)),
        "generated_row_count": int(len(generated)),
        "feature_columns": list(PHASE1_BLOCK_FEATURE_COLUMNS),
    }
    return {
        "summary": summary,
        "actual_correlation": actual_corr,
        "synthetic_correlation": generated_corr,
        "correlation_delta": delta_corr,
        "distribution_summary": distribution,
    }


def write_phase1_block_generation_package(
    actual_blocks: pd.DataFrame,
    output_dir: str | Path,
    n_blocks: int,
    seed: int = 2026,
    noise_ratio: float = 0.03,
    method: str = "pearson",
) -> dict[str, str]:
    """Generate block-only data and write CSV/JSON validation outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated = generate_phase1_block_data(
        actual_blocks=actual_blocks,
        n_blocks=n_blocks,
        seed=seed,
        noise_ratio=noise_ratio,
    )
    comparison = compare_phase1_block_data(actual_blocks, generated, method=method)

    synthetic_path = output_path / "phase1_synthetic_blocks.csv"
    summary_path = output_path / "summary.json"
    actual_corr_path = output_path / "actual_correlation.csv"
    synthetic_corr_path = output_path / "synthetic_correlation.csv"
    delta_corr_path = output_path / "correlation_delta.csv"
    distribution_path = output_path / "distribution_summary.csv"

    generated.to_csv(synthetic_path, index=False, encoding="utf-8-sig")
    comparison["actual_correlation"].to_csv(actual_corr_path, encoding="utf-8-sig")
    comparison["synthetic_correlation"].to_csv(synthetic_corr_path, encoding="utf-8-sig")
    comparison["correlation_delta"].to_csv(delta_corr_path, encoding="utf-8-sig")
    comparison["distribution_summary"].to_csv(distribution_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(comparison["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paths = {
        "synthetic_csv": str(synthetic_path),
        "summary_json": str(summary_path),
        "actual_correlation_csv": str(actual_corr_path),
        "synthetic_correlation_csv": str(synthetic_corr_path),
        "correlation_delta_csv": str(delta_corr_path),
        "distribution_summary_csv": str(distribution_path),
    }
    print(
        "[VALIDATION][phase1_block_data_generator.write_phase1_block_generation_package] "
        f"passed=true output_dir={output_path} correlation_mae={comparison['summary']['correlation_mae']:.6f}"
    )
    return paths


def _jitter_column(
    sampled: pd.Series,
    actual: pd.Series,
    column: str,
    rng: np.random.Generator,
    noise_ratio: float,
) -> pd.Series:
    if column == "THK":
        return sampled.astype(float).reset_index(drop=True)

    values = sampled.astype(float).to_numpy(copy=True)
    scale = float(actual.astype(float).std(ddof=0)) * noise_ratio
    if scale > 0:
        values = values + rng.normal(0, scale, size=len(values))

    lower = float(actual.min())
    upper = float(actual.max())
    values = np.clip(values, lower, upper)

    if column in {"STL_QTY", "BV_QTY"}:
        values = np.rint(values)
    if column in POSITIVE_COLUMNS:
        values = np.maximum(values, 1)
    if column in NON_NEGATIVE_COLUMNS:
        values = np.maximum(values, 0)
    return pd.Series(values)


def _upper_triangle_mae(actual_corr: pd.DataFrame, generated_corr: pd.DataFrame) -> float:
    delta = (generated_corr - actual_corr).to_numpy()
    rows, cols = np.triu_indices(len(PHASE1_BLOCK_FEATURE_COLUMNS), 1)
    values = np.abs(delta[rows, cols])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        print("[ERROR][phase1_block_data_generator._upper_triangle_mae] cause=no_finite_correlation_values")
        raise RuntimeError("no_finite_correlation_values")
    return float(values.mean())


def _distribution_summary(actual: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []
    for column in PHASE1_BLOCK_FEATURE_COLUMNS:
        actual_values = actual[column].astype(float)
        generated_values = generated[column].astype(float)
        rows.append(
            {
                "column": column,
                "actual_mean": float(actual_values.mean()),
                "generated_mean": float(generated_values.mean()),
                "actual_std": float(actual_values.std(ddof=0)),
                "generated_std": float(generated_values.std(ddof=0)),
                "actual_min": float(actual_values.min()),
                "generated_min": float(generated_values.min()),
                "actual_p50": float(actual_values.quantile(0.5)),
                "generated_p50": float(generated_values.quantile(0.5)),
                "actual_max": float(actual_values.max()),
                "generated_max": float(generated_values.max()),
            }
        )
    return pd.DataFrame(rows)
