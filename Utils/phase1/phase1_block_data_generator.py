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
# LINE-BY-LINE: 일반 episode는 기존 bootstrap+jitter만 사용합니다.
HARD_CASE_MODE_NONE = "none"
# LINE-BY-LINE: hard episode는 강재수량은 유지하고 절단장을 섞어 강재-절단장 상관을 낮춥니다.
HARD_CASE_MODE_CUT_SHUFFLE = "cut_shuffle"
# LINE-BY-LINE: 지원하는 hard-case mode 목록입니다. 여기에 없는 값은 조용히 무시하지 않고 실패합니다.
HARD_CASE_MODES = (HARD_CASE_MODE_NONE, HARD_CASE_MODE_CUT_SHUFFLE)


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
    hard_case_mode: str = HARD_CASE_MODE_NONE,
    hard_case_target_corr: float = 0.85,
    hard_case_max_attempts: int = 20,
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
    # LINE-BY-LINE: hard_case_mode 문자열을 표준화합니다. 공백이나 대소문자 차이는 허용합니다.
    hard_mode = str(hard_case_mode).strip().lower()
    # LINE-BY-LINE: 지원하지 않는 hard-case mode는 fallback 없이 실패합니다.
    if hard_mode not in HARD_CASE_MODES:
        print(
            "[ERROR][phase1_block_data_generator.generate_phase1_block_data] "
            f"cause=unsupported_hard_case_mode mode={hard_case_mode}"
        )
        raise RuntimeError(f"unsupported_hard_case_mode: {hard_case_mode}")
    # LINE-BY-LINE: target correlation은 Pearson 상관계수 범위 안이어야 합니다.
    if not -1.0 <= hard_case_target_corr <= 1.0:
        print(
            "[ERROR][phase1_block_data_generator.generate_phase1_block_data] "
            f"cause=invalid_hard_case_target_corr target={hard_case_target_corr}"
        )
        raise RuntimeError(f"invalid_hard_case_target_corr: {hard_case_target_corr}")
    # LINE-BY-LINE: cut shuffle을 시도할 횟수는 양수여야 합니다.
    if hard_mode != HARD_CASE_MODE_NONE and hard_case_max_attempts <= 0:
        print(
            "[ERROR][phase1_block_data_generator.generate_phase1_block_data] "
            f"cause=invalid_hard_case_max_attempts attempts={hard_case_max_attempts}"
        )
        raise RuntimeError(f"invalid_hard_case_max_attempts: {hard_case_max_attempts}")

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

    # LINE-BY-LINE: hard-case 적용 전 강재수량-절단장 상관 기록용 변수입니다. 일반 episode는 빈 값으로 둡니다.
    hard_corr_before: float | None = None
    # LINE-BY-LINE: hard-case 적용 후 강재수량-절단장 상관 기록용 변수입니다. 일반 episode는 빈 값으로 둡니다.
    hard_corr_after: float | None = None
    # LINE-BY-LINE: 요청된 경우 절단장을 섞어 "강재를 맞춰도 절단장이 따로 노는" 문제를 만듭니다.
    if hard_mode == HARD_CASE_MODE_CUT_SHUFFLE:
        hard_corr_before, hard_corr_after = _apply_cut_shuffle_hard_case(
            generated=generated,
            rng=rng,
            target_corr=hard_case_target_corr,
            max_attempts=hard_case_max_attempts,
        )

    # LINE-BY-LINE: 산출물 식별용 role column입니다. 학습에는 직접 쓰지 않습니다.
    generated["DATA_ROLE"] = (
        "synthetic_phase1_hard_cut_shuffle"
        if hard_mode == HARD_CASE_MODE_CUT_SHUFFLE
        else "synthetic_phase1_block_only"
    )
    # LINE-BY-LINE: episode가 어떤 hard-case mode로 만들어졌는지 CSV/metadata에 남깁니다.
    generated["HARD_CASE_MODE"] = hard_mode
    # LINE-BY-LINE: hard 적용 전 steel-cut correlation입니다. 일반 episode는 NaN입니다.
    generated["HARD_CASE_CORR_STEEL_CUT_BEFORE"] = hard_corr_before
    # LINE-BY-LINE: hard 적용 후 steel-cut correlation입니다. 일반 episode는 NaN입니다.
    generated["HARD_CASE_CORR_STEEL_CUT_AFTER"] = hard_corr_after
    # LINE-BY-LINE: 생성 설정을 로그로 남깁니다. 학습 루프에서는 verbose=False로 episode별 중복 로그를 줄입니다.
    if verbose:
        print(
            "[CHECK][phase1_block_data_generator.generate_phase1_block_data] "
            f"generator={GENERATOR_NAME} rows={len(generated)} seed={seed} noise_ratio={noise_ratio} "
            f"hard_case_mode={hard_mode} hard_corr_before={hard_corr_before} hard_corr_after={hard_corr_after}"
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


def _apply_cut_shuffle_hard_case(
    generated: pd.DataFrame,
    rng: np.random.Generator,
    target_corr: float,
    max_attempts: int,
) -> tuple[float, float]:
    # LINE-BY-LINE: 현재 synthetic episode의 강재수량 배열입니다.
    steel_values = generated["STL_QTY"].astype(float)
    # LINE-BY-LINE: 현재 synthetic episode의 절단장 배열입니다.
    original_cut_values = generated["CUT_LTH"].astype(float).to_numpy(copy=True)
    # LINE-BY-LINE: hard-case 적용 전 강재-절단장 Pearson 상관입니다.
    corr_before = _pearson_corr(steel_values, original_cut_values)
    # LINE-BY-LINE: 이미 target 이하이면 추가 조작 없이 hard-case로 사용할 수 있습니다.
    if corr_before <= target_corr:
        return corr_before, corr_before

    # LINE-BY-LINE: 지금까지 찾은 가장 낮은 correlation입니다.
    best_corr = corr_before
    # LINE-BY-LINE: 지금까지 찾은 가장 낮은 correlation을 만든 절단장 배열입니다.
    best_cut_values = original_cut_values
    # LINE-BY-LINE: 먼저 강재수량이 작은 row에 큰 절단장을 붙이는 deterministic hard case를 만듭니다.
    reverse_rank_cut_values = original_cut_values.copy()
    # LINE-BY-LINE: 강재수량 오름차순 row index입니다. 같은 강재수량이면 원래 순서를 유지합니다.
    steel_order = np.argsort(np.asarray(steel_values, dtype=float), kind="mergesort")
    # LINE-BY-LINE: 절단장은 내림차순으로 정렬해 강재수량과 반대 방향으로 배치합니다.
    reverse_rank_cut_values[steel_order] = np.sort(original_cut_values)[::-1]
    # LINE-BY-LINE: deterministic hard case의 correlation을 계산합니다.
    reverse_rank_corr = _pearson_corr(steel_values, reverse_rank_cut_values)
    # LINE-BY-LINE: reverse-rank가 더 어려우면 best로 보관합니다.
    if reverse_rank_corr < best_corr:
        best_corr = reverse_rank_corr
        best_cut_values = reverse_rank_cut_values
    # LINE-BY-LINE: reverse-rank만으로 target 이하를 만족하면 즉시 적용합니다.
    if reverse_rank_corr <= target_corr:
        generated["CUT_LTH"] = reverse_rank_cut_values
        return corr_before, reverse_rank_corr
    # LINE-BY-LINE: max_attempts만큼 절단장 순열을 샘플링합니다. 실패하면 조용히 원본으로 가지 않고 예외를 냅니다.
    for _attempt in range(max_attempts):
        # LINE-BY-LINE: 절단장 값들의 multiset은 유지하고 row 대응만 섞습니다.
        candidate_cut_values = rng.permutation(original_cut_values)
        # LINE-BY-LINE: 이번 순열의 강재-절단장 상관을 계산합니다.
        corr_candidate = _pearson_corr(steel_values, candidate_cut_values)
        # LINE-BY-LINE: 더 어려운 episode면 현재 best로 보관합니다.
        if corr_candidate < best_corr:
            best_corr = corr_candidate
            best_cut_values = candidate_cut_values
        # LINE-BY-LINE: target correlation 이하를 만족하면 즉시 채택합니다.
        if corr_candidate <= target_corr:
            generated["CUT_LTH"] = candidate_cut_values
            return corr_before, corr_candidate

    # LINE-BY-LINE: target은 못 맞췄지만 best조차 target보다 높으면 hard-case 생성 실패로 처리합니다.
    if best_corr > target_corr:
        print(
            "[ERROR][phase1_block_data_generator._apply_cut_shuffle_hard_case] "
            f"cause=target_corr_not_reached target={target_corr} before={corr_before:.6f} "
            f"best={best_corr:.6f} attempts={max_attempts}"
        )
        raise RuntimeError("target_corr_not_reached")
    # LINE-BY-LINE: 방어적 분기입니다. 위 loop에서 return되지 않은 best target 만족값을 적용합니다.
    generated["CUT_LTH"] = best_cut_values
    # LINE-BY-LINE: 적용 전/후 correlation을 반환해 metadata에 남깁니다.
    return corr_before, best_corr


def _pearson_corr(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    # LINE-BY-LINE: pandas/numpy 입력을 float numpy 배열로 통일합니다.
    left_values = np.asarray(left, dtype=float)
    # LINE-BY-LINE: 비교 대상 배열도 float numpy 배열로 통일합니다.
    right_values = np.asarray(right, dtype=float)
    # LINE-BY-LINE: 두 배열 길이가 다르면 correlation 의미가 없으므로 실패합니다.
    if len(left_values) != len(right_values):
        print(
            "[ERROR][phase1_block_data_generator._pearson_corr] "
            f"cause=length_mismatch left={len(left_values)} right={len(right_values)}"
        )
        raise RuntimeError("length_mismatch")
    # LINE-BY-LINE: Pearson correlation은 최소 2개 값이 필요합니다.
    if len(left_values) < 2:
        print("[ERROR][phase1_block_data_generator._pearson_corr] cause=too_few_values")
        raise RuntimeError("too_few_values")
    # LINE-BY-LINE: 상수 배열은 표준편차가 0이라 correlation이 정의되지 않습니다.
    if float(np.std(left_values)) == 0.0 or float(np.std(right_values)) == 0.0:
        print("[ERROR][phase1_block_data_generator._pearson_corr] cause=zero_variance")
        raise RuntimeError("zero_variance")
    # LINE-BY-LINE: numpy corrcoef 결과의 [0,1] 위치가 두 배열의 Pearson correlation입니다.
    corr = float(np.corrcoef(left_values, right_values)[0, 1])
    # LINE-BY-LINE: NaN/inf는 hard-case 판단에 사용할 수 없으므로 실패합니다.
    if not np.isfinite(corr):
        print("[ERROR][phase1_block_data_generator._pearson_corr] cause=non_finite_corr")
        raise RuntimeError("non_finite_corr")
    # LINE-BY-LINE: 유한한 correlation 값을 반환합니다.
    return corr


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
