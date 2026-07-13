"""Variable-size Phase 1 episode dataset builder.

One episode is one block-to-Bay assignment problem:

    N blocks -> SELECT_BLOCK/SELECT_BAY repeated N times -> 2N labels

The source distribution is the validated block-level actual data. This module
does not create W/O rows; Phase 2 will need a separate W/O generator.

입출력 요약:
- synthetic 학습 episode: block-only 실제 분포에서 `N`개 block을 뽑아 Job-like 객체로 변환한다.
- actual validation episode: W/O 실적 파일에서 특정 작업일의 block 집합을 만들고 Bay 22/23/24 문제로 변환한다.
- 출력: `jobs` mapping과 metadata. 학습 함수는 이 mapping만 보고 Phase 1 pair 후보를 만든다.
"""

# LINE-BY-LINE: 미래 타입 힌트를 문자열로 늦게 평가합니다. 사용: Python annotation 호환성 유지.
from __future__ import annotations

# LINE-BY-LINE: episode trace CSV 저장에 사용합니다.
import csv
# LINE-BY-LINE: action table JSONL과 manifest JSON 저장에 사용합니다.
import json
# LINE-BY-LINE: 실적 착수시간을 작업일로 변환할 때 datetime parsing에 사용합니다.
from datetime import datetime, timedelta
# LINE-BY-LINE: 입력/출력 파일 경로를 OS 독립적으로 다루기 위해 사용합니다.
from pathlib import Path
# LINE-BY-LINE: synthetic block row를 Job-like 객체로 가볍게 감싸기 위해 사용합니다.
from types import SimpleNamespace
# LINE-BY-LINE: 함수 인자/반환 타입을 명확히 표시하기 위한 typing import입니다.
from typing import Dict, List, Mapping, Sequence

# LINE-BY-LINE: episode별 block 수와 seed를 샘플링하기 위한 numpy random generator에 사용합니다.
import numpy as np
# LINE-BY-LINE: Excel/CSV 로딩과 DataFrame 변환에 사용합니다.
import pandas as pd

from Utils.data.multi_series_cutting_data import build_block_set_id

# LINE-BY-LINE: block-only synthetic row 생성과 필수 컬럼 검증 함수를 재사용합니다.
from Utils.phase1.phase1_block_data_generator import (
    HARD_CASE_MODE_CUT_SHUFFLE,
    HARD_CASE_MODE_NONE,
    HARD_CASE_MODES,
    generate_phase1_block_data,
    validate_phase1_block_data,
)
# LINE-BY-LINE: 과거 split-action trace/action table을 만들 때 사용하는 MDP trace builder입니다.
from Utils.phase1.phase1_mdp import PHASE1_TRACE_FIELDS, build_phase1_mdp_trace_package


# LINE-BY-LINE: actual validation용 W/O 파일에서 반드시 필요한 컬럼 목록입니다. 없으면 실패합니다.
PHASE1_ACTUAL_WO_REQUIRED_COLUMNS = (
    # LINE-BY-LINE: `PROJ_NO`는 호선번호입니다. block_set_id를 만들 때 사용합니다.
    "PROJ_NO",
    # LINE-BY-LINE: `BLK_NO`는 블록명입니다. block_set_id를 만들 때 사용합니다.
    "BLK_NO",
    # LINE-BY-LINE: `WK_ORD_NO`는 W/O 명입니다. W/O 단위 row 식별에 사용합니다.
    "WK_ORD_NO",
    # LINE-BY-LINE: `GYEL`은 계열입니다. 현재 Phase 1은 NP만 사용합니다.
    "GYEL",
    # LINE-BY-LINE: `RT_CUT_ST_DTM`은 실적 착수시간입니다. 08:00 작업일 rule 계산에 사용합니다.
    "RT_CUT_ST_DTM",
    # LINE-BY-LINE: `CUT_BAY`는 실적 절단 Bay입니다. Bay 22/23/24 외 block 제거와 actual 비교에 사용합니다.
    "CUT_BAY",
    # LINE-BY-LINE: `LTH`는 길이입니다. Phase 1 pair state의 길이 feature로 사용합니다.
    "LTH",
    # LINE-BY-LINE: `THK`는 두께입니다. Phase 1 pair state의 두께 feature로 사용합니다.
    "THK",
    # LINE-BY-LINE: `STL_QTY`는 강재수량입니다. Phase 1 목적함수 1순위입니다.
    "STL_QTY",
    # LINE-BY-LINE: `CUT_LTH`는 절단장입니다. Phase 1 목적함수 2순위입니다.
    "CUT_LTH",
    # LINE-BY-LINE: `BV_QTY`는 베벨/개선 수량입니다. Phase 1 목적함수 3순위입니다.
    "BV_QTY",
)


# LINE-BY-LINE: variable-size Phase 1 episode dataset을 파일 패키지로 저장하는 entry function입니다.
def write_phase1_episode_dataset(
    actual_blocks: pd.DataFrame,
    output_dir: str | Path,
    episode_count: int,
    min_blocks: int,
    max_blocks: int,
    train_ratio: float,
    bay_ids: Sequence[str],
    algorithm: str,
    seed: int = 2026,
    noise_ratio: float = 0.03,
) -> Dict[str, str]:
    """Sample variable-size episodes and write train/test self-label tables."""

    # LINE-BY-LINE: episode 개수, block 수 범위, train ratio를 먼저 검증합니다.
    _validate_episode_args(episode_count, min_blocks, max_blocks, train_ratio)
    # LINE-BY-LINE: 실제 block 분포 데이터가 필수 컬럼/값 검증을 통과하는지 확인합니다.
    actual = validate_phase1_block_data(actual_blocks, source_name="phase1_episode_actual_blocks")
    # LINE-BY-LINE: output directory를 만들고 이후 CSV/JSONL/manifest를 저장합니다.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: 전체 episode 중 train split에 들어갈 개수를 계산합니다.
    train_episode_count = _train_episode_count(episode_count, train_ratio)
    # LINE-BY-LINE: block 수가 episode마다 달라지는 synthetic 문제들을 생성합니다.
    episodes = build_phase1_episode_jobs(
        actual_blocks=actual,
        episode_count=episode_count,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        seed=seed,
        noise_ratio=noise_ratio,
    )

    # LINE-BY-LINE: 모든 episode의 trace CSV row를 누적합니다.
    all_trace_rows: List[Dict] = []
    # LINE-BY-LINE: 모든 episode의 action JSONL row를 누적합니다.
    all_action_rows: List[Dict] = []
    # LINE-BY-LINE: 모든 episode의 synthetic block row를 누적합니다.
    all_block_rows: List[Dict] = []
    # LINE-BY-LINE: episode별 block 수와 목적함수 gap 요약 row를 누적합니다.
    episode_summaries: List[Dict] = []

    # LINE-BY-LINE: 생성된 episode를 순서대로 train/test split에 배정합니다.
    for episode_index, episode in enumerate(episodes):
        # LINE-BY-LINE: episode id는 `EP00001` 같은 문자열입니다.
        episode_id = str(episode["episode_id"])
        # LINE-BY-LINE: train_episode_count 이전 episode는 train, 이후는 test로 둡니다.
        split = "train" if episode_index < train_episode_count else "test"
        # LINE-BY-LINE: 해당 episode의 block 수입니다.
        block_count = int(episode["block_count"])
        # LINE-BY-LINE: 해당 episode 생성에 사용된 seed입니다.
        episode_seed = int(episode["seed"])
        # LINE-BY-LINE: episode의 block DataFrame입니다.
        episode_blocks = episode["blocks"]
        # LINE-BY-LINE: episode의 Job-like mapping입니다. key는 synthetic job id입니다.
        jobs = episode["jobs"]
        # LINE-BY-LINE: 기존 split-action MDP trace package를 만듭니다.
        package = build_phase1_mdp_trace_package(
            jobs=jobs,
            bay_ids=bay_ids,
            algorithm=algorithm,
        )

        # LINE-BY-LINE: block row마다 episode_id/split을 붙여 전체 block CSV에 누적합니다.
        for block_row in episode_blocks.to_dict(orient="records"):
            all_block_rows.append({"episode_id": episode_id, "split": split, **block_row})
        # LINE-BY-LINE: trace row마다 episode_id/split을 붙여 전체 trace CSV에 누적합니다.
        for trace_row in package["trace_rows"]:
            all_trace_rows.append({"episode_id": episode_id, "split": split, **trace_row})
        # LINE-BY-LINE: action row마다 episode_id/split을 붙여 전체 action JSONL에 누적합니다.
        for action_row in package["action_table_rows"]:
            all_action_rows.append({"episode_id": episode_id, "split": split, **action_row})
        # LINE-BY-LINE: plan summary에서 목적함수 gap 정보를 가져옵니다.
        summary = dict(package["plan"]["summary"])
        # LINE-BY-LINE: episode summary row를 구성합니다.
        episode_summaries.append(
            {
                "episode_id": episode_id,
                "split": split,
                "block_count": block_count,
                "action_count": len(package["action_table_rows"]),
                "seed": episode_seed,
                "steel_quantity_gap": summary["steel_quantity_gap"],
                "cut_length_gap": summary["cut_length_gap"],
                "bevel_quantity_gap": summary["bevel_quantity_gap"],
                "long_cut_bay24_count": summary["long_cut_bay24_count"],
            }
        )

    # LINE-BY-LINE: episode block CSV path입니다.
    blocks_csv = output_path / "phase1_episode_blocks.csv"
    # LINE-BY-LINE: split-action trace CSV path입니다.
    trace_csv = output_path / "phase1_episode_trace.csv"
    # LINE-BY-LINE: 전체 action table JSONL path입니다.
    all_action_path = output_path / "phase1_episode_action_table.jsonl"
    # LINE-BY-LINE: train split action table JSONL path입니다.
    train_action_path = output_path / "phase1_train_action_table.jsonl"
    # LINE-BY-LINE: test split action table JSONL path입니다.
    test_action_path = output_path / "phase1_test_action_table.jsonl"
    # LINE-BY-LINE: episode별 요약 CSV path입니다.
    episode_summary_csv = output_path / "phase1_episode_summary.csv"
    # LINE-BY-LINE: dataset manifest JSON path입니다.
    manifest_path = output_path / "phase1_episode_dataset_manifest.json"

    # LINE-BY-LINE: 전체 block row를 CSV로 저장합니다.
    pd.DataFrame(all_block_rows).to_csv(blocks_csv, index=False, encoding="utf-8-sig")
    # LINE-BY-LINE: 전체 trace row를 CSV로 저장합니다.
    _write_csv(trace_csv, ["episode_id", "split", *PHASE1_TRACE_FIELDS], all_trace_rows)
    # LINE-BY-LINE: 전체 action row를 JSONL로 저장합니다.
    _write_action_jsonl(all_action_path, all_action_rows)
    # LINE-BY-LINE: train split action row만 JSONL로 저장합니다.
    _write_action_jsonl(train_action_path, [row for row in all_action_rows if row["split"] == "train"])
    # LINE-BY-LINE: test split action row만 JSONL로 저장합니다.
    _write_action_jsonl(test_action_path, [row for row in all_action_rows if row["split"] == "test"])
    # LINE-BY-LINE: episode summary를 CSV로 저장합니다.
    _write_csv(
        episode_summary_csv,
        [
            "episode_id",
            "split",
            "block_count",
            "action_count",
            "seed",
            "steel_quantity_gap",
            "cut_length_gap",
            "bevel_quantity_gap",
            "long_cut_bay24_count",
        ],
        episode_summaries,
    )

    # LINE-BY-LINE: dataset 생성 조건과 row count를 manifest에 기록합니다.
    manifest = {
        "dataset": "phase1_variable_size_episode_dataset",
        "algorithm": algorithm,
        "bay_ids": list(bay_ids),
        "episode_count": episode_count,
        "train_episode_count": train_episode_count,
        "test_episode_count": episode_count - train_episode_count,
        "min_blocks": min_blocks,
        "max_blocks": max_blocks,
        "train_ratio": train_ratio,
        "seed": seed,
        "noise_ratio": noise_ratio,
        "episode_block_counts": [int(row["block_count"]) for row in episode_summaries],
        "total_action_count": len(all_action_rows),
        "train_action_count": sum(1 for row in all_action_rows if row["split"] == "train"),
        "test_action_count": sum(1 for row in all_action_rows if row["split"] == "test"),
        "note": "Phase 1 block-only data. No W/O synthetic rows are generated here.",
    }
    # LINE-BY-LINE: manifest JSON을 저장합니다.
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    # LINE-BY-LINE: 생성 완료 로그를 출력합니다.
    print(
        "[VALIDATION][phase1_episode_dataset.write_phase1_episode_dataset] "
        f"passed=true episodes={episode_count} total_actions={len(all_action_rows)} output_dir={output_path}"
    )
    # LINE-BY-LINE: 호출자가 생성된 파일 위치를 출력/사용할 수 있도록 path dict를 반환합니다.
    return {
        "blocks_csv": str(blocks_csv),
        "trace_csv": str(trace_csv),
        "action_table_jsonl": str(all_action_path),
        "train_action_table_jsonl": str(train_action_path),
        "test_action_table_jsonl": str(test_action_path),
        "episode_summary_csv": str(episode_summary_csv),
        "manifest_json": str(manifest_path),
    }


# LINE-BY-LINE: 학습 루프에서 사용할 variable-size episode Job-like mapping을 생성합니다.
def build_phase1_episode_jobs(
    actual_blocks: pd.DataFrame,
    episode_count: int,
    min_blocks: int,
    max_blocks: int,
    seed: int = 2026,
    noise_ratio: float = 0.03,
    hard_case_ratio: float = 0.0,
    hard_case_mode: str = HARD_CASE_MODE_CUT_SHUFFLE,
    hard_case_target_corr: float = 0.85,
    hard_case_max_attempts: int = 20,
    verbose: bool = True,
) -> List[Dict]:
    """Sample variable-size Phase 1 Job-like mappings for self-labeling."""

    # LINE-BY-LINE: episode_count와 min/max block 범위를 검증합니다.
    _validate_episode_sampling_args(episode_count, min_blocks, max_blocks)
    # LINE-BY-LINE: hard_case_ratio는 episode 중 hard-case를 뽑을 확률입니다.
    if not 0.0 <= hard_case_ratio <= 1.0:
        print(
            "[ERROR][phase1_episode_dataset.build_phase1_episode_jobs] "
            f"cause=invalid_hard_case_ratio ratio={hard_case_ratio}"
        )
        raise RuntimeError(f"invalid_hard_case_ratio: {hard_case_ratio}")
    # LINE-BY-LINE: hard_case_mode는 generator가 지원하는 mode여야 합니다.
    normalized_hard_case_mode = str(hard_case_mode).strip().lower()
    # LINE-BY-LINE: 지원하지 않는 mode는 조용히 none으로 바꾸지 않고 실패합니다.
    if normalized_hard_case_mode not in HARD_CASE_MODES:
        print(
            "[ERROR][phase1_episode_dataset.build_phase1_episode_jobs] "
            f"cause=unsupported_hard_case_mode mode={hard_case_mode}"
        )
        raise RuntimeError(f"unsupported_hard_case_mode: {hard_case_mode}")
    # LINE-BY-LINE: 실제 block 데이터 필수 컬럼/값을 검증합니다.
    actual = validate_phase1_block_data(actual_blocks, source_name="phase1_episode_actual_blocks", verbose=verbose)
    # LINE-BY-LINE: 전체 episode 샘플링을 위한 난수 generator입니다.
    rng = np.random.default_rng(seed)
    # LINE-BY-LINE: 반환할 episode dict 목록입니다.
    episodes: List[Dict] = []
    # LINE-BY-LINE: 요청된 episode 개수만큼 독립 문제를 생성합니다.
    for episode_index in range(episode_count):
        # LINE-BY-LINE: episode id를 1부터 순번으로 생성합니다.
        episode_id = f"EP{episode_index + 1:05d}"
        # LINE-BY-LINE: 이번 episode의 block 수를 min/max 범위에서 뽑습니다.
        block_count = int(rng.integers(min_blocks, max_blocks + 1))
        # LINE-BY-LINE: 이번 episode의 block bootstrap에 사용할 seed를 뽑습니다.
        episode_seed = int(rng.integers(1, 2_147_483_647))
        # LINE-BY-LINE: 이번 episode를 hard-case로 만들지 확률적으로 결정합니다.
        is_hard_case = bool(rng.random() < hard_case_ratio)
        # LINE-BY-LINE: 일반 episode면 mode를 none으로 강제하고, hard episode면 요청 mode를 적용합니다.
        episode_hard_case_mode = normalized_hard_case_mode if is_hard_case else HARD_CASE_MODE_NONE
        # LINE-BY-LINE: 실제 block 분포에서 block_count개 synthetic block을 생성합니다.
        episode_blocks = generate_phase1_block_data(
            actual_blocks=actual,
            n_blocks=block_count,
            seed=episode_seed,
            noise_ratio=noise_ratio,
            hard_case_mode=episode_hard_case_mode,
            hard_case_target_corr=hard_case_target_corr,
            hard_case_max_attempts=hard_case_max_attempts,
            verbose=verbose,
        )
        # LINE-BY-LINE: hard-case 적용 전 correlation을 episode metadata에 기록합니다.
        hard_corr_before = _episode_block_scalar(episode_blocks, "HARD_CASE_CORR_STEEL_CUT_BEFORE")
        # LINE-BY-LINE: hard-case 적용 후 correlation을 episode metadata에 기록합니다.
        hard_corr_after = _episode_block_scalar(episode_blocks, "HARD_CASE_CORR_STEEL_CUT_AFTER")
        # LINE-BY-LINE: episode metadata, block DataFrame, Job-like mapping을 하나의 dict로 묶습니다.
        episodes.append(
            {
                "episode_id": episode_id,
                "problem_id": episode_id,
                "block_count": block_count,
                "seed": episode_seed,
                "case_type": "hard_cut_shuffle" if is_hard_case else "normal",
                "hard_case_mode": episode_hard_case_mode,
                "hard_case_ratio": hard_case_ratio,
                "hard_case_corr_steel_cut_before": hard_corr_before,
                "hard_case_corr_steel_cut_after": hard_corr_after,
                "blocks": episode_blocks,
                "jobs": jobs_from_phase1_episode_blocks(episode_blocks, episode_id),
            }
        )
    # LINE-BY-LINE: 생성된 episode 수와 block 범위를 로그로 남깁니다. 학습 중 반복 호출은 verbose=False로 숨깁니다.
    if verbose:
        print(
            "[VALIDATION][phase1_episode_dataset.build_phase1_episode_jobs] "
            f"passed=true episodes={episode_count} min_blocks={min_blocks} max_blocks={max_blocks} "
            f"hard_case_ratio={hard_case_ratio}"
        )
    # LINE-BY-LINE: 학습 함수가 바로 사용할 episode 목록을 반환합니다.
    return episodes


# LINE-BY-LINE: synthetic block DataFrame을 Phase 1 학습용 Job-like mapping으로 변환하는 공개 wrapper입니다.
def jobs_from_phase1_episode_blocks(blocks: pd.DataFrame, episode_id: str) -> Dict[str, SimpleNamespace]:
    """Convert block rows to one Job-like object per block for Phase 1."""

    # LINE-BY-LINE: 실제 변환 로직은 내부 helper에 위임합니다.
    return _jobs_from_episode_blocks(blocks, episode_id)


def _episode_block_scalar(blocks: pd.DataFrame, column: str) -> float | None:
    # LINE-BY-LINE: hard-case metadata column이 없으면 caller가 잘못 연결한 것이므로 실패합니다.
    if column not in blocks.columns:
        print(
            "[ERROR][phase1_episode_dataset._episode_block_scalar] "
            f"cause=missing_metadata_column column={column}"
        )
        raise RuntimeError(f"missing_metadata_column: {column}")
    # LINE-BY-LINE: column은 episode 전체에 동일한 값으로 채워져 있으므로 첫 row 값을 사용합니다.
    value = blocks[column].iloc[0]
    # LINE-BY-LINE: 일반 episode는 NaN이므로 None으로 정리해 JSON/CSV metadata에서 의미를 분명히 합니다.
    if pd.isna(value):
        return None
    # LINE-BY-LINE: hard episode의 correlation 값은 float으로 반환합니다.
    return float(value)


# LINE-BY-LINE: 실제 W/O 실적 파일에서 특정 작업일들을 Phase 1 actual validation 문제로 변환합니다.
def build_phase1_actual_workday_jobs(
    source_path: str | Path,
    workdays: Sequence[str],
    bay_ids: Sequence[str],
    gyel: str | None = "NP",
    require_complete_block_workday: bool = False,
) -> List[Dict]:
    """Build fixed actual Phase 1 validation problems from W/O rows.

    Workday rule: `RT_CUT_ST_DTM` 08:00 through next-day 07:59 belongs to
    the same planning workday. Blocks containing any Bay outside `bay_ids`
    are removed so Phase 1 validation targets only Bay 22/23/24 problems.
    """

    # LINE-BY-LINE: 입력 W/O 파일 경로를 Path 객체로 변환합니다.
    source = Path(source_path)
    # LINE-BY-LINE: 허용 Bay를 문자열 tuple로 고정합니다. 현재 validation 기준은 22/23/24입니다.
    allowed_bays = tuple(str(bay_id) for bay_id in bay_ids)
    # LINE-BY-LINE: 공백 workday 값은 제거합니다.
    requested_workdays = [str(workday).strip() for workday in workdays if str(workday).strip()]
    # LINE-BY-LINE: actual validation은 최소 1개 작업일이 필요합니다.
    if not requested_workdays:
        print("[ERROR][phase1_episode_dataset.build_phase1_actual_workday_jobs] cause=no_workdays")
        raise RuntimeError("actual validation workdays are required")
    raw = _read_phase1_actual_wo(source)
    _require_actual_wo_columns(raw, source)
    if require_complete_block_workday and "RT_CUT_ED_DTM" not in raw.columns:
        print(
            "[ERROR][phase1_episode_dataset.build_phase1_actual_workday_jobs] "
            f"cause=missing_required_columns missing=['RT_CUT_ED_DTM'] path={source}"
        )
        raise RuntimeError("missing_required_columns: RT_CUT_ED_DTM")
    df = raw.copy()
    if gyel is not None:
        before = len(df)
        df = df[df["GYEL"].astype(str).eq(str(gyel))].copy()
        print(
            "[CHECK][phase1_episode_dataset.build_phase1_actual_workday_jobs] "
            f"gyel={gyel} rows_before={before} rows_after={len(df)}"
        )
        if df.empty:
            print(
                "[ERROR][phase1_episode_dataset.build_phase1_actual_workday_jobs] "
                f"cause=no_rows_after_gyel_filter gyel={gyel} path={source}"
            )
            raise RuntimeError(f"no_rows_after_gyel_filter: {gyel}")

    df["__block_key"] = df.apply(
        lambda row: build_block_set_id(row["PROJ_NO"], row["GYEL"], row["BLK_NO"]),
        axis=1,
    )
    df["__cut_bay"] = df["CUT_BAY"].map(_normalize_bay_value)
    outside = df[~df["__cut_bay"].isin(allowed_bays)]
    outside_blocks = set(outside["__block_key"])
    if outside_blocks:
        df = df[~df["__block_key"].isin(outside_blocks)].copy()
    invalid_datetime_examples: List[str] = []
    workday_values: List[str | None] = []
    for value in df["RT_CUT_ST_DTM"]:
        workday_value = _actual_workday_or_none(value)
        if workday_value is None and len(invalid_datetime_examples) < 5:
            invalid_datetime_examples.append(str(value))
        workday_values.append(workday_value)
    df["__workday"] = workday_values
    invalid_datetime_count = int(pd.isna(df["__workday"]).sum())
    if invalid_datetime_count:
        print(
            "[CHECK][phase1_episode_dataset.build_phase1_actual_workday_jobs] "
            f"excluded_invalid_actual_start_rows={invalid_datetime_count} "
            f"examples={invalid_datetime_examples}"
        )
        df = df[df["__workday"].notna()].copy()
    if require_complete_block_workday:
        end_workday_values: List[str | None] = []
        invalid_end_examples: List[str] = []
        for value in df["RT_CUT_ED_DTM"]:
            end_workday_value = _actual_workday_or_none(value)
            if end_workday_value is None and len(invalid_end_examples) < 5:
                invalid_end_examples.append(str(value))
            end_workday_values.append(end_workday_value)
        df["__end_workday"] = end_workday_values
        invalid_end_count = int(pd.isna(df["__end_workday"]).sum())
        if invalid_end_count:
            print(
                "[CHECK][phase1_episode_dataset.build_phase1_actual_workday_jobs] "
                f"excluded_invalid_actual_end_rows={invalid_end_count} "
                f"examples={invalid_end_examples}"
            )
            df = df[df["__end_workday"].notna()].copy()
    if df.empty:
        print("[ERROR][phase1_episode_dataset.build_phase1_actual_workday_jobs] cause=no_rows_after_datetime_filter")
        raise RuntimeError("no_rows_after_datetime_filter")
    _coerce_actual_numeric_columns(df, source)

    payloads: List[Dict] = []
    for workday in requested_workdays:
        if require_complete_block_workday:
            block_workdays = df.groupby("__block_key").agg(
                start_workday_count=("__workday", "nunique"),
                end_workday_count=("__end_workday", "nunique"),
                first_start_workday=("__workday", "first"),
                first_end_workday=("__end_workday", "first"),
            )
            complete_blocks = block_workdays[
                (block_workdays["start_workday_count"].eq(1))
                & (block_workdays["end_workday_count"].eq(1))
                & (block_workdays["first_start_workday"].eq(workday))
                & (block_workdays["first_end_workday"].eq(workday))
            ].index
            subset = df[df["__block_key"].isin(complete_blocks)].copy()
            print(
                "[CHECK][phase1_episode_dataset.build_phase1_actual_workday_jobs] "
                f"workday={workday} complete_block_count={len(complete_blocks)} "
                f"rows={len(subset)} mode=complete_block_workday"
            )
        else:
            subset = df[df["__workday"].eq(workday)].copy()
        if subset.empty:
            print(
                "[ERROR][phase1_episode_dataset.build_phase1_actual_workday_jobs] "
                f"cause=no_rows_for_workday workday={workday} path={source}"
            )
            raise RuntimeError(f"no_rows_for_workday: {workday}")
        jobs = _jobs_from_actual_workday_rows(subset, workday)
        payloads.append(
            {
                "episode_id": f"ACTUAL_{workday}",
                "problem_id": f"ACTUAL_{workday}",
                "workday": workday,
                "block_count": len({job.block_set_id for job in jobs.values()}),
                "job_count": len(jobs),
                "excluded_outside_bay_block_count": len(outside_blocks),
                "jobs": jobs,
                "metadata": {
                    "episode_id": f"ACTUAL_{workday}",
                    "problem_id": f"ACTUAL_{workday}",
                    "workday": workday,
                    "block_count": len({job.block_set_id for job in jobs.values()}),
                    "job_count": len(jobs),
                    "validation_source": "actual_8days",
                    "evaluation_input_type": "wo_actual_workday",
                    "source_xlsx": str(source),
                    "require_complete_block_workday": int(require_complete_block_workday),
                    "seed": "",
                },
            }
        )
    print(
        "[VALIDATION][phase1_episode_dataset.build_phase1_actual_workday_jobs] "
        f"passed=true workdays={len(payloads)} excluded_outside_bay_blocks={len(outside_blocks)}"
    )
    return payloads


def build_phase1_candidate_workbook_jobs(
    candidate_path: str | Path,
    workdays: Sequence[str],
    bay_ids: Sequence[str],
) -> List[Dict]:
    """Build fixed actual-8days Phase 1 validation problems from candidate block workbook.

    The presentation/evaluation actual_8days problems must come from
    `착수일 후보 블록.xlsx`, not from W/O 08:00 workday reconstruction.
    Each `{YYYYMMDD}_BLK` sheet already defines the block universe for that
    validation day, so one sheet row becomes one Phase 1 block/job.
    """

    source = Path(candidate_path)
    allowed_bays = tuple(str(bay_id) for bay_id in bay_ids)
    requested_workdays = [str(workday).strip() for workday in workdays if str(workday).strip()]
    if not requested_workdays:
        print("[ERROR][phase1_episode_dataset.build_phase1_candidate_workbook_jobs] cause=no_workdays")
        raise RuntimeError("candidate workbook validation workdays are required")
    if not source.exists():
        print(
            "[ERROR][phase1_episode_dataset.build_phase1_candidate_workbook_jobs] "
            f"cause=missing_candidate_xlsx path={source}"
        )
        raise RuntimeError(f"missing candidate workbook: {source}")

    payloads: List[Dict] = []
    with pd.ExcelFile(source) as workbook:
        for workday in requested_workdays:
            sheet = f"{workday}_BLK"
            if sheet not in workbook.sheet_names:
                print(
                    "[ERROR][phase1_episode_dataset.build_phase1_candidate_workbook_jobs] "
                    f"cause=missing_sheet sheet={sheet} path={source}"
                )
                raise RuntimeError(f"missing sheet: {sheet}")
            frame = pd.read_excel(workbook, sheet_name=sheet).dropna(how="all").copy()
            jobs = _jobs_from_candidate_block_sheet(frame=frame, workday=str(workday), bay_ids=allowed_bays)
            payloads.append(
                {
                    "episode_id": f"ACTUAL_{workday}",
                    "problem_id": f"ACTUAL_{workday}",
                    "workday": str(workday),
                    "block_count": len(jobs),
                    "job_count": len(jobs),
                    "jobs": jobs,
                    "metadata": {
                        "episode_id": f"ACTUAL_{workday}",
                        "problem_id": f"ACTUAL_{workday}",
                        "workday": str(workday),
                        "block_count": len(jobs),
                        "job_count": len(jobs),
                        "validation_source": "actual_8days",
                        "evaluation_input_type": "candidate_workbook",
                        "candidate_xlsx": str(source),
                        "seed": "",
                    },
                }
            )
    print(
        "[VALIDATION][phase1_episode_dataset.build_phase1_candidate_workbook_jobs] "
        f"passed=true workdays={len(payloads)} path={source} input_type=candidate_workbook"
    )
    return payloads


def _jobs_from_episode_blocks(blocks: pd.DataFrame, episode_id: str) -> Dict[str, SimpleNamespace]:
    """Convert block rows to one Job-like object per block for Phase 1."""

    jobs: Dict[str, SimpleNamespace] = {}
    for row_index, row in blocks.reset_index(drop=True).iterrows():
        block_no = str(row.get("BLK_NO") or row.get("BLK_ID") or f"BLK_{row_index + 1:05d}")
        series = str(row["GYEL"]).strip()
        synthetic_block_no = f"{block_no}_{row_index + 1:05d}"
        block_set_id = build_block_set_id(episode_id, series, synthetic_block_no)
        job_id = f"{episode_id}_JOB_{row_index + 1:05d}"
        jobs[job_id] = SimpleNamespace(
            job_id=job_id,
            family=series,
            block_set_id=block_set_id,
            steel_quantity=int(row["STL_QTY"]),
            plate_length=float(row["LTH"]),
            thickness=float(row["THK"]),
            cut_length=float(row["CUT_LTH"]),
            bevel_quantity=int(row["BV_QTY"]),
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=(),
            extra={
                "source_project_no": episode_id,
                "source_block_no": block_no,
                "source_series": series,
                "source_wk_ord_no": job_id,
            },
        )
    if not jobs:
        print(f"[ERROR][phase1_episode_dataset._jobs_from_episode_blocks] cause=no_jobs episode_id={episode_id}")
        raise RuntimeError(f"episode has no blocks: {episode_id}")
    return jobs


def _jobs_from_candidate_block_sheet(frame: pd.DataFrame, workday: str, bay_ids: Sequence[str]) -> Dict[str, SimpleNamespace]:
    """Convert one `{workday}_BLK` sheet to one Job-like object per block."""

    required = {"PROJ_NO", "GYEL", "BLK_NO", "LTH", "THK", "CUT_LTH", "STL_QTY", "BV_QTY", "CUT_BAY"}
    missing = sorted(required - set(frame.columns))
    if missing:
        print(
            "[ERROR][phase1_episode_dataset._jobs_from_candidate_block_sheet] "
            f"cause=missing_required_columns workday={workday} missing={missing}"
        )
        raise RuntimeError(f"missing candidate workbook columns: {missing}")
    allowed_bays = {str(bay_id) for bay_id in bay_ids}
    jobs: Dict[str, SimpleNamespace] = {}
    seen_blocks: set[str] = set()
    for row_index, row in frame.reset_index(drop=True).iterrows():
        project_no = _required_cell_text(row["PROJ_NO"], "PROJ_NO", workday, row_index)
        series = _required_cell_text(row["GYEL"], "GYEL", workday, row_index)
        block_no = _required_cell_text(row["BLK_NO"], "BLK_NO", workday, row_index)
        block_set_id = build_block_set_id(project_no, series, block_no)
        if block_set_id in seen_blocks:
            print(
                "[ERROR][phase1_episode_dataset._jobs_from_candidate_block_sheet] "
                f"cause=duplicate_block workday={workday} block_set_id={block_set_id}"
            )
            raise RuntimeError(f"duplicate block in candidate sheet: {block_set_id}")
        seen_blocks.add(block_set_id)

        source_bay = _normalize_bay_value(row["CUT_BAY"])
        if source_bay not in allowed_bays:
            print(
                "[ERROR][phase1_episode_dataset._jobs_from_candidate_block_sheet] "
                f"cause=bay_outside_scope workday={workday} block_set_id={block_set_id} bay={source_bay}"
            )
            raise RuntimeError(f"candidate workbook Bay outside scope: {source_bay}")

        job_id = f"{workday}_BLK_{row_index + 1:05d}"
        jobs[job_id] = SimpleNamespace(
            job_id=job_id,
            family=series,
            block_set_id=block_set_id,
            steel_quantity=_required_int(row["STL_QTY"], "STL_QTY", workday, row_index),
            plate_length=_required_float(row["LTH"], "LTH", workday, row_index),
            thickness=_required_float(row["THK"], "THK", workday, row_index),
            cut_length=_required_float(row["CUT_LTH"], "CUT_LTH", workday, row_index),
            bevel_quantity=_required_int(row["BV_QTY"], "BV_QTY", workday, row_index),
            cut_bay=None,
            source_cut_bay=source_bay,
            allowed_bay_ids=(),
            extra={
                "source_project_no": project_no,
                "source_block_no": block_no,
                "source_series": series,
                "source_wk_ord_no": job_id,
                "source_workday": workday,
                "evaluation_input_type": "candidate_workbook",
            },
        )
    if not jobs:
        print(f"[ERROR][phase1_episode_dataset._jobs_from_candidate_block_sheet] cause=no_blocks workday={workday}")
        raise RuntimeError(f"no blocks in candidate sheet: {workday}")
    return jobs


def _read_phase1_actual_wo(path: Path) -> pd.DataFrame:
    """Read W/O actual rows for fixed Phase 1 validation."""

    print(f"[CHECK][phase1_episode_dataset._read_phase1_actual_wo] path={path}")
    if not path.exists():
        print(f"[ERROR][phase1_episode_dataset._read_phase1_actual_wo] cause=missing_source_file path={path}")
        raise RuntimeError(f"missing_source_file: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    print(
        "[ERROR][phase1_episode_dataset._read_phase1_actual_wo] "
        f"cause=unsupported_source_suffix suffix={path.suffix}"
    )
    raise RuntimeError(f"unsupported_source_suffix: {path.suffix}")


def _require_actual_wo_columns(df: pd.DataFrame, source: Path) -> None:
    """Fail fast when actual validation source misses required W/O columns."""

    missing = [column for column in PHASE1_ACTUAL_WO_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        print(
            "[ERROR][phase1_episode_dataset._require_actual_wo_columns] "
            f"cause=missing_required_columns missing={missing} path={source}"
        )
        raise RuntimeError(f"missing_required_columns: {','.join(missing)}")


def _required_cell_text(value: object, column: str, workday: str, row_index: int) -> str:
    """Return a non-empty text cell or fail with row/column context."""

    if pd.isna(value) or not str(value).strip():
        print(
            "[ERROR][phase1_episode_dataset._required_cell_text] "
            f"cause=blank_value workday={workday} row={row_index} column={column}"
        )
        raise RuntimeError(f"blank {column} at {workday} row {row_index}")
    return str(value).strip()


def _required_float(value: object, column: str, workday: str, row_index: int) -> float:
    """Return a numeric cell as float or fail; no default value is allowed."""

    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        print(
            "[ERROR][phase1_episode_dataset._required_float] "
            f"cause=non_numeric_or_missing workday={workday} row={row_index} column={column} value={value}"
        )
        raise RuntimeError(f"non_numeric_or_missing {column} at {workday} row {row_index}")
    return float(number)


def _required_int(value: object, column: str, workday: str, row_index: int) -> int:
    """Return a numeric cell as int when it is integer-like; no rounding fallback."""

    number = _required_float(value, column, workday, row_index)
    if not float(number).is_integer():
        print(
            "[ERROR][phase1_episode_dataset._required_int] "
            f"cause=non_integer_value workday={workday} row={row_index} column={column} value={value}"
        )
        raise RuntimeError(f"non_integer_value {column} at {workday} row {row_index}")
    return int(number)


def _normalize_bay_value(value: object) -> str:
    """Normalize Excel Bay values like 22.0 to '22' without defaulting."""

    if pd.isna(value):
        print("[ERROR][phase1_episode_dataset._normalize_bay_value] cause=missing_cut_bay")
        raise RuntimeError("missing_cut_bay")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text:
        print("[ERROR][phase1_episode_dataset._normalize_bay_value] cause=empty_cut_bay")
        raise RuntimeError("empty_cut_bay")
    return text


def _actual_workday_from_value(value: object) -> str:
    """Return planning workday from compact `YYYYMMDDHHMM` actual start."""

    text = _compact_datetime_text(value)
    try:
        started_at = datetime.strptime(text, "%Y%m%d%H%M")
    except ValueError as exc:
        print(
            "[ERROR][phase1_episode_dataset._actual_workday_from_value] "
            f"cause=invalid_actual_start_datetime value={value}"
        )
        raise RuntimeError(f"invalid_actual_start_datetime: {value}") from exc
    if started_at.hour < 8:
        started_at = started_at - timedelta(days=1)
    return started_at.strftime("%Y%m%d")


def _actual_workday_or_none(value: object) -> str | None:
    """Parse actual workday; return None only so caller can exclude and report."""

    text = _compact_datetime_text_or_none(value)
    if text is None:
        return None
    try:
        started_at = datetime.strptime(text, "%Y%m%d%H%M")
    except ValueError:
        return None
    if started_at.hour < 8:
        started_at = started_at - timedelta(days=1)
    return started_at.strftime("%Y%m%d")


def _compact_datetime_text(value: object) -> str:
    """Convert compact Excel datetime values to a strict 12-digit string."""

    if pd.isna(value):
        print("[ERROR][phase1_episode_dataset._compact_datetime_text] cause=missing_datetime")
        raise RuntimeError("missing_datetime")
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) != 12 or not text.isdigit():
        print(
            "[ERROR][phase1_episode_dataset._compact_datetime_text] "
            f"cause=invalid_compact_datetime value={value} parsed={text}"
        )
        raise RuntimeError(f"invalid_compact_datetime: {value}")
    return text


def _compact_datetime_text_or_none(value: object) -> str | None:
    """Return compact datetime text for exclusion-mode parsing."""

    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) != 12 or not text.isdigit():
        return None
    return text


def _coerce_actual_numeric_columns(df: pd.DataFrame, source: Path) -> None:
    """Validate numeric Phase 1 workload columns in-place."""

    for column in ("LTH", "THK", "STL_QTY", "CUT_LTH", "BV_QTY"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        missing = int(df[column].isna().sum())
        if missing:
            print(
                "[ERROR][phase1_episode_dataset._coerce_actual_numeric_columns] "
                f"cause=non_numeric_or_missing column={column} count={missing} path={source}"
            )
            raise RuntimeError(f"non_numeric_or_missing: {column}")
    non_positive_steel = int((df["STL_QTY"] <= 0).sum())
    if non_positive_steel:
        print(
            "[ERROR][phase1_episode_dataset._coerce_actual_numeric_columns] "
            f"cause=non_positive_steel_quantity count={non_positive_steel} path={source}"
        )
        raise RuntimeError("non_positive_steel_quantity")
    for column in ("LTH", "THK", "CUT_LTH", "BV_QTY"):
        negative = int((df[column] < 0).sum())
        if negative:
            print(
                "[ERROR][phase1_episode_dataset._coerce_actual_numeric_columns] "
                f"cause=negative_value column={column} count={negative} path={source}"
            )
            raise RuntimeError(f"negative_value: {column}")


def _jobs_from_actual_workday_rows(rows: pd.DataFrame, workday: str) -> Dict[str, SimpleNamespace]:
    """Convert W/O rows from one actual workday into Phase 1 Job-like rows."""

    jobs: Dict[str, SimpleNamespace] = {}
    for index, row in rows.sort_values(["PROJ_NO", "GYEL", "BLK_NO", "WK_ORD_NO"]).reset_index(drop=True).iterrows():
        job_id = str(row["WK_ORD_NO"]).strip()
        if not job_id:
            print(
                "[ERROR][phase1_episode_dataset._jobs_from_actual_workday_rows] "
                f"cause=empty_wk_ord_no workday={workday} row_index={index}"
            )
            raise RuntimeError("empty_wk_ord_no")
        if job_id in jobs:
            print(
                "[ERROR][phase1_episode_dataset._jobs_from_actual_workday_rows] "
                f"cause=duplicate_wk_ord_no workday={workday} job_id={job_id}"
            )
            raise RuntimeError(f"duplicate_wk_ord_no: {job_id}")
        project_no = str(row["PROJ_NO"]).strip()
        series = str(row["GYEL"]).strip()
        block_no = str(row["BLK_NO"]).strip()
        block_set_id = build_block_set_id(project_no, series, block_no)
        jobs[job_id] = SimpleNamespace(
            job_id=job_id,
            family=series,
            block_set_id=block_set_id,
            steel_quantity=int(row["STL_QTY"]),
            plate_length=float(row["LTH"]),
            thickness=float(row["THK"]),
            cut_length=float(row["CUT_LTH"]),
            bevel_quantity=int(row["BV_QTY"]),
            cut_bay=None,
            source_cut_bay=str(row["__cut_bay"]),
            allowed_bay_ids=(),
            extra={
                "source_project_no": project_no,
                "source_block_no": block_no,
                "source_series": series,
                "source_wk_ord_no": job_id,
                "source_workday": workday,
            },
        )
    if not jobs:
        print(f"[ERROR][phase1_episode_dataset._jobs_from_actual_workday_rows] cause=no_jobs workday={workday}")
        raise RuntimeError(f"actual workday has no jobs: {workday}")
    return jobs


def _validate_episode_args(episode_count: int, min_blocks: int, max_blocks: int, train_ratio: float) -> None:
    """Validate dataset sampling controls."""

    _validate_episode_sampling_args(episode_count, min_blocks, max_blocks)
    if episode_count < 2:
        print(
            "[ERROR][phase1_episode_dataset._validate_episode_args] "
            f"cause=episode_count_too_small episode_count={episode_count}"
        )
        raise RuntimeError("episode_count must be at least 2 for train/test split")
    if not 0 < train_ratio < 1:
        print(
            "[ERROR][phase1_episode_dataset._validate_episode_args] "
            f"cause=invalid_train_ratio train_ratio={train_ratio}"
        )
        raise RuntimeError("train_ratio must be between 0 and 1")


def _validate_episode_sampling_args(episode_count: int, min_blocks: int, max_blocks: int) -> None:
    """Validate controls shared by dataset writing and sampled self-labeling."""

    if episode_count <= 0:
        print(
            "[ERROR][phase1_episode_dataset._validate_episode_sampling_args] "
            f"cause=non_positive_episode_count episode_count={episode_count}"
        )
        raise RuntimeError("episode_count must be positive")
    if min_blocks <= 0 or max_blocks < min_blocks:
        print(
            "[ERROR][phase1_episode_dataset._validate_episode_sampling_args] "
            f"cause=invalid_block_range min_blocks={min_blocks} max_blocks={max_blocks}"
        )
        raise RuntimeError("invalid Phase 1 episode block range")


def _train_episode_count(episode_count: int, train_ratio: float) -> int:
    """Return train count while preserving at least one holdout episode."""

    return max(1, min(episode_count - 1, int(round(episode_count * train_ratio))))


def _write_action_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    """Write action examples as JSONL."""

    if not rows:
        print(f"[ERROR][phase1_episode_dataset._write_action_jsonl] cause=no_rows path={path}")
        raise RuntimeError(f"cannot write empty action table: {path}")
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping]) -> None:
    """Write CSV with explicit field order."""

    if not rows:
        print(f"[ERROR][phase1_episode_dataset._write_csv] cause=no_rows path={path}")
        raise RuntimeError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
