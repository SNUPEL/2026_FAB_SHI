"""실적 Excel에서 MIXED 학습용 고정 JSON profile을 산출한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.data.multi_series_formula_data_generator import (
    DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
    DEFAULT_MULTI_SERIES_GENERATION_PROFILE,
    DEFAULT_MULTI_SERIES_WO_SOURCE,
    write_multi_series_generation_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="실적을 한 번 적합해 Phase 1/2가 읽을 고정 JSON profile을 만듭니다."
    )
    parser.add_argument("--wo-xlsx", default=str(DEFAULT_MULTI_SERIES_WO_SOURCE))
    parser.add_argument("--block-xlsx", default=str(DEFAULT_MULTI_SERIES_BLOCK_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_MULTI_SERIES_GENERATION_PROFILE))
    args = parser.parse_args()
    print("[build-multi-series-generation-profile]")
    print(f"- wo_xlsx: {args.wo_xlsx}")
    print(f"- block_xlsx: {args.block_xlsx}")
    print(f"- output: {args.output}")
    print("- scope: physical_block_joint_only (계수는 block_params/wo_params)")
    output = write_multi_series_generation_profile(
        output_path=args.output,
        wo_source_path=args.wo_xlsx,
        block_source_path=args.block_xlsx,
    )
    print(f"[VALIDATION][build_multi_series_generation_profile.main] passed=true output={output}")


if __name__ == "__main__":
    main()
