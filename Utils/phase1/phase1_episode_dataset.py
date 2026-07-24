"""MIXED Phase 1 on-the-fly episode 생성기."""

from __future__ import annotations

from typing import Dict, List

from Utils.data.report_formula_data_generator import build_report_formula_episode_jobs
from Utils.phase1.multi_series_planner import build_multi_series_phase1_training_problem


def build_phase1_episode_jobs(
    episode_count: int,
    min_blocks: int,
    max_blocks: int,
    seed: int = 2026,
    verbose: bool = True,
) -> List[Dict]:
    """물리 블록 공동분포를 보존한 MIXED episode를 메모리에 생성한다.

    ``min_blocks``/``max_blocks``는 물리 블록 수 범위다. 하나의 물리 블록이
    여러 계열을 가지면 Phase 1 decision block 수는 더 커질 수 있다.
    """

    _validate_sampling_args(episode_count, min_blocks, max_blocks)
    generated = build_report_formula_episode_jobs(
        episode_count=episode_count,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        seed=seed,
        gyel="MIXED",
    )
    episodes: List[Dict] = []
    for episode_index, spec in enumerate(generated, start=1):
        problem = build_multi_series_phase1_training_problem(spec["jobs"])
        episode_id = f"EP{episode_index:05d}"
        metadata = {
            "episode_id": episode_id,
            "problem_id": episode_id,
            "physical_block_count": int(spec["physical_block_count"]),
            "block_count": int(problem["block_count"]),
            "job_count": int(problem["job_count"]),
            "seed": int(spec["seed"]),
            "case_type": "mixed_physical_block_joint_distribution",
            "balancing_groups": list(problem["balancing_groups"]),
            "bay_ids": list(problem["bay_ids"]),
            "bay_capacity_weights": dict(problem["bay_capacity_weights"]),
            "series_combinations": list(spec["series_combinations"]),
        }
        episodes.append(
            {
                **metadata,
                "metadata": metadata,
                "blocks": spec["block_df"],
                "wo_df": spec["wo_df"],
                "jobs": problem["jobs"],
            }
        )
    if verbose:
        print(
            "[VALIDATION][phase1_episode_dataset.build_phase1_episode_jobs] "
            f"passed=true episodes={episode_count} physical_blocks={min_blocks}..{max_blocks} "
            "source=mixed_physical_block_joint_distribution"
        )
    return episodes


def _validate_sampling_args(episode_count: int, min_blocks: int, max_blocks: int) -> None:
    if episode_count <= 0 or min_blocks <= 0 or max_blocks < min_blocks:
        print(
            "[ERROR][phase1_episode_dataset._validate_sampling_args] "
            f"cause=invalid_range episodes={episode_count} min_blocks={min_blocks} max_blocks={max_blocks}"
        )
        raise RuntimeError("invalid MIXED Phase 1 episode sampling range")


__all__ = ["build_phase1_episode_jobs"]
