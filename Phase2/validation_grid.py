"""Phase 2의 고정 물리 블록 크기×분포 Type validation 문제를 만든다."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


_DISTRIBUTION_FIELDS = (
    "np_wo_ratio",
    "nc_wo_ratio",
    "fn_wo_ratio",
    "fl_wo_ratio",
    "wo_per_block",
    "cut_per_wo",
    "bevel_per_wo",
    "tact_per_wo",
)


@dataclass(frozen=True)
class Phase2ValidationProblem:
    """한 번 생성한 뒤 모든 validation checkpoint에서 재사용하는 상위 문제."""

    problem_id: str
    block_count: int
    distribution_type: int
    generation_seed: int
    target_distribution: Mapping[str, float]
    normalized_distribution: Mapping[str, float]
    actual_distribution: Mapping[str, float]
    jobs: Mapping[str, object]
    metadata: Mapping[str, object]


def build_phase2_validation_grid(
    problem_factory: Callable[[int, int], Mapping[str, object]],
    min_blocks: int,
    max_blocks: int,
    block_gap: int,
    type_count: int,
    seed: int,
) -> tuple[Phase2ValidationProblem, ...]:
    """크기별로 동일한 정규화 분포 목표를 갖는 Type 문제를 고정 생성한다."""

    block_sizes = _inclusive_block_sizes(min_blocks, max_blocks, block_gap)
    if type_count <= 0:
        print(
            "[ERROR][Phase2.validation_grid.build_phase2_validation_grid] "
            f"cause=invalid_type_count value={type_count}"
        )
        raise RuntimeError("validation type_count must be positive")

    targets = _distribution_targets(type_count, seed)
    candidate_count = max(12, type_count * 3)
    problems: list[Phase2ValidationProblem] = []
    for size_index, block_count in enumerate(block_sizes):
        candidates = [
            _build_candidate(
                problem_factory,
                block_count,
                seed + (size_index + 1) * 1_000_003 + (candidate_index + 1) * 104_729,
            )
            for candidate_index in range(candidate_count)
        ]
        ranked_profiles = _rank_normalize_profiles(
            [candidate["actual_distribution"] for candidate in candidates]
        )
        unused = set(range(candidate_count))
        for distribution_type, target in enumerate(targets, start=1):
            candidate_index = min(
                unused,
                key=lambda index: (
                    _profile_distance(ranked_profiles[index], target),
                    int(candidates[index]["generation_seed"]),
                ),
            )
            unused.remove(candidate_index)
            candidate = candidates[candidate_index]
            problem_id = f"B{block_count:03d}_T{distribution_type:02d}"
            metadata = dict(candidate["metadata"])
            metadata.update(
                {
                    "problem_id": problem_id,
                    "physical_block_count": block_count,
                    "distribution_type": distribution_type,
                    "generation_seed": candidate["generation_seed"],
                    "distribution_distance": _profile_distance(
                        ranked_profiles[candidate_index],
                        target,
                    ),
                }
            )
            problems.append(
                Phase2ValidationProblem(
                    problem_id=problem_id,
                    block_count=block_count,
                    distribution_type=distribution_type,
                    generation_seed=int(candidate["generation_seed"]),
                    target_distribution=target,
                    normalized_distribution=ranked_profiles[candidate_index],
                    actual_distribution=candidate["actual_distribution"],
                    jobs=candidate["jobs"],
                    metadata=metadata,
                )
            )

    print(
        "[VALIDATION][Phase2.validation_grid.build_phase2_validation_grid] "
        f"passed=true block_sizes={list(block_sizes)} types={type_count} "
        f"problems={len(problems)} candidates_per_size={candidate_count}"
    )
    return tuple(problems)


def phase2_validation_grid_contract(
    problems: Sequence[Phase2ValidationProblem],
) -> dict[str, object]:
    """Checkpoint resume 시 비교할 validation 문제 계약을 반환한다."""

    if not problems:
        return {"schema": "phase2_validation_grid_v1", "problems": []}
    return {
        "schema": "phase2_validation_grid_v1",
        "problems": [
            {
                "problem_id": problem.problem_id,
                "block_count": problem.block_count,
                "distribution_type": problem.distribution_type,
                "generation_seed": problem.generation_seed,
                "target_distribution": dict(problem.target_distribution),
                "normalized_distribution": dict(problem.normalized_distribution),
                "actual_distribution": dict(problem.actual_distribution),
                "jobs_sha256": _jobs_sha256(problem.jobs),
            }
            for problem in problems
        ],
    }


def _jobs_sha256(jobs: Mapping[str, object]) -> str:
    payload = {
        str(job_id): _canonical_contract_value(vars(job))
        for job_id, job in sorted(jobs.items())
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_contract_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_contract_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_contract_value(item) for item in value]
    print(
        "[ERROR][Phase2.validation_grid._canonical_contract_value] "
        f"cause=unsupported_type type={type(value).__name__}"
    )
    raise RuntimeError("validation Job contains a non-serializable contract value")


def _inclusive_block_sizes(
    min_blocks: int,
    max_blocks: int,
    block_gap: int,
) -> tuple[int, ...]:
    if min_blocks <= 0 or max_blocks < min_blocks or block_gap <= 0:
        print(
            "[ERROR][Phase2.validation_grid._inclusive_block_sizes] "
            f"cause=invalid_range min={min_blocks} max={max_blocks} gap={block_gap}"
        )
        raise RuntimeError("invalid Phase 2 validation block range")
    if (max_blocks - min_blocks) % block_gap != 0:
        print(
            "[ERROR][Phase2.validation_grid._inclusive_block_sizes] "
            f"cause=non_divisible_range min={min_blocks} max={max_blocks} gap={block_gap}"
        )
        raise RuntimeError("validation block range must include max_blocks exactly")
    return tuple(range(min_blocks, max_blocks + 1, block_gap))


def _distribution_targets(
    type_count: int,
    seed: int,
) -> tuple[dict[str, float], ...]:
    """Latin-hypercube 순위 목표로 Type별 분포 방향을 고정한다."""

    targets = [dict() for _ in range(type_count)]
    for field_index, field in enumerate(_DISTRIBUTION_FIELDS):
        slots = list(range(type_count))
        random.Random(seed + (field_index + 1) * 65_537).shuffle(slots)
        for type_index, slot in enumerate(slots):
            targets[type_index][field] = (slot + 0.5) / type_count
    return tuple(targets)


def _build_candidate(
    problem_factory: Callable[[int, int], Mapping[str, object]],
    block_count: int,
    generation_seed: int,
) -> dict[str, object]:
    payload = problem_factory(block_count, generation_seed)
    jobs = payload.get("jobs")
    metadata = payload.get("metadata")
    if not isinstance(jobs, Mapping) or not jobs:
        print(
            "[ERROR][Phase2.validation_grid._build_candidate] "
            f"cause=invalid_jobs block_count={block_count} seed={generation_seed}"
        )
        raise RuntimeError("validation problem factory must return non-empty jobs")
    if not isinstance(metadata, Mapping):
        print(
            "[ERROR][Phase2.validation_grid._build_candidate] "
            f"cause=invalid_metadata block_count={block_count} seed={generation_seed}"
        )
        raise RuntimeError("validation problem factory must return metadata")
    actual_count = int(metadata.get("physical_block_count", -1))
    if actual_count != block_count:
        print(
            "[ERROR][Phase2.validation_grid._build_candidate] "
            f"cause=physical_block_count_mismatch expected={block_count} actual={actual_count} "
            f"seed={generation_seed}"
        )
        raise RuntimeError("validation generator did not preserve requested block count")
    return {
        "generation_seed": generation_seed,
        "jobs": jobs,
        "metadata": metadata,
        "actual_distribution": _distribution_profile(jobs, block_count),
    }


def _distribution_profile(
    jobs: Mapping[str, object],
    block_count: int,
) -> dict[str, float]:
    family_counts = {family: 0 for family in ("NP", "NC", "FN", "FL")}
    cut_sum = 0.0
    bevel_sum = 0.0
    tact_sum = 0.0
    for job_id, job in jobs.items():
        family = str(getattr(job, "family", "")).upper()
        if family not in family_counts:
            print(
                "[ERROR][Phase2.validation_grid._distribution_profile] "
                f"cause=unsupported_family job_id={job_id} family={family}"
            )
            raise RuntimeError("validation grid contains unsupported job family")
        family_counts[family] += 1
        cut_sum += _required_non_negative(job, "cut_length", job_id)
        bevel_sum += _required_non_negative(job, "bevel_quantity", job_id)
        stage_minutes = getattr(job, "base_stage_minutes", None)
        if not isinstance(stage_minutes, Mapping) or "cut" not in stage_minutes:
            print(
                "[ERROR][Phase2.validation_grid._distribution_profile] "
                f"cause=missing_cut_stage_minutes job_id={job_id}"
            )
            raise RuntimeError("validation job requires base_stage_minutes['cut']")
        tact_sum += _non_negative_number(stage_minutes["cut"], "base_stage_minutes.cut", job_id)
    job_count = len(jobs)
    return {
        "np_wo_ratio": family_counts["NP"] / job_count,
        "nc_wo_ratio": family_counts["NC"] / job_count,
        "fn_wo_ratio": family_counts["FN"] / job_count,
        "fl_wo_ratio": family_counts["FL"] / job_count,
        "wo_per_block": job_count / block_count,
        "cut_per_wo": cut_sum / job_count,
        "bevel_per_wo": bevel_sum / job_count,
        "tact_per_wo": tact_sum / job_count,
    }


def _required_non_negative(job: object, field: str, job_id: object) -> float:
    if not hasattr(job, field):
        print(
            "[ERROR][Phase2.validation_grid._required_non_negative] "
            f"cause=missing_field job_id={job_id} field={field}"
        )
        raise RuntimeError(f"validation job requires {field}")
    return _non_negative_number(getattr(job, field), field, job_id)


def _non_negative_number(value: object, field: str, job_id: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][Phase2.validation_grid._non_negative_number] "
            f"cause=invalid_number job_id={job_id} field={field} value={value}"
        )
        raise RuntimeError(f"invalid validation job field: {field}") from exc
    if not math.isfinite(number) or number < 0:
        print(
            "[ERROR][Phase2.validation_grid._non_negative_number] "
            f"cause=invalid_range job_id={job_id} field={field} value={value}"
        )
        raise RuntimeError(f"validation job field must be finite and non-negative: {field}")
    return number


def _rank_normalize_profiles(
    profiles: Sequence[Mapping[str, float]],
) -> tuple[dict[str, float], ...]:
    normalized = [dict() for _ in profiles]
    denominator = max(1, len(profiles) - 1)
    for field in _DISTRIBUTION_FIELDS:
        ordered = sorted(range(len(profiles)), key=lambda index: (profiles[index][field], index))
        for rank, profile_index in enumerate(ordered):
            normalized[profile_index][field] = rank / denominator
    return tuple(normalized)


def _profile_distance(
    actual: Mapping[str, float],
    target: Mapping[str, float],
) -> float:
    return sum((actual[field] - target[field]) ** 2 for field in _DISTRIBUTION_FIELDS)
