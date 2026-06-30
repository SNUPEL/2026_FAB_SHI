"""Phase 1 direct pair-action self-labeling.

This is the replacement path for the weak split action model:

    old: SELECT_BLOCK -> SELECT_BAY
    new: SELECT_PAIR(block, bay)

The pair features include projected objective deltas, so the policy scores the
actual decision unit used by the workload-balancing objective.

입출력 요약:
- 입력: episode별 Job-like mapping, Bay 후보(`22,23,24`), heuristic bank, 학습 설정.
- 내부 후보: 매 step마다 가능한 모든 `(block_set_id, bay_id)` pair.
- pseudo-label: agent rollout 후보와 8개 휴리스틱 후보 중 목적함수 tuple이 가장 작은 complete assignment.
- 출력: checkpoint, metrics.csv, candidate_summary.csv, best_action_table.jsonl, validation CSV/PNG.
"""

# LINE-BY-LINE: 미래 타입 힌트를 문자열로 늦게 평가합니다. 사용: Python 버전별 annotation 충돌을 줄입니다.
from __future__ import annotations

# LINE-BY-LINE: CSV audit 파일을 읽고 쓰기 위해 사용합니다. 예: metrics.csv, validation_candidate_summary.csv.
import csv
# LINE-BY-LINE: JSON/JSONL 저장에 사용합니다. 예: score_json, best_action_table.jsonl, summary.json.
import json
# LINE-BY-LINE: `dataclass`는 transition/candidate record class를 간결하게 정의하는 데 사용합니다.
from dataclasses import dataclass
# LINE-BY-LINE: output path/checkpoint path를 OS 독립적으로 다루기 위해 사용합니다.
from pathlib import Path
# LINE-BY-LINE: 함수 인자/반환 타입을 명확히 하기 위한 typing import입니다.
from typing import Callable, Dict, List, Mapping, Sequence

# LINE-BY-LINE: PyTorch tensor, model checkpoint, random seed, optimizer 실행에 사용합니다.
import torch
# LINE-BY-LINE: cross entropy loss 계산에 사용합니다.
import torch.nn.functional as F

# LINE-BY-LINE: 직접 pair action을 scoring하는 pointer network입니다. 입력: pair feature matrix + env feature vector.
from Train.network.phase1_pointer import Phase1PairPointerPolicy
# LINE-BY-LINE: Phase 1 Bay 부하 계산과 목적함수 점수 계산에 필요한 기존 balancer 함수/타입입니다.
from Utils.phase1_bay_balancer import (
    Phase1Block,
    _add_block_load,
    _collect_blocks,
    _multi_objective_load_score,
    _normalize_bay_ids,
)
# LINE-BY-LINE: Bay별 누적 부하 dict를 초기화하는 기존 MDP helper입니다.
from Utils.phase1_mdp import _empty_bay_loads
# LINE-BY-LINE: 8개 휴리스틱 후보 bank와 complete heuristic assignment 생성 함수를 재사용합니다.
from Train.algorithm.phase1_self_labeling import (
    PHASE1_SELF_LABEL_HEURISTIC_BANK,
    run_phase1_heuristic_candidate,
)


# LINE-BY-LINE: pair 후보 하나의 feature 이름 목록입니다. 순서는 network checkpoint와 CSV audit의 계약이므로 바꾸면 기존 checkpoint와 호환되지 않습니다.
PHASE1_PAIR_FEATURE_NAMES = [
    # LINE-BY-LINE: 후보 block의 강재수량이 전체 강재수량에서 차지하는 비율입니다.
    "block_steel_ratio",
    # LINE-BY-LINE: 후보 block의 절단장이 전체 절단장에서 차지하는 비율입니다.
    "block_cut_ratio",
    # LINE-BY-LINE: 후보 block의 베벨수량이 전체 베벨수량에서 차지하는 비율입니다.
    "block_bevel_ratio",
    # LINE-BY-LINE: 후보 Bay가 22인지 나타내는 one-hot flag입니다.
    "bay_22_flag",
    # LINE-BY-LINE: 후보 Bay가 23인지 나타내는 one-hot flag입니다.
    "bay_23_flag",
    # LINE-BY-LINE: 후보 Bay가 24인지 나타내는 one-hot flag입니다.
    "bay_24_flag",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay 22의 강재수량 누적 비율입니다.
    "projected_bay22_steel_ratio",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay 23의 강재수량 누적 비율입니다.
    "projected_bay23_steel_ratio",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay 24의 강재수량 누적 비율입니다.
    "projected_bay24_steel_ratio",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay 22의 절단장 누적 비율입니다.
    "projected_bay22_cut_ratio",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay 23의 절단장 누적 비율입니다.
    "projected_bay23_cut_ratio",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay 24의 절단장 누적 비율입니다.
    "projected_bay24_cut_ratio",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay 22의 베벨수량 누적 비율입니다.
    "projected_bay22_bevel_ratio",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay 23의 베벨수량 누적 비율입니다.
    "projected_bay23_bevel_ratio",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay 24의 베벨수량 누적 비율입니다.
    "projected_bay24_bevel_ratio",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay별 강재수량 max-min gap 비율입니다.
    "projected_steel_gap",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay별 절단장 max-min gap 비율입니다.
    "projected_cut_gap",
    # LINE-BY-LINE: 이 pair를 넣은 뒤 Bay별 베벨수량 max-min gap 비율입니다.
    "projected_bevel_gap",
]
# LINE-BY-LINE: 전체 환경 상태 feature 이름입니다. 후보별로 변하지 않는 값은 pair feature가 아니라 여기 둡니다.
PHASE1_PAIR_ENV_FEATURE_NAMES = [
    # LINE-BY-LINE: 현재까지 배정 완료된 block 비율입니다.
    "progress_ratio",
    # LINE-BY-LINE: 아직 남은 block 비율입니다.
    "remaining_block_ratio",
    # LINE-BY-LINE: 현재 Bay 22의 강재수량 누적 비율입니다.
    "current_bay22_steel_ratio",
    # LINE-BY-LINE: 현재 Bay 23의 강재수량 누적 비율입니다.
    "current_bay23_steel_ratio",
    # LINE-BY-LINE: 현재 Bay 24의 강재수량 누적 비율입니다.
    "current_bay24_steel_ratio",
    # LINE-BY-LINE: 현재 Bay 22의 절단장 누적 비율입니다.
    "current_bay22_cut_ratio",
    # LINE-BY-LINE: 현재 Bay 23의 절단장 누적 비율입니다.
    "current_bay23_cut_ratio",
    # LINE-BY-LINE: 현재 Bay 24의 절단장 누적 비율입니다.
    "current_bay24_cut_ratio",
    # LINE-BY-LINE: 현재 Bay 22의 베벨수량 누적 비율입니다.
    "current_bay22_bevel_ratio",
    # LINE-BY-LINE: 현재 Bay 23의 베벨수량 누적 비율입니다.
    "current_bay23_bevel_ratio",
    # LINE-BY-LINE: 현재 Bay 24의 베벨수량 누적 비율입니다.
    "current_bay24_bevel_ratio",
    # LINE-BY-LINE: 현재 Bay별 강재수량 max-min gap 비율입니다.
    "steel_gap_ratio",
    # LINE-BY-LINE: 현재 Bay별 절단장 max-min gap 비율입니다.
    "cut_gap_ratio",
    # LINE-BY-LINE: 현재 Bay별 베벨수량 max-min gap 비율입니다.
    "bevel_gap_ratio",
]
# LINE-BY-LINE: gap-only 목적함수 tuple 전체를 CSV에 score_0~score_3으로 저장하기 위한 고정 column 이름입니다.
PHASE1_SCORE_FIELD_NAMES = [f"score_{index}" for index in range(4)]


# LINE-BY-LINE: `Phase1PairTransition`은 학습 pseudo-label 한 step을 담는 immutable record입니다.
@dataclass(frozen=True)
class Phase1PairTransition:
    """One direct pair decision used for pair-policy training."""

    # LINE-BY-LINE: decision phase 이름입니다. 현재 pair action에서는 항상 `SELECT_PAIR`입니다.
    phase: str
    # LINE-BY-LINE: 현재 step에서 가능한 모든 pair 후보 feature matrix입니다. shape 예: `(남은 block 수*3, 18)`.
    candidate_features: List[List[float]]
    # LINE-BY-LINE: 현재 step의 환경 feature vector입니다. shape 예: `(14,)`.
    env_features: List[float]
    # LINE-BY-LINE: pseudo-label로 선택된 후보 index입니다. cross entropy target으로 사용합니다.
    selected_action_index: int
    # LINE-BY-LINE: 후보 id 목록입니다. 예: `PROJ_1::BLK_1@22`.
    candidate_ids: List[str]
    # LINE-BY-LINE: 실제 선택된 후보 id입니다. audit와 replay에 사용합니다.
    selected_action_id: str
    # LINE-BY-LINE: 선택된 block_set_id입니다. 예: `PROJ_1::BLK_1`.
    selected_block_set_id: str
    # LINE-BY-LINE: 선택된 Bay입니다. 예: `22`, `23`, `24`.
    selected_bay: str


# LINE-BY-LINE: `Phase1PairCandidate`는 한 episode 전체를 끝까지 배정한 complete solution 후보입니다.
@dataclass(frozen=True)
class Phase1PairCandidate:
    """One complete Phase 1 pair-action assignment candidate."""

    # LINE-BY-LINE: 후보를 만든 방법입니다. 예: `agent_greedy`, `agent_sample_3`, `bevel_first_long_cut_preferred`.
    source: str
    # LINE-BY-LINE: 이 complete assignment를 만들 때 거친 step별 transition 목록입니다.
    transitions: List[Phase1PairTransition]
    # LINE-BY-LINE: 최종 block->Bay 배정 결과입니다. key=block_set_id, value=Bay.
    assignments: Dict[str, str]
    # LINE-BY-LINE: 최종 Bay별 누적 부하입니다. 목적함수 점수와 CSV report에 사용합니다.
    bay_loads: Dict[str, Dict[str, int | float]]


def build_phase1_pair_candidates(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    bay_loads: Mapping[str, Mapping[str, int | float]] | None = None,
    remaining_block_ids: Sequence[str] | None = None,
    assigned_block_count: int = 0,
) -> List[Dict]:
    """Build direct `(block, bay)` candidates for the current Phase 1 state.

    입력 예시:
    - `jobs`: `{job_id: JobLike}`. 여러 W/O가 같은 block_set_id를 가질 수 있다.
    - `bay_ids`: `["22", "23", "24"]`.
    - `bay_loads`: 현재까지 Bay별 누적 부하. None이면 모두 0으로 시작.
    - `remaining_block_ids`: 아직 배정하지 않은 block만 후보로 만들 때 사용.

    출력 예시:
    - `[{"action_id": "PROJ_1::BLK_2@22", "features": [...]}, ...]`
    """

    # LINE-BY-LINE: Bay ID를 문자열 tuple로 정규화하고, 빈 Bay 입력이면 여기서 실패합니다.
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    # LINE-BY-LINE: W/O job들을 block_set_id 기준으로 묶고, 강재수량/절단장/베벨수량 등을 block 단위로 합산합니다.
    blocks = _collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids, require_multi_objective=True)
    # LINE-BY-LINE: 남은 block 목록이 외부에서 오면 그것만 사용하고, 없으면 전체 block을 남은 후보로 봅니다.
    remaining = set(remaining_block_ids) if remaining_block_ids is not None else {block.block_set_id for block in blocks}
    # LINE-BY-LINE: 현재 Bay별 부하를 복사합니다. 원본 dict를 직접 수정하지 않기 위해 새 dict를 만듭니다.
    current_loads = _empty_bay_loads(normalized_bay_ids) if bay_loads is None else {
        str(bay_id): dict(loads)
        for bay_id, loads in bay_loads.items()
    }
    # LINE-BY-LINE: feature 정규화 denominator입니다. 예: 전체 강재수량, 전체 절단장, 최대 길이.
    totals = _problem_totals(blocks)
    # LINE-BY-LINE: 반환할 pair 후보 row를 누적합니다.
    candidates: List[Dict] = []
    # LINE-BY-LINE: block 순서는 deterministic하게 block_set_id 오름차순으로 고정합니다. 재현 가능한 학습을 위함입니다.
    for block in sorted(blocks, key=lambda item: item.block_set_id):
        # LINE-BY-LINE: 이미 배정된 block이면 이번 step 후보에서 제외합니다.
        if block.block_set_id not in remaining:
            continue
        # LINE-BY-LINE: 해당 block이 갈 수 있는 Bay마다 하나의 `(block,bay)` action 후보를 만듭니다.
        for bay_id in block.allowed_bay_ids:
            # LINE-BY-LINE: 후보를 넣은 뒤의 projected load를 계산하기 위해 현재 부하를 복사합니다.
            projected = {current_bay_id: dict(loads) for current_bay_id, loads in current_loads.items()}
            # LINE-BY-LINE: 이 block을 후보 Bay에 넣었다고 가정하고 projected 부하를 갱신합니다.
            _add_block_load(projected[bay_id], bay_id, block)
            # LINE-BY-LINE: action id, block id, Bay id, feature 이름, feature 값을 한 row로 저장합니다.
            candidates.append(
                {
                    "action_id": f"{block.block_set_id}@{bay_id}",
                    "block_set_id": block.block_set_id,
                    "bay_id": bay_id,
                    "feature_names": list(PHASE1_PAIR_FEATURE_NAMES),
                    "features": _pair_features(
                        block=block,
                        bay_id=bay_id,
                        projected_loads=projected,
                        totals=totals,
                    ),
                }
            )
    # LINE-BY-LINE: 후보가 하나도 없으면 학습/추론을 계속할 수 없으므로 fallback 없이 실패시킵니다.
    if not candidates:
        print("[ERROR][phase1_pair_self_labeling.build_phase1_pair_candidates] cause=no_pair_candidates")
        raise RuntimeError("Phase 1 pair MDP has no feasible pair candidates")
    # LINE-BY-LINE: 현재 step에서 선택 가능한 모든 pair 후보를 반환합니다.
    return candidates


# LINE-BY-LINE: Phase 1 pair-action self-labeling의 메인 학습 함수입니다. CLI `phase1-train-pair-self-labeling`이 호출합니다.
def train_phase1_pair_self_labeling(
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_metadata: Sequence[Mapping[str, object]] | None,
    bay_ids: Sequence[str],
    output_dir: str | Path,
    episodes: int,
    rollout_samples: int = 4,
    heuristic_algorithms: Sequence[str] = PHASE1_SELF_LABEL_HEURISTIC_BANK,
    score_mode: str = "steel_first",
    lr: float = 1e-3,
    hidden_dim: int = 128,
    temperature: float = 1.0,
    seed: int = 0,
    episode_factory: Callable[[int], Mapping[str, object]] | None = None,
    checkpoint_every: int = 100,
    validation_every: int = 100,
    validation_episodes: int = 20,
    validation_episode_factory: Callable[[int], Mapping[str, object]] | None = None,
    resume_checkpoint: str | Path | None = None,
) -> Dict:
    """Train a pair-action policy from best-of-K complete assignments.

    학습 절차:
    1. episode 문제를 하나 가져온다.
    2. 현재 policy rollout 후보 K개를 만든다. 첫 번째는 greedy, 나머지는 sample.
    3. 8개 휴리스틱 complete assignment 후보를 만든다.
    4. 목적함수 tuple 기준 best 후보 하나를 pseudo-label로 선택한다.
    5. best 후보의 step별 선택 index를 cross entropy target으로 학습한다.
    6. 일정 주기마다 checkpoint와 validation report를 저장한다.
    """

    # LINE-BY-LINE: 입력 episode 수, rollout 수, learning rate, checkpoint interval 등을 먼저 검증합니다.
    _validate_train_inputs(
        episode_jobs,
        episode_metadata,
        episodes,
        rollout_samples,
        lr,
        temperature,
        episode_factory,
        checkpoint_every,
        validation_every,
        validation_episodes,
        validation_episode_factory,
    )
    # LINE-BY-LINE: PyTorch 난수 seed를 고정해 같은 입력에서 같은 초기 모델/샘플링을 재현합니다.
    torch.manual_seed(seed)
    # LINE-BY-LINE: Bay ID를 정규화합니다. 현재 Phase 1 권장값은 `("22", "23", "24")`입니다.
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    # LINE-BY-LINE: output_dir를 Path 객체로 바꾸고 없으면 생성합니다.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: pair feature 21차원, env feature 6차원을 받는 pointer policy를 생성합니다.
    model = Phase1PairPointerPolicy(
        pair_feature_dim=len(PHASE1_PAIR_FEATURE_NAMES),
        env_feature_dim=len(PHASE1_PAIR_ENV_FEATURE_NAMES),
        hidden_dim=hidden_dim,
    )
    # LINE-BY-LINE: Adam optimizer를 생성합니다. 학습 대상은 model parameter 전체입니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # LINE-BY-LINE: 주기 checkpoint를 저장할 하위 디렉터리입니다.
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: resume 옵션이 있으면 실제 checkpoint path를 계산합니다. 예: `latest`.
    resume_path = _resolve_resume_checkpoint(output_path, resume_checkpoint)
    # LINE-BY-LINE: 새 학습이면 1 episode부터, resume이면 checkpoint 다음 episode부터 시작합니다.
    start_episode = 1
    if resume_path is not None:
        # LINE-BY-LINE: checkpoint에서 model/optimizer state를 복원하고 완료 episode 번호를 읽습니다.
        start_episode = _load_pair_checkpoint(
            model=model,
            optimizer=optimizer,
            path=resume_path,
            hidden_dim=hidden_dim,
        ) + 1
    # LINE-BY-LINE: resume 시 기존 metrics.csv에서 start_episode 이전 row만 보존합니다.
    metrics_rows: List[Dict] = _read_csv_rows(output_path / "metrics.csv", "episode", start_episode)
    # LINE-BY-LINE: resume 시 기존 후보 audit CSV에서 start_episode 이전 row만 보존합니다.
    candidate_rows: List[Dict] = _read_csv_rows(output_path / "candidate_summary.csv", "episode", start_episode)
    # LINE-BY-LINE: resume 시 기존 pseudo-label JSONL에서 start_episode 이전 row만 보존합니다.
    best_action_rows: List[Dict] = _read_jsonl_rows(output_path / "best_action_table.jsonl", "episode", start_episode)
    # LINE-BY-LINE: resume 시 기존 validation summary에서 start_episode 이전 row만 보존합니다.
    validation_rows: List[Dict] = _read_csv_rows(output_path / "validation_summary.csv", "train_episode", start_episode)
    # LINE-BY-LINE: resume 시 기존 validation 후보별 score CSV에서 start_episode 이전 row만 보존합니다.
    validation_candidate_rows: List[Dict] = _read_csv_rows(output_path / "validation_candidate_summary.csv", "train_episode", start_episode)
    # LINE-BY-LINE: best validation checkpoint 비교를 위해 이전 summary의 best score를 읽습니다.
    best_validation_score: tuple | None = _read_best_validation_score(output_path / "summary.json") if resume_path is not None else None
    # LINE-BY-LINE: validation 기준으로 가장 좋은 모델을 저장할 path입니다.
    best_checkpoint_path = output_path / "phase1_pair_pointer_best.pt"
    # LINE-BY-LINE: validation을 한 번이라도 돌리면 여기에 PNG 경로들이 들어갑니다.
    validation_plot_paths: Dict[str, str] = {}

    print("[phase1-train-pair-self-labeling]")
    print(f"- episodes: {episodes}")
    print(f"- start_episode: {start_episode}")
    print(f"- rollout_samples: {rollout_samples}")
    print(f"- heuristic_algorithms: {','.join(heuristic_algorithms)}")
    print(f"- episode_mode: {'on_the_fly' if episode_factory is not None else 'prebuilt'}")
    print(f"- resume_checkpoint: {resume_path or ''}")

    # LINE-BY-LINE: start_episode부터 사용자가 요청한 episodes까지 학습 loop를 수행합니다.
    for episode in range(start_episode, episodes + 1):
        # LINE-BY-LINE: 현재 episode의 Job-like mapping과 metadata를 가져옵니다. on-the-fly factory도 여기서 호출됩니다.
        jobs, metadata = _episode_payload(episode, episode_jobs, episode_metadata, episode_factory)
        # LINE-BY-LINE: problem_id는 출력 CSV/JSONL에서 episode 문제를 식별하는 이름입니다.
        problem_id = str(metadata.get("problem_id") or metadata.get("episode_id") or f"EP{episode:05d}")
        # LINE-BY-LINE: block_count는 로그와 CSV audit에 저장하는 문제 크기입니다.
        block_count = int(metadata.get("block_count") or len(jobs))
        # LINE-BY-LINE: synthetic generator seed를 audit에 남깁니다. 없으면 빈 문자열입니다.
        problem_seed = metadata.get("seed", "")
        # LINE-BY-LINE: 현재 episode에서 비교할 complete assignment 후보 목록입니다.
        candidates: List[Phase1PairCandidate] = []
        # LINE-BY-LINE: 현재 policy로 rollout 후보를 만듭니다. 첫 sample은 greedy, 나머지는 확률 sampling입니다.
        for sample_index in range(1, rollout_samples + 1):
            # LINE-BY-LINE: sample_index=1은 deterministic greedy 후보로 두어 검증/추론 기준과 맞춥니다.
            is_greedy = sample_index == 1
            # LINE-BY-LINE: policy rollout 결과 complete assignment 후보를 후보 bank에 추가합니다.
            candidates.append(
                run_phase1_pair_policy_rollout(
                    jobs=jobs,
                    bay_ids=normalized_bay_ids,
                    model=model,
                    temperature=temperature,
                    seed=seed + episode * 1000 + sample_index,
                    source="agent_greedy" if is_greedy else f"agent_sample_{sample_index}",
                    selection="greedy" if is_greedy else "sample",
                )
            )
        # LINE-BY-LINE: 8개 휴리스틱 complete assignment도 같은 후보 bank에 추가합니다.
        for algorithm in heuristic_algorithms:
            candidates.append(_run_phase1_pair_heuristic_candidate(jobs, normalized_bay_ids, algorithm))

        # LINE-BY-LINE: 모든 후보 중 목적함수 tuple이 가장 작은 후보를 pseudo-label teacher로 선택합니다.
        best = min(candidates, key=lambda candidate: _score_bay_loads(candidate.bay_loads, score_mode))
        # LINE-BY-LINE: best 후보의 transition sequence를 target으로 cross entropy 학습 1회를 수행합니다.
        loss = _teacher_forcing_update(model, optimizer, best.transitions)
        # LINE-BY-LINE: 선택된 best 후보의 목적함수 score tuple입니다. 낮을수록 좋습니다.
        score = _score_bay_loads(best.bay_loads, score_mode)
        # LINE-BY-LINE: 모든 agent/heuristic 후보를 CSV audit row로 저장합니다.
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_rows.append(
                _candidate_summary_row(
                    episode=episode,
                    problem_id=problem_id,
                    block_count=block_count,
                    problem_seed=problem_seed,
                    candidate_index=candidate_index,
                    candidate=candidate,
                    best=best,
                    score_mode=score_mode,
                )
            )
        # LINE-BY-LINE: 실제 학습 target으로 사용한 best 후보의 step별 action table을 JSONL에 누적합니다.
        best_action_rows.extend(
            _best_action_rows(
                episode=episode,
                problem_id=problem_id,
                block_count=block_count,
                problem_seed=problem_seed,
                best=best,
            )
        )
        # LINE-BY-LINE: episode 단위 학습 metric row를 누적합니다.
        metrics_rows.append(
            {
                "episode": episode,
                "problem_id": problem_id,
                "block_count": block_count,
                "problem_seed": problem_seed,
                "best_source": best.source,
                "loss": loss,
                "score_json": json.dumps(list(score), ensure_ascii=False),
                "candidate_count": len(candidates),
            }
        )
        # LINE-BY-LINE: 현재까지 metrics를 즉시 파일에 씁니다. 중간 중단되어도 진행 상황이 남습니다.
        _write_metrics(output_path / "metrics.csv", metrics_rows)
        # LINE-BY-LINE: 현재까지 후보 audit row를 즉시 파일에 씁니다.
        _write_candidate_summary(output_path / "candidate_summary.csv", candidate_rows)
        # LINE-BY-LINE: 현재까지 pseudo-label action table을 즉시 파일에 씁니다.
        _write_jsonl(output_path / "best_action_table.jsonl", best_action_rows)
        # LINE-BY-LINE: 콘솔에 episode 진행 상황과 best source/score를 출력합니다.
        print(
            "[CHECK][phase1_pair_self_labeling.train] "
            f"episode={episode} problem_id={problem_id} block_count={block_count} "
            f"best_source={best.source} loss={loss:.6f} score={score}"
        )
        # LINE-BY-LINE: checkpoint interval에 도달하면 주기 checkpoint를 저장합니다.
        if episode % checkpoint_every == 0:
            _save_pair_checkpoint(
                model=model,
                optimizer=optimizer,
                path=checkpoint_dir / f"phase1_pair_pointer_ep{episode:05d}.pt",
                hidden_dim=hidden_dim,
                episode=episode,
                validation_score=None,
            )
        # LINE-BY-LINE: validation factory가 있고 validation interval에 도달하면 holdout 검증을 실행합니다.
        if validation_episode_factory is not None and episode % validation_every == 0:
            # LINE-BY-LINE: validation 문제에서 agent_greedy와 heuristic bank를 비교합니다.
            validation = _validate_pair_policy(
                model=model,
                validation_episode_factory=validation_episode_factory,
                validation_episodes=validation_episodes,
                bay_ids=normalized_bay_ids,
                heuristic_algorithms=heuristic_algorithms,
                score_mode=score_mode,
                episode=episode,
            )
            # LINE-BY-LINE: validation episode별 agent 요약 row를 누적합니다.
            validation_rows.extend(validation["rows"])
            # LINE-BY-LINE: validation에서 평가한 모든 방법론 후보 row를 누적합니다.
            validation_candidate_rows.extend(validation["candidate_rows"])
            # LINE-BY-LINE: validation summary CSV를 저장합니다.
            _write_validation_summary(output_path / "validation_summary.csv", validation_rows)
            # LINE-BY-LINE: validation 후보별 score/rank CSV를 저장합니다.
            _write_validation_candidate_summary(output_path / "validation_candidate_summary.csv", validation_candidate_rows)
            # LINE-BY-LINE: 최신 validation 기준 scatter/bar plot PNG를 저장합니다.
            validation_plot_paths = _write_validation_plots(
                output_path=output_path,
                candidate_rows=validation_candidate_rows,
                summary_rows=validation_rows,
                score_mode=score_mode,
            )
            # LINE-BY-LINE: 실적 validation이 있으면 actual_8days 평균 score를 best checkpoint 기준으로 우선 사용합니다.
            current_score = validation["agent_mean_score_by_source"].get(
                "actual_8days",
                validation["agent_mean_score"],
            )
            # LINE-BY-LINE: 이전 best보다 lexicographic score가 좋으면 best checkpoint를 갱신합니다.
            saved_best = best_validation_score is None or current_score < best_validation_score
            if saved_best:
                best_validation_score = current_score
                _save_pair_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    path=best_checkpoint_path,
                    hidden_dim=hidden_dim,
                    episode=episode,
                    validation_score=current_score,
                )
            # LINE-BY-LINE: validation 요약을 콘솔에 출력합니다. best_checkpoint_saved 여부가 핵심입니다.
            print(
                "[VALIDATION][phase1_pair_self_labeling.validate] "
                f"episode={episode} validation_episodes={validation_episodes} "
                f"agent_best_rate={validation['agent_best_rate']:.6f} "
                f"checkpoint_score={current_score} best_checkpoint_saved={saved_best}"
            )

    # LINE-BY-LINE: 학습 종료 시점의 최종 모델 checkpoint path입니다.
    checkpoint_path = output_path / "phase1_pair_pointer.pt"
    # LINE-BY-LINE: 학습 결과 요약 JSON path입니다.
    summary_path = output_path / "summary.json"
    # LINE-BY-LINE: 마지막 episode의 model/optimizer state를 최종 checkpoint로 저장합니다.
    _save_pair_checkpoint(
        model=model,
        optimizer=optimizer,
        path=checkpoint_path,
        hidden_dim=hidden_dim,
        episode=episodes,
        validation_score=None,
    )
    # LINE-BY-LINE: CLI와 외부 notebook이 읽을 수 있는 summary dict를 구성합니다.
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path) if best_validation_score is not None else "",
        "metrics_csv": str(output_path / "metrics.csv"),
        "candidate_summary_csv": str(output_path / "candidate_summary.csv"),
        "best_action_table_jsonl": str(output_path / "best_action_table.jsonl"),
        "validation_summary_csv": str(output_path / "validation_summary.csv") if validation_rows else "",
        "validation_candidate_summary_csv": str(output_path / "validation_candidate_summary.csv") if validation_candidate_rows else "",
        **validation_plot_paths,
        "summary_json": str(summary_path),
        "episodes": episodes,
        "rollout_samples": rollout_samples,
        "heuristic_algorithms": list(heuristic_algorithms),
        "score_mode": score_mode,
        "best_source_counts": _source_counts(metrics_rows),
        "episode_mode": "on_the_fly" if episode_factory is not None else "prebuilt",
        "checkpoint_every": checkpoint_every,
        "validation_every": validation_every,
        "validation_episodes": validation_episodes,
        "resume_checkpoint": str(resume_path) if resume_path is not None else "",
        "start_episode": start_episode,
        "resumed_from_episode": start_episode - 1 if resume_path is not None else 0,
        "best_validation_score": list(best_validation_score) if best_validation_score is not None else [],
    }
    # LINE-BY-LINE: summary.json을 UTF-8 JSON으로 저장합니다.
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    # LINE-BY-LINE: 호출자(main.py)가 출력할 수 있도록 summary dict를 반환합니다.
    return summary


def run_phase1_pair_policy_rollout(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    model: Phase1PairPointerPolicy | None,
    temperature: float,
    seed: int,
    source: str,
    selection: str = "sample",
) -> Phase1PairCandidate:
    """Sample one direct pair-action assignment."""

    if temperature <= 0:
        print(f"[ERROR][phase1_pair_self_labeling.run_phase1_pair_policy_rollout] cause=non_positive_temperature value={temperature}")
        raise ValueError("temperature must be positive")
    if selection not in {"sample", "greedy"}:
        print(f"[ERROR][phase1_pair_self_labeling.run_phase1_pair_policy_rollout] cause=unknown_selection selection={selection}")
        raise RuntimeError(f"unknown pair rollout selection: {selection}")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    blocks = _collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids, require_multi_objective=True)
    bay_loads = _empty_bay_loads(normalized_bay_ids)
    remaining = {block.block_set_id for block in blocks}
    transitions: List[Phase1PairTransition] = []
    assignments: Dict[str, str] = {}

    for block_step in range(len(blocks)):
        candidates = build_phase1_pair_candidates(
            jobs=jobs,
            bay_ids=normalized_bay_ids,
            bay_loads=bay_loads,
            remaining_block_ids=sorted(remaining),
            assigned_block_count=block_step,
        )
        env_features = _pair_env_features(
            bay_loads=bay_loads,
            assigned_block_count=block_step,
            total_block_count=len(blocks),
            totals=_problem_totals(blocks),
        )
        logits = _score_pair_candidates(model, [row["features"] for row in candidates], env_features)
        selected_index = int(torch.argmax(logits).item()) if selection == "greedy" else _sample_index(
            logits=logits,
            temperature=temperature,
            generator=generator,
        )
        selected = candidates[selected_index]
        selected_block_id = str(selected["block_set_id"])
        selected_bay = str(selected["bay_id"])
        transitions.append(
            Phase1PairTransition(
                phase="SELECT_PAIR",
                candidate_features=[row["features"] for row in candidates],
                env_features=env_features,
                selected_action_index=selected_index,
                candidate_ids=[str(row["action_id"]) for row in candidates],
                selected_action_id=str(selected["action_id"]),
                selected_block_set_id=selected_block_id,
                selected_bay=selected_bay,
            )
        )
        block = next(item for item in blocks if item.block_set_id == selected_block_id)
        assignments[selected_block_id] = selected_bay
        _add_block_load(bay_loads[selected_bay], selected_bay, block)
        remaining.remove(selected_block_id)

    return Phase1PairCandidate(
        source=source,
        transitions=transitions,
        assignments=assignments,
        bay_loads=_plain_bay_loads(bay_loads),
    )


def _validate_pair_policy(
    model: Phase1PairPointerPolicy,
    validation_episode_factory: Callable[[int], Mapping[str, object]],
    validation_episodes: int,
    bay_ids: Sequence[str],
    heuristic_algorithms: Sequence[str],
    score_mode: str,
    episode: int,
) -> Dict:
    """Evaluate deterministic greedy policy against the heuristic bank."""

    rows: List[Dict] = []
    candidate_rows: List[Dict] = []
    agent_scores: List[tuple] = []
    agent_scores_by_source: Dict[str, List[tuple]] = {}
    agent_best_count = 0
    for validation_index in range(1, validation_episodes + 1):
        jobs, metadata = _episode_payload(validation_index, None, None, validation_episode_factory)
        problem_id = str(metadata.get("problem_id") or metadata.get("episode_id") or f"VAL{validation_index:05d}")
        block_count = int(metadata.get("block_count") or len(jobs))
        validation_source = str(metadata.get("validation_source") or "synthetic")
        candidates = [
            run_phase1_pair_policy_rollout(
                jobs=jobs,
                bay_ids=bay_ids,
                model=model,
                temperature=1.0,
                seed=episode * 100_000 + validation_index,
                source="agent_greedy",
                selection="greedy",
            )
        ]
        for algorithm in heuristic_algorithms:
            candidates.append(_run_phase1_pair_heuristic_candidate(jobs, bay_ids, algorithm))
        scored = [
            (_score_bay_loads(candidate.bay_loads, score_mode), candidate)
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item[0])
        rank_by_source = {
            candidate.source: rank
            for rank, (_, candidate) in enumerate(scored, start=1)
        }
        agent_score = _score_bay_loads(candidates[0].bay_loads, score_mode)
        best_score, best_candidate = scored[0]
        agent_rank = rank_by_source["agent_greedy"]
        agent_is_best = int(agent_rank == 1)
        agent_best_count += agent_is_best
        agent_scores.append(agent_score)
        agent_scores_by_source.setdefault(validation_source, []).append(agent_score)
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_rows.append(
                _validation_candidate_summary_row(
                    train_episode=episode,
                    validation_episode=validation_index,
                    validation_source=validation_source,
                    problem_id=problem_id,
                    block_count=block_count,
                    candidate_index=candidate_index,
                    candidate=candidate,
                    best=best_candidate,
                    rank=rank_by_source[candidate.source],
                    score_mode=score_mode,
                )
            )
        rows.append(
            {
                "train_episode": episode,
                "validation_episode": validation_index,
                "validation_source": validation_source,
                "problem_id": problem_id,
                "block_count": block_count,
                "agent_score_json": json.dumps(list(agent_score), ensure_ascii=False),
                "best_score_json": json.dumps(list(best_score), ensure_ascii=False),
                "best_source": best_candidate.source,
                "agent_is_best": agent_is_best,
                "agent_rank": agent_rank,
                "candidate_count": len(candidates),
            }
        )
    return {
        "rows": rows,
        "candidate_rows": candidate_rows,
        "agent_mean_score": _mean_score(agent_scores),
        "agent_mean_score_by_source": {
            source: _mean_score(source_scores)
            for source, source_scores in sorted(agent_scores_by_source.items())
        },
        "agent_best_rate": agent_best_count / validation_episodes,
    }


def _save_pair_checkpoint(
    model: Phase1PairPointerPolicy,
    optimizer: torch.optim.Optimizer,
    path: Path,
    hidden_dim: int,
    episode: int,
    validation_score: tuple | None,
) -> None:
    """Save an auditable pair-policy checkpoint."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pair_feature_names": PHASE1_PAIR_FEATURE_NAMES,
            "env_feature_names": PHASE1_PAIR_ENV_FEATURE_NAMES,
            "hidden_dim": hidden_dim,
            "episode": episode,
            "validation_score": list(validation_score) if validation_score is not None else [],
        },
        path,
    )


def _resolve_resume_checkpoint(output_path: Path, resume_checkpoint: str | Path | None) -> Path | None:
    """Resolve resume checkpoint path, supporting `latest` under output/checkpoints."""

    if resume_checkpoint in (None, ""):
        return None
    text = str(resume_checkpoint)
    if text == "latest":
        candidates = sorted((output_path / "checkpoints").glob("phase1_pair_pointer_ep*.pt"))
        if not candidates:
            print(
                "[ERROR][phase1_pair_self_labeling._resolve_resume_checkpoint] "
                f"cause=no_latest_checkpoint output_dir={output_path}"
            )
            raise RuntimeError(f"no latest checkpoint in {output_path / 'checkpoints'}")
        return candidates[-1]
    path = Path(text)
    if not path.exists():
        print(
            "[ERROR][phase1_pair_self_labeling._resolve_resume_checkpoint] "
            f"cause=missing_resume_checkpoint path={path}"
        )
        raise RuntimeError(f"missing_resume_checkpoint: {path}")
    return path


def _load_pair_checkpoint(
    model: Phase1PairPointerPolicy,
    optimizer: torch.optim.Optimizer,
    path: Path,
    hidden_dim: int,
) -> int:
    """Load model/optimizer state and return the completed episode number."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_hidden_dim = int(checkpoint.get("hidden_dim", -1))
    if checkpoint_hidden_dim != hidden_dim:
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=hidden_dim_mismatch checkpoint={checkpoint_hidden_dim} requested={hidden_dim} path={path}"
        )
        raise RuntimeError("resume checkpoint hidden_dim mismatch")
    if checkpoint.get("pair_feature_names") != PHASE1_PAIR_FEATURE_NAMES:
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=pair_feature_names_mismatch path={path}"
        )
        raise RuntimeError("resume checkpoint pair features mismatch")
    if checkpoint.get("env_feature_names") != PHASE1_PAIR_ENV_FEATURE_NAMES:
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=env_feature_names_mismatch path={path}"
        )
        raise RuntimeError("resume checkpoint env features mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    episode = int(checkpoint.get("episode", 0))
    if episode <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=invalid_checkpoint_episode episode={episode} path={path}"
        )
        raise RuntimeError("invalid resume checkpoint episode")
    print(
        "[CHECK][phase1_pair_self_labeling._load_pair_checkpoint] "
        f"path={path} completed_episode={episode}"
    )
    return episode


def _read_csv_rows(path: Path, episode_field: str, start_episode: int) -> List[Dict]:
    """Read existing CSV rows before `start_episode` for resume."""

    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = [dict(row) for row in csv.DictReader(file)]
    kept = [row for row in rows if _row_episode(row, episode_field, path) < start_episode]
    print(
        "[CHECK][phase1_pair_self_labeling._read_csv_rows] "
        f"path={path} loaded={len(rows)} kept={len(kept)} start_episode={start_episode}"
    )
    return kept


def _read_jsonl_rows(path: Path, episode_field: str, start_episode: int) -> List[Dict]:
    """Read existing JSONL rows before `start_episode` for resume."""

    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [row for row in rows if _row_episode(row, episode_field, path) < start_episode]
    print(
        "[CHECK][phase1_pair_self_labeling._read_jsonl_rows] "
        f"path={path} loaded={len(rows)} kept={len(kept)} start_episode={start_episode}"
    )
    return kept


def _read_best_validation_score(path: Path) -> tuple | None:
    """Read previous best validation score when resuming."""

    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    score = summary.get("best_validation_score") or []
    if not score:
        return None
    return tuple(float(value) for value in score)


def _row_episode(row: Mapping, episode_field: str, path: Path) -> int:
    """Parse episode field from persisted row."""

    try:
        return int(row[episode_field])
    except (KeyError, TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_pair_self_labeling._row_episode] "
            f"cause=invalid_episode_field field={episode_field} path={path} row={row}"
        )
        raise RuntimeError(f"invalid episode field in {path}") from exc


def _mean_score(scores: Sequence[tuple]) -> tuple:
    """Return mean objective tuple for validation checkpoint selection."""

    if not scores:
        print("[ERROR][phase1_pair_self_labeling._mean_score] cause=no_scores")
        raise RuntimeError("validation produced no scores")
    width = len(scores[0])
    return tuple(
        round(sum(float(score[index]) for score in scores) / len(scores), 12)
        for index in range(width)
    )


def _run_phase1_pair_heuristic_candidate(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str,
) -> Phase1PairCandidate:
    """Convert an existing complete heuristic assignment into pair transitions."""

    heuristic = run_phase1_heuristic_candidate(jobs=jobs, bay_ids=bay_ids, algorithm=algorithm)
    transitions = _pair_transitions_from_assignment(jobs, bay_ids, heuristic.assignments)
    return Phase1PairCandidate(
        source=algorithm,
        transitions=transitions,
        assignments=dict(heuristic.assignments),
        bay_loads=dict(heuristic.bay_loads),
    )


def _pair_transitions_from_assignment(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    assignments: Mapping[str, str],
) -> List[Phase1PairTransition]:
    """Replay a complete assignment into one `SELECT_PAIR` row per block."""

    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    blocks = _collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids, require_multi_objective=True)
    block_by_id = {block.block_set_id: block for block in blocks}
    bay_loads = _empty_bay_loads(normalized_bay_ids)
    remaining = set(block_by_id)
    transitions: List[Phase1PairTransition] = []
    for block_step, block_id in enumerate(assignments):
        if block_id not in remaining:
            print(f"[ERROR][phase1_pair_self_labeling._pair_transitions_from_assignment] cause=bad_block_order block_id={block_id}")
            raise RuntimeError(f"bad heuristic block order: {block_id}")
        candidates = build_phase1_pair_candidates(
            jobs=jobs,
            bay_ids=normalized_bay_ids,
            bay_loads=bay_loads,
            remaining_block_ids=sorted(remaining),
            assigned_block_count=block_step,
        )
        selected_bay = str(assignments[block_id])
        selected_action_id = f"{block_id}@{selected_bay}"
        candidate_ids = [str(row["action_id"]) for row in candidates]
        selected_index = _candidate_index(candidate_ids, selected_action_id)
        transitions.append(
            Phase1PairTransition(
                phase="SELECT_PAIR",
                candidate_features=[row["features"] for row in candidates],
                env_features=_pair_env_features(
                    bay_loads=bay_loads,
                    assigned_block_count=block_step,
                    total_block_count=len(blocks),
                    totals=_problem_totals(blocks),
                ),
                selected_action_index=selected_index,
                candidate_ids=candidate_ids,
                selected_action_id=selected_action_id,
                selected_block_set_id=str(block_id),
                selected_bay=selected_bay,
            )
        )
        _add_block_load(bay_loads[selected_bay], selected_bay, block_by_id[block_id])
        remaining.remove(block_id)
    return transitions


def _pair_features(
    block: Phase1Block,
    bay_id: str,
    projected_loads: Mapping[str, Mapping[str, int | float]],
    totals: Mapping[str, float],
) -> List[float]:
    """Return normalized features for one `(block, bay)` candidate."""

    steel_values = [float(row["steel_quantity_sum"]) for row in projected_loads.values()]
    cut_values = [float(row["cut_length_sum"]) for row in projected_loads.values()]
    bevel_values = [float(row["bevel_quantity_sum"]) for row in projected_loads.values()]
    return [
        block.steel_quantity_sum / totals["steel"],
        block.cut_length_sum / totals["cut"],
        block.bevel_quantity_sum / totals["bevel"],
        *_bay_one_hot(bay_id),
        *_bay_load_ratios(projected_loads, "steel_quantity_sum", totals["steel"]),
        *_bay_load_ratios(projected_loads, "cut_length_sum", totals["cut"]),
        *_bay_load_ratios(projected_loads, "bevel_quantity_sum", totals["bevel"]),
        _gap(steel_values) / totals["steel"],
        _gap(cut_values) / totals["cut"],
        _gap(bevel_values) / totals["bevel"],
    ]


def _pair_env_features(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    assigned_block_count: int,
    total_block_count: int,
    totals: Mapping[str, float],
) -> List[float]:
    """Return normalized environment features before selecting a pair."""

    steel_values = [float(row["steel_quantity_sum"]) for row in bay_loads.values()]
    cut_values = [float(row["cut_length_sum"]) for row in bay_loads.values()]
    bevel_values = [float(row["bevel_quantity_sum"]) for row in bay_loads.values()]
    return [
        assigned_block_count / total_block_count,
        (total_block_count - assigned_block_count) / total_block_count,
        *_bay_load_ratios(bay_loads, "steel_quantity_sum", totals["steel"]),
        *_bay_load_ratios(bay_loads, "cut_length_sum", totals["cut"]),
        *_bay_load_ratios(bay_loads, "bevel_quantity_sum", totals["bevel"]),
        _gap(steel_values) / totals["steel"],
        _gap(cut_values) / totals["cut"],
        _gap(bevel_values) / totals["bevel"],
    ]


def _teacher_forcing_update(
    model: Phase1PairPointerPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Phase1PairTransition],
) -> float:
    """Cross-entropy update for selected pair sequence."""

    if not transitions:
        print("[ERROR][phase1_pair_self_labeling._teacher_forcing_update] cause=no_transitions")
        raise RuntimeError("best pair candidate has no transitions")
    total_loss = torch.zeros((), dtype=torch.float32)
    for transition in transitions:
        logits = model.score_pairs(
            torch.tensor(transition.candidate_features, dtype=torch.float32),
            torch.tensor(transition.env_features, dtype=torch.float32),
        )
        if transition.selected_action_index >= logits.numel():
            print(
                "[ERROR][phase1_pair_self_labeling._teacher_forcing_update] "
                f"cause=selected_index_out_of_range selected={transition.selected_action_index} candidate_count={logits.numel()}"
            )
            raise RuntimeError("selected pair index out of range")
        total_loss = total_loss + F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([transition.selected_action_index], dtype=torch.long),
        )
    loss = total_loss / len(transitions)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def _score_pair_candidates(
    model: Phase1PairPointerPolicy | None,
    candidate_features: List[List[float]],
    env_features: List[float],
) -> torch.Tensor:
    """Score pair candidates, or return uniform logits for random rollout."""

    if model is None:
        return torch.zeros((len(candidate_features),), dtype=torch.float32)
    return model.score_pairs(
        torch.tensor(candidate_features, dtype=torch.float32),
        torch.tensor(env_features, dtype=torch.float32),
    ).detach()


def _sample_index(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> int:
    """Sample one candidate index from logits."""

    if logits.numel() == 0:
        print("[ERROR][phase1_pair_self_labeling._sample_index] cause=no_logits")
        raise RuntimeError("cannot sample from empty pair logits")
    return int(torch.multinomial(torch.softmax(logits / temperature, dim=0), num_samples=1, generator=generator).item())


def _candidate_index(candidate_ids: Sequence[str], selected: str) -> int:
    """Return selected action index or fail if heuristic violates the mask."""

    try:
        return list(candidate_ids).index(selected)
    except ValueError as exc:
        print(
            "[ERROR][phase1_pair_self_labeling._candidate_index] "
            f"cause=selected_not_in_pair_candidates selected={selected} candidates={list(candidate_ids)}"
        )
        raise RuntimeError(f"selected pair is not feasible: {selected}") from exc


def _score_bay_loads(bay_loads: Mapping[str, Mapping[str, int | float]], score_mode: str) -> tuple:
    """Score final Bay loads with the confirmed Phase 1 objective."""

    return _multi_objective_load_score(bay_loads, score_mode=score_mode)


def _plain_bay_loads(bay_loads: Mapping[str, Mapping[str, int | float]]) -> Dict[str, Dict[str, int | float]]:
    """Return JSON-safe Bay load mapping."""

    return {
        str(bay_id): {
            "steel_quantity_sum": int(loads["steel_quantity_sum"]),
            "cut_length_sum": float(loads["cut_length_sum"]),
            "bevel_quantity_sum": int(loads["bevel_quantity_sum"]),
            "long_cut_bay24_count": int(loads["long_cut_bay24_count"]),
            "wo_count": int(loads["wo_count"]),
            "block_count": int(loads["block_count"]),
        }
        for bay_id, loads in bay_loads.items()
    }


def _problem_totals(blocks: Sequence[Phase1Block]) -> Dict[str, float]:
    """Return non-zero denominators for normalized pair features."""

    if not blocks:
        print("[ERROR][phase1_pair_self_labeling._problem_totals] cause=no_blocks")
        raise RuntimeError("Phase 1 pair MDP requires at least one block")
    return {
        "steel": max(1.0, float(sum(block.steel_quantity_sum for block in blocks))),
        "cut": max(1.0, float(sum(block.cut_length_sum for block in blocks))),
        "bevel": max(1.0, float(sum(block.bevel_quantity_sum for block in blocks))),
        "block": max(1.0, float(len(blocks))),
    }


def _bay_one_hot(bay_id: str) -> List[float]:
    """Encode Phase 1 Bay candidates as fixed 22/23/24 one-hot values."""

    normalized = str(bay_id)
    bay_order = ("22", "23", "24")
    if normalized not in bay_order:
        print(f"[ERROR][phase1_pair_self_labeling._bay_one_hot] cause=unsupported_phase1_bay bay_id={bay_id}")
        raise RuntimeError(f"unsupported_phase1_bay: {bay_id}")
    return [1.0 if normalized == candidate else 0.0 for candidate in bay_order]


def _bay_load_ratios(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    field: str,
    denominator: float,
) -> List[float]:
    """Return Bay 22/23/24 load ratios for one workload field."""

    bay_order = ("22", "23", "24")
    return [
        float(bay_loads.get(bay_id, {}).get(field, 0.0)) / denominator
        for bay_id in bay_order
    ]


def _gap(values: Sequence[float]) -> float:
    """Return max-min gap."""

    return max(values) - min(values) if values else 0.0


def _validate_train_inputs(
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_metadata: Sequence[Mapping[str, object]] | None,
    episodes: int,
    rollout_samples: int,
    lr: float,
    temperature: float,
    episode_factory: Callable[[int], Mapping[str, object]] | None,
    checkpoint_every: int,
    validation_every: int,
    validation_episodes: int,
    validation_episode_factory: Callable[[int], Mapping[str, object]] | None,
) -> None:
    """Validate pair self-labeling train controls."""

    if episodes <= 0:
        print(f"[ERROR][phase1_pair_self_labeling._validate_train_inputs] cause=non_positive_episodes episodes={episodes}")
        raise RuntimeError("episodes must be positive")
    if episode_factory is None and (
        episode_jobs is None
        or episode_metadata is None
        or len(episode_jobs) != episodes
        or len(episode_metadata) != episodes
    ):
        print(
            "[ERROR][phase1_pair_self_labeling._validate_train_inputs] "
            f"cause=episode_count_mismatch episodes={episodes} "
            f"jobs={0 if episode_jobs is None else len(episode_jobs)} "
            f"metadata={0 if episode_metadata is None else len(episode_metadata)}"
        )
        raise RuntimeError("pair self-labeling requires one job mapping and metadata row per episode")
    if rollout_samples <= 0:
        print(f"[ERROR][phase1_pair_self_labeling._validate_train_inputs] cause=non_positive_rollout_samples rollout_samples={rollout_samples}")
        raise RuntimeError("rollout_samples must be positive")
    if lr <= 0 or temperature <= 0:
        print(f"[ERROR][phase1_pair_self_labeling._validate_train_inputs] cause=invalid_lr_or_temperature lr={lr} temperature={temperature}")
        raise RuntimeError("lr and temperature must be positive")
    if checkpoint_every <= 0 or validation_every <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling._validate_train_inputs] "
            f"cause=invalid_checkpoint_or_validation_interval checkpoint_every={checkpoint_every} validation_every={validation_every}"
        )
        raise RuntimeError("checkpoint_every and validation_every must be positive")
    if validation_episode_factory is not None and validation_episodes <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling._validate_train_inputs] "
            f"cause=non_positive_validation_episodes validation_episodes={validation_episodes}"
        )
        raise RuntimeError("validation_episodes must be positive when validation is enabled")


def _episode_payload(
    episode: int,
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_metadata: Sequence[Mapping[str, object]] | None,
    episode_factory: Callable[[int], Mapping[str, object]] | None,
) -> tuple[Mapping[str, object], Dict[str, object]]:
    """Return one training episode, generating it on demand when configured."""

    if episode_factory is not None:
        payload = episode_factory(episode)
        if not isinstance(payload, Mapping):
            print(
                "[ERROR][phase1_pair_self_labeling._episode_payload] "
                f"cause=invalid_factory_payload episode={episode} type={type(payload).__name__}"
            )
            raise RuntimeError("episode_factory must return a mapping")
        jobs = payload.get("jobs")
        metadata = payload.get("metadata")
    else:
        if episode_jobs is None or episode_metadata is None:
            print(
                "[ERROR][phase1_pair_self_labeling._episode_payload] "
                f"cause=missing_prebuilt_episode episode={episode}"
            )
            raise RuntimeError("prebuilt episode inputs are missing")
        jobs = episode_jobs[episode - 1]
        metadata = episode_metadata[episode - 1]
    if not isinstance(jobs, Mapping) or not jobs:
        print(
            "[ERROR][phase1_pair_self_labeling._episode_payload] "
            f"cause=invalid_jobs episode={episode} type={type(jobs).__name__}"
        )
        raise RuntimeError(f"invalid episode jobs: {episode}")
    if not isinstance(metadata, Mapping):
        print(
            "[ERROR][phase1_pair_self_labeling._episode_payload] "
            f"cause=invalid_metadata episode={episode} type={type(metadata).__name__}"
        )
        raise RuntimeError(f"invalid episode metadata: {episode}")
    return jobs, dict(metadata)


def _candidate_summary_row(
    episode: int,
    problem_id: str,
    block_count: int,
    problem_seed: int | str,
    candidate_index: int,
    candidate: Phase1PairCandidate,
    best: Phase1PairCandidate,
    score_mode: str,
) -> Dict:
    """Return one candidate audit row."""

    score = _score_bay_loads(candidate.bay_loads, score_mode)
    row = {
        "episode": episode,
        "problem_id": problem_id,
        "block_count": block_count,
        "problem_seed": problem_seed,
        "candidate_index": candidate_index,
        "source": candidate.source,
        "is_best": int(candidate is best),
        "score_mode": score_mode,
        "score_json": json.dumps(list(score), ensure_ascii=False),
        "assignment_count": len(candidate.assignments),
        "transition_count": len(candidate.transitions),
        "bay_loads_json": json.dumps(candidate.bay_loads, ensure_ascii=False, sort_keys=True),
    }
    row.update(_score_columns(score))
    return row


def _validation_candidate_summary_row(
    train_episode: int,
    validation_episode: int,
    validation_source: str,
    problem_id: str,
    block_count: int,
    candidate_index: int,
    candidate: Phase1PairCandidate,
    best: Phase1PairCandidate,
    rank: int,
    score_mode: str,
) -> Dict:
    """Return one validation candidate audit row."""

    score = _score_bay_loads(candidate.bay_loads, score_mode)
    row = {
        "train_episode": train_episode,
        "validation_episode": validation_episode,
        "validation_source": validation_source,
        "problem_id": problem_id,
        "block_count": block_count,
        "candidate_index": candidate_index,
        "source": candidate.source,
        "rank": rank,
        "is_best": int(candidate is best),
        "score_mode": score_mode,
        "score_json": json.dumps(list(score), ensure_ascii=False),
        "assignment_count": len(candidate.assignments),
        "transition_count": len(candidate.transitions),
        "bay_loads_json": json.dumps(candidate.bay_loads, ensure_ascii=False, sort_keys=True),
    }
    row.update(_score_columns(score))
    return row


def _score_columns(score: Sequence[int | float]) -> Dict[str, int | float]:
    """Return stable score_N columns for the full objective tuple."""

    return {
        field: score[index] if index < len(score) else ""
        for index, field in enumerate(PHASE1_SCORE_FIELD_NAMES)
    }


def _best_action_rows(
    episode: int,
    problem_id: str,
    block_count: int,
    problem_seed: int | str,
    best: Phase1PairCandidate,
) -> List[Dict]:
    """Return JSONL rows for selected pair pseudo-label sequence."""

    return [
        {
            "episode": episode,
            "problem_id": problem_id,
            "block_count": block_count,
            "problem_seed": problem_seed,
            "step": step,
            "source": best.source,
            "phase": transition.phase,
            "selected_action_index": transition.selected_action_index,
            "selected_action_id": transition.selected_action_id,
            "selected_block_set_id": transition.selected_block_set_id,
            "selected_bay": transition.selected_bay,
            "candidate_ids": transition.candidate_ids,
            "candidate_features": transition.candidate_features,
            "env_features": transition.env_features,
        }
        for step, transition in enumerate(best.transitions)
    ]


def _source_counts(rows: Sequence[Mapping]) -> Dict[str, int]:
    """Count best candidate sources."""

    counts: Dict[str, int] = {}
    for row in rows:
        source = str(row["best_source"])
        counts[source] = counts.get(source, 0) + 1
    return counts


def _write_metrics(path: Path, rows: Sequence[Mapping]) -> None:
    """Write per-episode train metrics."""

    _write_csv(
        path,
        [
            "episode",
            "problem_id",
            "block_count",
            "problem_seed",
            "best_source",
            "loss",
            "score_json",
            "candidate_count",
        ],
        rows,
    )


def _write_candidate_summary(path: Path, rows: Sequence[Mapping]) -> None:
    """Write all complete candidates for audit."""

    _write_csv(
        path,
        [
            "episode",
            "problem_id",
            "block_count",
            "problem_seed",
            "candidate_index",
            "source",
            "is_best",
            "score_mode",
            "score_json",
            *PHASE1_SCORE_FIELD_NAMES,
            "assignment_count",
            "transition_count",
            "bay_loads_json",
        ],
        rows,
    )


def _write_validation_candidate_summary(path: Path, rows: Sequence[Mapping]) -> None:
    """Write all method candidates evaluated during validation."""

    _write_csv(
        path,
        [
            "train_episode",
            "validation_episode",
            "validation_source",
            "problem_id",
            "block_count",
            "candidate_index",
            "source",
            "rank",
            "is_best",
            "score_mode",
            "score_json",
            *PHASE1_SCORE_FIELD_NAMES,
            "assignment_count",
            "transition_count",
            "bay_loads_json",
        ],
        rows,
    )


def _write_validation_plots(
    output_path: Path,
    candidate_rows: Sequence[Mapping],
    summary_rows: Sequence[Mapping],
    score_mode: str,
) -> Dict[str, str]:
    """Write validation comparison plots for the latest validation checkpoint."""

    if not candidate_rows:
        return {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR][phase1_pair_self_labeling._write_validation_plots] cause=matplotlib_import_failed error={exc}")
        raise RuntimeError("matplotlib is required to write Phase 1 pair validation plots") from exc

    latest_candidate_rows = _latest_rows(candidate_rows, "train_episode")
    latest_summary_rows = _latest_rows(summary_rows, "train_episode")
    plot_paths: Dict[str, str] = {}
    for output_key, filename, score_field, title, ylabel in _validation_plot_specs(score_mode):
        path = output_path / filename
        _plot_validation_metric(
            plt=plt,
            path=path,
            rows=latest_candidate_rows,
            score_field=score_field,
            title=title,
            ylabel=ylabel,
        )
        plot_paths[output_key] = str(path)
    best_source_path = output_path / "validation_best_source_counts.png"
    _plot_validation_best_source_counts(plt, best_source_path, latest_summary_rows)
    plot_paths["validation_best_source_counts_png"] = str(best_source_path)
    agent_rank_path = output_path / "validation_agent_rank.png"
    _plot_validation_agent_rank(plt, agent_rank_path, latest_summary_rows)
    plot_paths["validation_agent_rank_png"] = str(agent_rank_path)
    return plot_paths


def _validation_plot_specs(score_mode: str) -> List[tuple[str, str, str, str, str]]:
    """Return score-column mapping for validation plots."""

    if score_mode != "steel_first":
        print(f"[ERROR][phase1_pair_self_labeling._validation_plot_specs] cause=unknown_score_mode score_mode={score_mode}")
        raise RuntimeError(f"unknown_score_mode: {score_mode}")
    return [
        ("validation_steel_gap_png", "validation_steel_gap.png", "score_0", "Steel load gap", "Gap"),
        ("validation_cut_gap_png", "validation_cut_gap.png", "score_1", "Cut length gap", "Gap"),
        ("validation_bevel_gap_png", "validation_bevel_gap.png", "score_2", "Bevel quantity gap", "Gap"),
        ("validation_long_cut_png", "validation_long_cut.png", "score_3", "Long-cut Bay24 assignments", "Count"),
    ]


def _latest_rows(rows: Sequence[Mapping], episode_field: str) -> List[Mapping]:
    """Return rows from the latest train episode."""

    if not rows:
        return []
    latest_episode = max(int(row[episode_field]) for row in rows)
    return [row for row in rows if int(row[episode_field]) == latest_episode]


def _plot_validation_metric(plt, path: Path, rows: Sequence[Mapping], score_field: str, title: str, ylabel: str) -> None:
    """Scatter one validation metric by problem and method."""

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["validation_source"]),
            int(row["validation_episode"]),
            str(row["problem_id"]),
            str(row["source"]),
        ),
    )
    problem_keys = []
    for row in ordered_rows:
        key = (str(row["validation_source"]), str(row["problem_id"]))
        if key not in problem_keys:
            problem_keys.append(key)
    problem_index = {key: index + 1 for index, key in enumerate(problem_keys)}
    sources = _ordered_sources(row["source"] for row in ordered_rows)
    markers = ["D", "s", "^", "o", "v", "P", "X", "*", "h", "p"]
    plt.figure(figsize=(11, 5))
    for source_index, source in enumerate(sources):
        source_rows = [row for row in ordered_rows if str(row["source"]) == source]
        x_values = [
            problem_index[(str(row["validation_source"]), str(row["problem_id"]))]
            for row in source_rows
        ]
        y_values = [float(row[score_field]) for row in source_rows]
        plt.scatter(
            x_values,
            y_values,
            label=_display_source_name(source),
            marker=markers[source_index % len(markers)],
            s=42,
            alpha=0.85,
        )
    plt.xlabel("Validation problem")
    plt.ylabel(ylabel)
    plt.title(f"{title} by method (lower is better)")
    plt.xticks(
        list(problem_index.values()),
        [key[1] for key in problem_keys],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_validation_best_source_counts(plt, path: Path, rows: Sequence[Mapping]) -> None:
    """Plot which source wins each latest validation problem."""

    counts: Dict[str, int] = {}
    for row in rows:
        source = str(row["best_source"])
        counts[source] = counts.get(source, 0) + 1
    sources = _ordered_sources(counts)
    plt.figure(figsize=(9, 4))
    plt.bar([_display_source_name(source) for source in sources], [counts[source] for source in sources])
    plt.xlabel("Best source")
    plt.ylabel("Validation wins")
    plt.title("Best source counts in latest validation")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_validation_agent_rank(plt, path: Path, rows: Sequence[Mapping]) -> None:
    """Plot agent rank in each latest validation problem."""

    ordered_rows = sorted(
        rows,
        key=lambda row: (str(row["validation_source"]), int(row["validation_episode"]), str(row["problem_id"])),
    )
    x_values = list(range(1, len(ordered_rows) + 1))
    y_values = [int(row["agent_rank"]) for row in ordered_rows]
    labels = [str(row["problem_id"]) for row in ordered_rows]
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel("Validation problem")
    plt.ylabel("Agent rank")
    plt.title("Agent greedy rank in latest validation (1 is best)")
    plt.xticks(x_values, labels, rotation=45, ha="right", fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _ordered_sources(sources: Sequence[object]) -> List[str]:
    """Return stable method order with the learned policy first."""

    unique = {str(source) for source in sources}
    preferred = ["agent_greedy", *PHASE1_SELF_LABEL_HEURISTIC_BANK]
    return [source for source in preferred if source in unique] + sorted(unique - set(preferred))


def _display_source_name(source: str) -> str:
    """Shorten source names for plot legends."""

    names = {
        "agent_greedy": "Proposed",
        "steel_first_balanced": "Steel",
        "cut_first_balanced": "Cut",
        "bevel_first_balanced": "Bevel",
        "long_cut_first_balanced": "LongCut",
        "steel_first_long_cut_preferred": "Steel+Long",
        "cut_first_long_cut_preferred": "Cut+Long",
        "bevel_first_long_cut_preferred": "Bevel+Long",
        "long_cut_first_long_cut_preferred": "Long+Long",
    }
    return names.get(source, source)


def _write_validation_summary(path: Path, rows: Sequence[Mapping]) -> None:
    """Write periodic holdout validation summary."""

    _write_csv(
        path,
        [
            "train_episode",
            "validation_episode",
            "validation_source",
            "problem_id",
            "block_count",
            "agent_score_json",
            "best_score_json",
            "best_source",
            "agent_is_best",
            "agent_rank",
            "candidate_count",
        ],
        rows,
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping]) -> None:
    """Write CSV rows with explicit fields."""

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    """Write JSONL rows."""

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
