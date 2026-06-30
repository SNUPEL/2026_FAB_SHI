"""Phase 1 pointer-policy imitation training.

Input is the self-label package created by:

    python3 main.py phase1-mdp-trace ...

The trainer reads `phase1_action_table.jsonl`, trains `Phase1PointerPolicy`
against `selected_action_index`, and writes a small checkpoint plus metrics.

주의:
- 이 파일은 고정 teacher imitation이다.
- self-labeling처럼 agent sample과 휴리스틱을 경쟁시키지 않는다.
- 입력 JSONL에 이미 정답 index가 들어 있고, 이 index를 그대로 맞히는 supervised baseline이다.
"""

# LINE-BY-LINE: 미래 타입 힌트를 문자열로 늦게 평가합니다. 사용: Python annotation 호환성 유지.
from __future__ import annotations

# LINE-BY-LINE: metrics.csv 저장에 사용합니다.
import csv
# LINE-BY-LINE: action table JSONL과 summary JSON 저장/읽기에 사용합니다.
import json
# LINE-BY-LINE: feature normalization에서 finite check 등에 사용합니다.
import math
# LINE-BY-LINE: supervised example record class 정의에 사용합니다.
from dataclasses import dataclass
# LINE-BY-LINE: 입력 action table과 output path를 OS 독립적으로 다루기 위해 사용합니다.
from pathlib import Path
# LINE-BY-LINE: 함수 인자/반환 타입을 명확히 표시하기 위한 typing import입니다.
from typing import Dict, List, Sequence

# LINE-BY-LINE: PyTorch model, tensor, optimizer, checkpoint 저장에 사용합니다.
import torch
# LINE-BY-LINE: cross entropy loss 계산에 사용합니다.
import torch.nn.functional as F

# LINE-BY-LINE: split-action pointer policy입니다. imitation baseline은 이 network를 학습합니다.
from Train.network.phase1_pointer import Phase1PointerPolicy
# LINE-BY-LINE: action table feature dict를 고정 순서 vector로 바꾸기 위한 feature 이름 목록입니다.
from Utils.phase1_mdp import (
    PHASE1_BAY_FEATURE_NAMES,
    PHASE1_BLOCK_FEATURE_NAMES,
    PHASE1_ENV_FEATURE_NAMES,
)


# LINE-BY-LINE: action table JSONL의 한 row를 학습에 필요한 tensor 입력 형태로 담는 immutable record입니다.
@dataclass(frozen=True)
class Phase1ImitationExample:
    """One supervised Phase 1 decision example."""

    # LINE-BY-LINE: decision phase입니다. 값은 `SELECT_BLOCK` 또는 `SELECT_BAY`.
    phase: str
    # LINE-BY-LINE: 현재 phase의 후보 feature matrix입니다.
    candidate_features: List[List[float]]
    # LINE-BY-LINE: 현재 환경 feature vector입니다.
    env_features: List[float]
    # LINE-BY-LINE: teacher가 선택한 후보 index입니다. cross entropy target입니다.
    selected_action_index: int
    # LINE-BY-LINE: `SELECT_BAY` phase에서 선택된 block feature입니다. `SELECT_BLOCK`에서는 None입니다.
    selected_block_features: List[float] | None


# LINE-BY-LINE: Phase 1 imitation baseline 학습 entry function입니다.
def train_phase1_pointer_imitation(
    action_table_path: str | Path,
    output_dir: str | Path,
    eval_action_table_path: str | Path | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    seed: int = 0,
) -> Dict:
    """Train a small Phase 1 pointer policy from self-label examples.

    입력:
    - `action_table_path`: `phase1-mdp-trace` 또는 episode dataset이 만든 JSONL.
    - `eval_action_table_path`: 선택 사항. 있으면 holdout accuracy도 계산한다.

    출력:
    - checkpoint, metrics.csv, summary.json.
    """

    # LINE-BY-LINE: epoch 수가 0 이하이면 학습 loop가 의미 없으므로 실패합니다.
    if epochs <= 0:
        print(f"[ERROR][phase1_imitation.train_phase1_pointer_imitation] cause=non_positive_epochs epochs={epochs}")
        raise ValueError("epochs must be positive")
    # LINE-BY-LINE: learning rate가 0 이하이면 optimizer update가 잘못되므로 실패합니다.
    if lr <= 0:
        print(f"[ERROR][phase1_imitation.train_phase1_pointer_imitation] cause=non_positive_lr lr={lr}")
        raise ValueError("lr must be positive")

    # LINE-BY-LINE: PyTorch seed를 고정해 model initialization을 재현 가능하게 합니다.
    torch.manual_seed(seed)
    # LINE-BY-LINE: train action table JSONL을 supervised example 목록으로 읽습니다.
    examples = load_phase1_imitation_examples(action_table_path)
    # LINE-BY-LINE: eval action table이 있으면 별도 example 목록으로 읽고, 없으면 빈 list입니다.
    eval_examples = load_phase1_imitation_examples(eval_action_table_path) if eval_action_table_path else []
    # LINE-BY-LINE: train example 기준으로 feature 정규화 평균/표준편차를 계산합니다.
    normalization = _fit_normalization(examples)
    # LINE-BY-LINE: output directory를 생성합니다.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # LINE-BY-LINE: split-action pointer policy를 생성합니다.
    model = Phase1PointerPolicy(
        block_feature_dim=len(PHASE1_BLOCK_FEATURE_NAMES),
        bay_feature_dim=len(PHASE1_BAY_FEATURE_NAMES),
        env_feature_dim=len(PHASE1_ENV_FEATURE_NAMES),
        hidden_dim=hidden_dim,
    )
    # LINE-BY-LINE: model parameter를 학습할 Adam optimizer입니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # LINE-BY-LINE: epoch별 loss/accuracy row를 누적합니다.
    metrics_rows: List[Dict] = []

    # LINE-BY-LINE: epoch loop입니다. 모든 example을 매 epoch 한 번씩 사용합니다.
    for epoch in range(1, epochs + 1):
        # LINE-BY-LINE: epoch 전체 loss를 tensor로 누적합니다.
        total_loss = torch.zeros((), dtype=torch.float32)
        # LINE-BY-LINE: epoch 내 정답 예측 개수를 셉니다.
        correct = 0
        # LINE-BY-LINE: supervised example을 하나씩 학습합니다.
        for example in examples:
            # LINE-BY-LINE: 현재 example의 candidate logits를 계산합니다.
            logits = _score_example(model, example, normalization)
            # LINE-BY-LINE: selected_action_index를 cross entropy target tensor로 만듭니다.
            target = torch.tensor([example.selected_action_index], dtype=torch.long)
            # LINE-BY-LINE: target index가 후보 수보다 크면 action table이 깨진 것이므로 실패합니다.
            if example.selected_action_index >= logits.numel():
                print(
                    "[ERROR][phase1_imitation.train_phase1_pointer_imitation] "
                    f"cause=target_out_of_range phase={example.phase} "
                    f"target={example.selected_action_index} logits={logits.numel()}"
                )
                raise RuntimeError("Phase 1 imitation target out of range")
            # LINE-BY-LINE: 현재 example loss를 epoch loss에 더합니다.
            total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), target)
            # LINE-BY-LINE: argmax가 teacher index와 같으면 correct를 1 증가시킵니다.
            correct += int(torch.argmax(logits).item() == example.selected_action_index)

        # LINE-BY-LINE: 평균 loss를 계산합니다.
        loss = total_loss / len(examples)
        # LINE-BY-LINE: 이전 gradient를 지웁니다.
        optimizer.zero_grad()
        # LINE-BY-LINE: backpropagation으로 gradient를 계산합니다.
        loss.backward()
        # LINE-BY-LINE: gradient 폭주를 막기 위해 norm을 1.0으로 clip합니다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # LINE-BY-LINE: optimizer가 model parameter를 업데이트합니다.
        optimizer.step()
        # LINE-BY-LINE: 업데이트 후 train set 기준 loss/accuracy를 다시 계산합니다.
        train_eval = _evaluate_examples(model, examples, normalization)
        # LINE-BY-LINE: epoch metric row를 누적합니다.
        metrics_rows.append(
            {
                "epoch": epoch,
                "loss": train_eval["loss"],
                "accuracy": train_eval["accuracy"],
                "example_count": len(examples),
            }
        )

    final_train_eval = _evaluate_examples(model, examples, normalization)
    final_eval = _evaluate_examples(model, eval_examples, normalization) if eval_examples else None
    checkpoint_path = output_path / "phase1_pointer.pt"
    metrics_csv = output_path / "metrics.csv"
    summary_json = output_path / "summary.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "block_feature_names": PHASE1_BLOCK_FEATURE_NAMES,
            "bay_feature_names": PHASE1_BAY_FEATURE_NAMES,
            "env_feature_names": PHASE1_ENV_FEATURE_NAMES,
            "hidden_dim": hidden_dim,
            "normalization": normalization,
        },
        checkpoint_path,
    )
    _write_metrics(metrics_csv, metrics_rows)
    summary = {
        "action_table_path": str(action_table_path),
        "checkpoint_path": str(checkpoint_path),
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
        "epochs": epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "example_count": len(examples),
        "final_loss": final_train_eval["loss"],
        "final_accuracy": final_train_eval["accuracy"],
        "final_select_block_accuracy": final_train_eval["select_block_accuracy"],
        "final_select_bay_accuracy": final_train_eval["select_bay_accuracy"],
    }
    if final_eval is not None:
        summary["eval_action_table_path"] = str(eval_action_table_path)
        summary["eval_example_count"] = len(eval_examples)
        summary["eval_loss"] = final_eval["loss"]
        summary["eval_accuracy"] = final_eval["accuracy"]
        summary["eval_select_block_accuracy"] = final_eval["select_block_accuracy"]
        summary["eval_select_bay_accuracy"] = final_eval["select_bay_accuracy"]
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)

    print(
        "[CHECK][phase1_imitation.train_phase1_pointer_imitation] "
        f"examples={len(examples)} epochs={epochs} final_loss={summary['final_loss']:.6f} "
        f"final_accuracy={summary['final_accuracy']:.6f} "
        f"eval_accuracy={summary.get('eval_accuracy', 'none')} checkpoint={checkpoint_path}"
    )
    return summary


def load_phase1_imitation_examples(action_table_path: str | Path) -> List[Phase1ImitationExample]:
    """Load Phase 1 JSONL examples without default-value fallback."""

    path = Path(action_table_path)
    if not path.exists():
        print(f"[ERROR][phase1_imitation.load_phase1_imitation_examples] cause=missing_file path={path}")
        raise FileNotFoundError(path)
    examples: List[Phase1ImitationExample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(_parse_example(row, line_no))
    if not examples:
        print(f"[ERROR][phase1_imitation.load_phase1_imitation_examples] cause=no_examples path={path}")
        raise RuntimeError("Phase 1 imitation examples are empty")
    return examples


def _parse_example(row: Dict, line_no: int) -> Phase1ImitationExample:
    """Parse one action-table JSON row."""

    phase = _require_phase(row.get("phase"), line_no)
    selected_action_index = _require_non_negative_int(row.get("selected_action_index"), "selected_action_index", line_no)
    env_features = _features_from_dict(row.get("env_features"), PHASE1_ENV_FEATURE_NAMES, "env_features", line_no)
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        print(
            "[ERROR][phase1_imitation._parse_example] "
            f"cause=missing_candidates line_no={line_no} candidates_type={type(candidates).__name__}"
        )
        raise RuntimeError(f"missing candidates at line {line_no}")
    candidate_features = [_candidate_features(candidate, phase, line_no) for candidate in candidates]
    if selected_action_index >= len(candidate_features):
        print(
            "[ERROR][phase1_imitation._parse_example] "
            f"cause=selected_index_out_of_candidates line_no={line_no} "
            f"selected_action_index={selected_action_index} candidate_count={len(candidate_features)}"
        )
        raise RuntimeError(f"selected action index out of range at line {line_no}")

    selected_block_features = None
    if phase == "SELECT_BAY":
        selected_block_features = _features_from_dict(
            row.get("selected_block_features"),
            PHASE1_BLOCK_FEATURE_NAMES,
            "selected_block_features",
            line_no,
        )
    return Phase1ImitationExample(
        phase=phase,
        candidate_features=candidate_features,
        env_features=env_features,
        selected_action_index=selected_action_index,
        selected_block_features=selected_block_features,
    )


def _score_example(
    model: Phase1PointerPolicy,
    example: Phase1ImitationExample,
    normalization: Dict[str, Dict[str, List[float]]],
) -> torch.Tensor:
    """Run the correct pointer head for one example."""

    candidate_tensor = torch.tensor(example.candidate_features, dtype=torch.float32)
    env_tensor = torch.tensor(example.env_features, dtype=torch.float32)
    if example.phase == "SELECT_BLOCK":
        return model.score_blocks(
            _normalize_tensor(candidate_tensor, normalization["block"]),
            _normalize_tensor(env_tensor, normalization["env"]),
        )
    if example.phase == "SELECT_BAY":
        if example.selected_block_features is None:
            print("[ERROR][phase1_imitation._score_example] cause=missing_selected_block_features")
            raise RuntimeError("SELECT_BAY example requires selected_block_features")
        selected_block_tensor = torch.tensor(example.selected_block_features, dtype=torch.float32)
        return model.score_bays(
            _normalize_tensor(candidate_tensor, normalization["bay"]),
            _normalize_tensor(env_tensor, normalization["env"]),
            _normalize_tensor(selected_block_tensor, normalization["block"]),
        )
    print(f"[ERROR][phase1_imitation._score_example] cause=unknown_phase phase={example.phase}")
    raise RuntimeError(f"unknown Phase 1 imitation phase: {example.phase}")


def _evaluate_examples(
    model: Phase1PointerPolicy,
    examples: Sequence[Phase1ImitationExample],
    normalization: Dict[str, Dict[str, List[float]]],
) -> Dict[str, float]:
    """Evaluate loss and action-index accuracy without updating weights."""

    if not examples:
        print("[ERROR][phase1_imitation._evaluate_examples] cause=no_examples")
        raise RuntimeError("cannot evaluate empty Phase 1 imitation examples")
    total_loss = 0.0
    correct = 0
    phase_stats = {
        "SELECT_BLOCK": {"count": 0, "correct": 0, "loss": 0.0},
        "SELECT_BAY": {"count": 0, "correct": 0, "loss": 0.0},
    }
    with torch.no_grad():
        for example in examples:
            logits = _score_example(model, example, normalization)
            target = torch.tensor([example.selected_action_index], dtype=torch.long)
            loss = float(F.cross_entropy(logits.unsqueeze(0), target).item())
            is_correct = int(torch.argmax(logits).item() == example.selected_action_index)
            total_loss += loss
            correct += is_correct
            phase_stats[example.phase]["count"] += 1
            phase_stats[example.phase]["correct"] += is_correct
            phase_stats[example.phase]["loss"] += loss
    return {
        "loss": total_loss / len(examples),
        "accuracy": correct / len(examples),
        "select_block_loss": _phase_average(phase_stats["SELECT_BLOCK"], "loss"),
        "select_block_accuracy": _phase_average(phase_stats["SELECT_BLOCK"], "correct"),
        "select_bay_loss": _phase_average(phase_stats["SELECT_BAY"], "loss"),
        "select_bay_accuracy": _phase_average(phase_stats["SELECT_BAY"], "correct"),
    }


def _phase_average(stats: Dict[str, float], field: str) -> float:
    """Return per-phase average with a hard failure on missing phase rows."""

    count = int(stats["count"])
    if count <= 0:
        print(f"[ERROR][phase1_imitation._phase_average] cause=no_phase_rows field={field}")
        raise RuntimeError("Phase 1 evaluation is missing a decision phase")
    return float(stats[field]) / count


def _candidate_features(candidate: Dict, phase: str, line_no: int) -> List[float]:
    """Return candidate feature vector for the current decision phase."""

    if not isinstance(candidate, dict):
        print(
            "[ERROR][phase1_imitation._candidate_features] "
            f"cause=invalid_candidate line_no={line_no} candidate_type={type(candidate).__name__}"
        )
        raise RuntimeError(f"invalid candidate at line {line_no}")
    features = candidate.get("features")
    expected_dim = len(PHASE1_BLOCK_FEATURE_NAMES) if phase == "SELECT_BLOCK" else len(PHASE1_BAY_FEATURE_NAMES)
    if not isinstance(features, list) or len(features) != expected_dim:
        print(
            "[ERROR][phase1_imitation._candidate_features] "
            f"cause=invalid_candidate_features line_no={line_no} phase={phase} "
            f"features_type={type(features).__name__} expected_dim={expected_dim}"
        )
        raise RuntimeError(f"invalid candidate features at line {line_no}")
    return [float(value) for value in features]


def _fit_normalization(examples: Sequence[Phase1ImitationExample]) -> Dict[str, Dict[str, List[float]]]:
    """Fit simple mean/std normalization from the current training examples."""

    block_rows: List[List[float]] = []
    bay_rows: List[List[float]] = []
    env_rows: List[List[float]] = []
    for example in examples:
        env_rows.append(example.env_features)
        if example.phase == "SELECT_BLOCK":
            block_rows.extend(example.candidate_features)
        elif example.phase == "SELECT_BAY":
            bay_rows.extend(example.candidate_features)
            if example.selected_block_features is None:
                print("[ERROR][phase1_imitation._fit_normalization] cause=missing_selected_block_features")
                raise RuntimeError("SELECT_BAY normalization requires selected_block_features")
            block_rows.append(example.selected_block_features)
    return {
        "block": _feature_stats(block_rows, len(PHASE1_BLOCK_FEATURE_NAMES), "block"),
        "bay": _feature_stats(bay_rows, len(PHASE1_BAY_FEATURE_NAMES), "bay"),
        "env": _feature_stats(env_rows, len(PHASE1_ENV_FEATURE_NAMES), "env"),
    }


def _feature_stats(rows: Sequence[Sequence[float]], expected_dim: int, name: str) -> Dict[str, List[float]]:
    """Return mean/std for one feature group."""

    if not rows:
        print(f"[ERROR][phase1_imitation._feature_stats] cause=no_rows name={name}")
        raise RuntimeError(f"normalization has no rows for {name}")
    if any(len(row) != expected_dim for row in rows):
        print(
            "[ERROR][phase1_imitation._feature_stats] "
            f"cause=dimension_mismatch name={name} expected_dim={expected_dim}"
        )
        raise RuntimeError(f"normalization dimension mismatch for {name}")
    count = len(rows)
    mean = [sum(row[index] for row in rows) / count for index in range(expected_dim)]
    std = []
    for index in range(expected_dim):
        variance = sum((row[index] - mean[index]) ** 2 for row in rows) / count
        std.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
    return {"mean": mean, "std": std}


def _normalize_tensor(tensor: torch.Tensor, stats: Dict[str, List[float]]) -> torch.Tensor:
    """Apply feature normalization fitted by `_fit_normalization`."""

    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    return (tensor - mean) / std


def _features_from_dict(value, field_names: Sequence[str], field_name: str, line_no: int) -> List[float]:
    """Read named features in the exact model order."""

    if not isinstance(value, dict):
        print(
            "[ERROR][phase1_imitation._features_from_dict] "
            f"cause=missing_feature_dict field={field_name} line_no={line_no} "
            f"value_type={type(value).__name__}"
        )
        raise RuntimeError(f"missing feature dict: {field_name} at line {line_no}")
    missing = [name for name in field_names if name not in value]
    if missing:
        print(
            "[ERROR][phase1_imitation._features_from_dict] "
            f"cause=missing_feature_keys field={field_name} line_no={line_no} missing={missing}"
        )
        raise RuntimeError(f"missing feature keys for {field_name} at line {line_no}: {missing}")
    return [float(value[name]) for name in field_names]


def _require_phase(value, line_no: int) -> str:
    """Validate decision phase."""

    if value not in {"SELECT_BLOCK", "SELECT_BAY"}:
        print(f"[ERROR][phase1_imitation._require_phase] cause=invalid_phase line_no={line_no} value={value}")
        raise RuntimeError(f"invalid Phase 1 imitation phase at line {line_no}: {value}")
    return str(value)


def _require_non_negative_int(value, field_name: str, line_no: int) -> int:
    """Validate non-negative integer fields."""

    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_imitation._require_non_negative_int] "
            f"cause=invalid_int field={field_name} line_no={line_no} value={value}"
        )
        raise RuntimeError(f"invalid integer field {field_name} at line {line_no}") from exc
    if number < 0:
        print(
            "[ERROR][phase1_imitation._require_non_negative_int] "
            f"cause=negative_int field={field_name} line_no={line_no} value={value}"
        )
        raise RuntimeError(f"negative integer field {field_name} at line {line_no}")
    return number


def _write_metrics(path: Path, rows: Sequence[Dict]) -> None:
    """Write training metrics CSV."""

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "accuracy", "example_count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
