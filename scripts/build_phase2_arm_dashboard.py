#!/usr/bin/env python3
"""Phase 2 학습 run 하나를 (Baseline) Phase 2 대시보드와 동일한 형식의 HTML로 만든다.

`scripts/phase2_dashboard_template/`의 head/tail을 그대로 쓰고 `const D = {...}`만
run 디렉터리에서 계산해 끼워 넣는다. 차트 종류·축·색·집계식은 baseline과 동일하다.

집계 계약 (baseline 대시보드에서 역산해 수치 일치를 확인한 것):
- Bay별 best-rate = 그 Bay subproblem에서 `proposed_best_rank == 1`인 문제 비율(동률 포함).
- overall / mean_rank = 5개 Bay 값의 단순 평균(문제 수 가중이 아니다).
- agent 채택률 = subproblem_metrics.csv의 `best_source`를 greedy/sample/heuristic으로
  분류해 최근 `--adoption-window` episode 창에서 계산한 비율.
- loss/hv는 raw를 그대로 싣고 이동평균은 대시보드 JS가 계산한다.

MAIN(in-distribution) validation이 정식 지표이므로 `<run>/validation/`을 valind로,
generalization은 `<run>/validation_generalization/`을 val로 읽는다. main validation 도입
이전 run처럼 두 디렉터리 구성이 다르면 `--main-validation-dir`로 직접 지정한다.

사용 예:
    python scripts/build_phase2_arm_dashboard.py \
        --run-dir output/phase2_h128 \
        --title "(hidden_dim 128) Phase 2" \
        --out output/dashboard_h128.html
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

BAYS = ("22", "23", "24", "25", "trans")
TEMPLATE_DIR = Path(__file__).resolve().parent / "phase2_dashboard_template"


def fail(cause: str, detail: str) -> None:
    """AGENTS §4-5: 진단을 출력하고 예외를 던진다. 조용한 대체값을 쓰지 않는다."""

    print(f"[ERROR][build_phase2_arm_dashboard] cause={cause} {detail}")
    raise RuntimeError(f"{cause}: {detail}")


def read_rows(path: Path, required: Iterable[str]) -> list[dict]:
    if not path.is_file():
        fail("missing_csv", f"path={path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    missing = [name for name in required if name not in rows[0]]
    if missing:
        fail("missing_columns", f"path={path} missing={missing}")
    return rows


def downsample(points: list[list], target: int) -> list[list]:
    """차트 표본 수를 target 근처로 줄인다. 마지막 점은 항상 남긴다."""

    if len(points) <= target:
        return points
    step = max(1, round(len(points) / target))
    out = points[::step]
    if out[-1][0] != points[-1][0]:
        out.append(points[-1])
    return out


def training_series(run_dir: Path, adoption_window: int, target_points: int) -> dict:
    metrics = read_rows(run_dir / "metrics.csv", ("episode", "loss", "hard_violation_count"))
    if not metrics:
        fail("empty_metrics", f"path={run_dir / 'metrics.csv'}")
    loss = [[int(r["episode"]), round(float(r["loss"]), 4)] for r in metrics]
    hv = [[int(r["episode"]), int(float(r["hard_violation_count"]))] for r in metrics]

    sub = read_rows(run_dir / "subproblem_metrics.csv", ("episode", "best_source"))
    by_episode: dict[int, Counter] = defaultdict(Counter)
    for row in sub:
        source = row["best_source"]
        if source == "agent_greedy":
            kind = "greedy"
        elif source.startswith("agent_sample"):
            kind = "sample"
        else:
            kind = "heur"
        by_episode[int(row["episode"])][kind] += 1

    greedy: list[list] = []
    agent_all: list[list] = []
    heur: list[list] = []
    window: deque[Counter] = deque()
    total = Counter()
    for episode in sorted(by_episode):
        counts = by_episode[episode]
        window.append(counts)
        total.update(counts)
        while len(window) > adoption_window:
            total.subtract(window.popleft())
        n = sum(total.values())
        if n <= 0:
            continue
        g = 100.0 * total["greedy"] / n
        a = 100.0 * (total["greedy"] + total["sample"]) / n
        greedy.append([episode, round(g, 1)])
        agent_all.append([episode, round(a, 1)])
        heur.append([episode, round(100.0 * total["heur"] / n, 1)])

    return {
        "loss": downsample(loss, target_points),
        "hv": downsample(hv, target_points),
        "greedy": downsample(greedy, target_points),
        "sample": downsample(agent_all, target_points),
        "heur": downsample(heur, target_points),
        "last_episode": loss[-1][0],
        "loss_first": loss[0][1],
        "loss_latest": loss[-1][1],
    }


def validation_series(bay_history: Path) -> list[dict]:
    """checkpoint별 Bay best-rate + 5개 Bay 평균(overall/mean_rank)."""

    rows = read_rows(
        bay_history,
        ("train_episode", "bay_id", "proposed_best_rank"),
    )
    if not rows:
        return []
    per: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        per[int(row["train_episode"])][row["bay_id"]].append(int(row["proposed_best_rank"]))

    out: list[dict] = []
    for episode in sorted(per):
        entry: dict = {"ep": episode}
        rates: list[float] = []
        ranks: list[float] = []
        for bay in BAYS:
            values = per[episode].get(bay)
            if not values:
                # 실제 W/O가 없는 Bay는 0으로 채우지 않고 빈 값으로 남긴다(계약 §5).
                entry[bay] = None
                continue
            rate = 100.0 * sum(1 for v in values if v == 1) / len(values)
            entry[bay] = round(rate, 1)
            rates.append(rate)
            ranks.append(statistics.mean(values))
        if not rates:
            fail("no_bay_rows", f"path={bay_history} episode={episode}")
        entry["overall"] = round(statistics.mean(rates), 1)
        entry["best_rate"] = entry["overall"]
        entry["mean_rank"] = round(statistics.mean(ranks), 2)
        out.append(entry)
    return out


def run_meta(run_dir: Path, series: dict, target_episodes: int | None, running: bool) -> dict:
    summary_path = run_dir / "summary.json"
    target = target_episodes
    temperature = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        target = target or int(summary.get("episodes", 0)) or None
        temperature = summary.get("temperature_min")
    if target is None:
        fail("unknown_target_episodes", "summary.json이 없으면 --target-episodes를 넘겨야 한다")
    current = series["last_episode"]
    return {
        "current_ep": current,
        "target": target,
        "progress_pct": round(100.0 * current / target, 1),
        "running": running,
        "elapsed_h": None,
        "sec_per_ep": None,
        "eta_h": None,
        "loss_first": series["loss_first"],
        "loss_latest": series["loss_latest"],
        "temperature": temperature,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--eyebrow", default="")
    parser.add_argument("--sub", default="")
    parser.add_argument("--main-validation-dir", default="validation")
    parser.add_argument("--generalization-validation-dir", default="validation_generalization")
    parser.add_argument("--target-episodes", type=int, default=None)
    parser.add_argument("--adoption-window", type=int, default=25, help="agent 채택률 이동평균 창(episode)")
    parser.add_argument("--target-points", type=int, default=460, help="차트당 표본 수 상한")
    parser.add_argument("--running", action="store_true", help="진행 중 표시(생략하면 중단/완료)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        fail("missing_run_dir", f"path={run_dir}")

    series = training_series(run_dir, args.adoption_window, args.target_points)

    main_history = run_dir / args.main_validation_dir / "validation_bay_history.csv"
    gen_history = run_dir / args.generalization_validation_dir / "validation_bay_history.csv"
    valind = validation_series(main_history) if main_history.is_file() else []
    val = validation_series(gen_history) if gen_history.is_file() else []
    if not valind and not val:
        print(
            "[CHECK][build_phase2_arm_dashboard] validation 결과가 아직 없다 "
            f"(main={main_history} generalization={gen_history})"
        )

    indist = None
    if valind:
        best = max(valind, key=lambda v: v["overall"])
        indist = {
            "ep": best["ep"],
            "rate": best["overall"],
            "mean_rank": best["mean_rank"],
            "n": len(valind),
        }

    payload = {
        "cur": {
            "current_ep": series["last_episode"],
            "meta": run_meta(run_dir, series, args.target_episodes, args.running),
            "loss": series["loss"],
            "hv": series["hv"],
            "greedy": series["greedy"],
            "sample": series["sample"],
            "heur": series["heur"],
            "val": val,
            "valind": valind,
            "indist": indist,
        }
    }

    head = (TEMPLATE_DIR / "head.html").read_text(encoding="utf-8")
    tail = (TEMPLATE_DIR / "tail.html").read_text(encoding="utf-8")
    head = (
        head.replace("{{TITLE}}", args.title)
        .replace("{{EYEBROW}}", args.eyebrow or args.title)
        .replace("{{SUB}}", args.sub or "")
        .replace("{{RUNDIR}}", str(run_dir))
    )
    body = head + "const D = " + json.dumps(payload, ensure_ascii=False) + ";" + tail

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print(
        "[CHECK][build_phase2_arm_dashboard] "
        f"run_dir={run_dir} out={out_path} current_ep={series['last_episode']} "
        f"main_validation_points={len(valind)} generalization_points={len(val)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
