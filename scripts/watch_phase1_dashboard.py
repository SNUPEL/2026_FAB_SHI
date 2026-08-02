"""Phase 1 대시보드를 validation 마다 재생성하는 워처.

validation_summary.csv 의 최대 train_episode 를 폴링하다가 값이 커지면(=새 validation)
build_phase1_dashboard 로 HTML 을 다시 쓴다. 학습이 target 에 도달하면 마지막으로 한 번
렌더하고 종료한다. 학습 코드는 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_phase1_dashboard as dash  # noqa: E402
from build_training_dashboard import _read_csv_rows, _to_int  # noqa: E402


def latest_validation_episode(run_dir: Path) -> int | None:
    rows = _read_csv_rows(run_dir / "validation_summary.csv")
    episodes = [ep for ep in (_to_int(r.get("train_episode")) for r in rows) if ep is not None]
    return max(episodes) if episodes else None


def current_train_episode(run_dir: Path) -> int | None:
    rows = _read_csv_rows(run_dir / "metrics.csv")
    if not rows:
        return None
    return _to_int(rows[-1].get("episode"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-episodes", type=int, default=20000)
    parser.add_argument("--output", default="output/dashboard/phase1_v2/dashboard.html")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--window", type=int, default=dash.DEFAULT_WINDOW)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[ERROR][watch_phase1_dashboard] cause=missing_run_dir path={run_dir}")
        raise RuntimeError(f"missing run directory: {run_dir}")

    def render() -> int:
        payload = dash.write_dashboard(run_dir, args.target_episodes, args.output, args.window)
        return payload["run"]["current_ep"]

    ep = render()
    last_val = latest_validation_episode(run_dir)
    print(f"[CHECK][watch_phase1_dashboard] initial render current_ep={ep} last_val={last_val}")

    while True:
        if (current_train_episode(run_dir) or 0) >= args.target_episodes:
            render()
            print("[CHECK][watch_phase1_dashboard] target reached, final render, exit")
            return 0
        time.sleep(args.interval)
        cur_val = latest_validation_episode(run_dir)
        if cur_val is not None and cur_val != last_val:
            ep = render()
            last_val = cur_val
            print(f"[CHECK][watch_phase1_dashboard] validation train_ep={cur_val} rendered current_ep={ep}")


if __name__ == "__main__":
    sys.exit(main())
