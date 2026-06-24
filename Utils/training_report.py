"""학습 smoke 결과를 사람이 확인 가능한 파일로 저장한다.

이 모듈은 학습 성능을 포장하기 위한 것이 아니라,
episode별 reward/makespan/load imbalance가 실제로 어떤 값을 냈는지
CSV/JSON/HTML로 남겨 디버깅할 수 있게 하는 역할이다.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Dict, List


METRIC_FIELDS = [
    "episode",
    "total_reward",
    "makespan",
    "load_imbalance",
    "scheduled_jobs",
    "transition_count",
    "actor_loss",
    "value_loss",
    "entropy",
]


def _require_metric_fields(metrics: List[Dict]) -> None:
    """metric row가 report에 필요한 모든 필드를 갖는지 확인한다."""

    if not metrics:
        print("[ERROR][training_report._require_metric_fields] cause=empty_metrics")
        raise RuntimeError("training metrics are empty")
    for row_index, row in enumerate(metrics):
        missing = [field for field in METRIC_FIELDS if field not in row]
        if missing:
            print(
                "[ERROR][training_report._require_metric_fields] "
                f"cause=missing_metric_fields row_index={row_index} missing={missing}"
            )
            raise KeyError(f"missing metric fields: {missing}")


def _write_metrics_csv(path: Path, metrics: List[Dict]) -> None:
    """episode metrics를 CSV로 저장한다."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in metrics:
            writer.writerow({field: row[field] for field in METRIC_FIELDS})


def _polyline_points(metrics: List[Dict], field: str, width: int, height: int) -> str:
    """HTML SVG chart에서 사용할 polyline 좌표 문자열을 만든다."""

    values = [float(row[field]) for row in metrics]
    min_value = min(values)
    max_value = max(values)
    value_span = max(max_value - min_value, 1e-9)
    x_span = max(len(values) - 1, 1)
    points = []
    for index, value in enumerate(values):
        x = 24 + (width - 48) * index / x_span
        y = height - 24 - (height - 48) * (value - min_value) / value_span
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _write_html_report(path: Path, metrics: List[Dict], summary: Dict) -> None:
    """브라우저에서 바로 열 수 있는 간단한 학습 report HTML을 저장한다."""

    width = 900
    chart_height = 170
    table_rows = []
    for row in metrics:
        table_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row[field]))}</td>" for field in METRIC_FIELDS)
            + "</tr>"
        )
    chart_sections = []
    for label, field in (
        ("Reward", "total_reward"),
        ("Makespan", "makespan"),
        ("Load Imbalance", "load_imbalance"),
    ):
        points = _polyline_points(metrics, field, width=width, height=chart_height)
        chart_sections.append(
            f"""
            <section class="chart-card">
              <h2>{html.escape(label)}</h2>
              <svg viewBox="0 0 {width} {chart_height}" role="img" aria-label="{html.escape(label)} curve">
                <line x1="24" y1="{chart_height - 24}" x2="{width - 24}" y2="{chart_height - 24}" />
                <line x1="24" y1="24" x2="24" y2="{chart_height - 24}" />
                <polyline points="{points}" />
              </svg>
            </section>
            """
        )

    body = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>PMSP Training Report</title>
  <style>
    body {{ margin: 0; background: #f4efe7; color: #1e2525; font-family: Georgia, 'Noto Serif KR', serif; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 36px 24px 60px; }}
    h1 {{ margin: 0 0 8px; font-size: 36px; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .pill {{ padding: 14px 16px; background: #fffaf2; border: 1px solid #d9cbb5; border-radius: 16px; }}
    .pill b {{ display: block; font-size: 12px; text-transform: uppercase; color: #6a5a43; }}
    .chart-card {{ margin: 18px 0; padding: 18px; background: #fffaf2; border: 1px solid #d9cbb5; border-radius: 18px; }}
    svg {{ width: 100%; height: auto; }}
    line {{ stroke: #b9aa94; stroke-width: 1; }}
    polyline {{ fill: none; stroke: #315f5b; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 18px; background: #fffaf2; }}
    th, td {{ border: 1px solid #d9cbb5; padding: 8px 10px; text-align: right; font-size: 13px; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #efe2cc; }}
  </style>
</head>
<body>
  <main>
    <h1>PMSP Training Report</h1>
    <p>학습 smoke 결과를 숨기지 않고 episode별 metric으로 표시한다.</p>
    <section class="meta">
      <div class="pill"><b>algorithm</b>{html.escape(str(summary["algorithm"]))}</div>
      <div class="pill"><b>episodes</b>{html.escape(str(summary["episodes"]))}</div>
      <div class="pill"><b>best makespan</b>{html.escape(str(summary["best_makespan"]))}</div>
      <div class="pill"><b>checkpoint</b>{html.escape(str(summary["checkpoint_path"]))}</div>
    </section>
    {''.join(chart_sections)}
    <table>
      <thead><tr>{''.join(f'<th>{html.escape(field)}</th>' for field in METRIC_FIELDS)}</tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def write_training_report(
    *,
    metrics: List[Dict],
    output_dir: str,
    algorithm: str,
    checkpoint_path: str,
) -> Dict[str, str]:
    """학습 metric을 CSV/JSON/HTML로 저장하고 경로를 반환한다."""

    _require_metric_fields(metrics)
    report_dir = Path(output_dir) / "training_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = report_dir / "metrics.csv"
    summary_json = report_dir / "summary.json"
    report_html = report_dir / "training_report.html"
    summary = {
        "algorithm": algorithm,
        "episodes": len(metrics),
        "best_makespan": min(float(row["makespan"]) for row in metrics),
        "best_reward": max(float(row["total_reward"]) for row in metrics),
        "checkpoint_path": checkpoint_path,
        "metrics_csv": str(metrics_csv),
        "report_html": str(report_html),
    }
    _write_metrics_csv(metrics_csv, metrics)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump({**summary, "metrics": metrics}, handle, ensure_ascii=False, indent=2)
    _write_html_report(report_html, metrics, summary)
    print(
        "[CHECK][training_report.write_training_report] "
        f"rows={len(metrics)} output={report_html}"
    )
    return {
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
        "report_html": str(report_html),
    }
