"""Draw a presentation PNG for the current hierarchical PMSP graph MDP."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


FONT_CANDIDATES = [
    Path("/mnt/c/Windows/Fonts/malgun.ttf"),
    Path("/mnt/c/Windows/Fonts/NotoSansKR-VF.ttf"),
    Path("/mnt/c/Windows/Fonts/gulim.ttc"),
]


def _configure_font() -> None:
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    print(f"[ERROR][draw_phase_graph_environment._configure_font] cause=korean_font_not_found candidates={FONT_CANDIDATES}")
    raise RuntimeError("Korean font is required to draw the PMSP graph environment")


def _box(ax, x, y, w, h, text, fc="#ffffff", ec="#6f7f8f", lw=1.6, size=10, weight="normal", color="#111827"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=color)
    return patch


def _label_box(ax, x, y, w, h, title, lines, fc="#f8fafc", ec="#cbd5e1", title_fc="#1f77b4"):
    _box(ax, x, y, w, h - 0.035, "\n".join(lines), fc=fc, ec=ec, size=9)
    _box(ax, x, y + h - 0.045, w, 0.045, title, fc=title_fc, ec=title_fc, size=11, weight="bold", color="white")


def _node(ax, x, y, label, fc, ec, r=0.022, size=9, weight="bold"):
    circle = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.8)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center", fontsize=size, weight=weight, color="#111827")
    return circle


def _arrow(ax, xy1, xy2, color="#334155", lw=1.3, style="-", alpha=1.0, rad=0.0, ms=10):
    arrow = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=style,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def _line(ax, xy1, xy2, color="#94a3b8", lw=1.1, style="-", alpha=1.0):
    ax.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], linestyle=style, color=color, linewidth=lw, alpha=alpha)


def _draw_left_dec_pomdp(ax):
    _box(ax, 0.025, 0.885, 0.37, 0.07, "절단 공장 계층형 Graph MDP / Dec-POMDP", fc="#1f77b4", ec="#1f77b4", size=16, weight="bold", color="white")
    ax.text(
        0.03,
        0.825,
        "• Phase 1: 블록→Bay 그룹핑, Phase 2: W/O→설비 배정\n"
        "• 에이전트 간 전달은 요약 메시지로 제한\n"
        "• 공동 목표: 부하평준화 + Phase 2-2 batch/makespan",
        ha="left",
        va="top",
        fontsize=11,
        color="#111827",
        linespacing=1.45,
    )
    rows = [
        ("지역 관찰", "Phase 1\nBlock node + Bay node\nBlock→Bay 후보 edge\n현재 Bay 부하", "Phase 2\nW/O node + Machine node\nW/O→Machine 후보 edge\nPhase 1 배정 Bay"),
        ("분산 행동", "SELECT_EDGE(Block, Bay)\n= 블록 선택 + Bay 선택", "SELECT_EDGE(W/O, Machine)\n= W/O 설비 배정"),
        ("공동 점수", "Bay별 gap 최소화\n강재 → 절단장 → 베벨\n장척 Bay24 penalty", "Machine별 gap 최소화\nW/O수 → TACT → 절단장 → 베벨\nBatch/makespan feedback"),
    ]
    y = 0.585
    for title, p1, p2 in rows:
        _box(ax, 0.03, y, 0.08, 0.095, title, fc="#6b7280", ec="#6b7280", size=11, weight="bold", color="white")
        _box(ax, 0.125, y, 0.13, 0.095, p1, fc="#ffffff", ec="#d1d5db", size=9)
        _box(ax, 0.265, y, 0.13, 0.095, p2, fc="#ffffff", ec="#d1d5db", size=9)
        y -= 0.135
    _box(
        ax,
        0.075,
        0.095,
        0.28,
        0.115,
        "Phase 1 → Phase 2 메시지\nblock→Bay, block feature, allowed Bay, 계열/장척/광폭 flag\npolicy logits와 전체 machine timeline은 전달하지 않음",
        fc="#eef6ff",
        ec="#60a5fa",
        size=9,
    )


def _draw_phase1_graph(ax):
    _box(ax, 0.43, 0.855, 0.24, 0.045, "Phase 1: Block-Bay grouping graph", fc="#f97316", ec="#f97316", size=13, weight="bold", color="white")
    blocks = [("B1", 0.45, 0.79), ("B2", 0.45, 0.71), ("B3", 0.45, 0.63), ("B4", 0.45, 0.55)]
    bays = [("22", 0.62, 0.80), ("23", 0.62, 0.72), ("24", 0.62, 0.64), ("25", 0.62, 0.56)]
    for label, x, y in blocks:
        _node(ax, x, y, label, "#ffedd5", "#f97316")
    for label, x, y in bays:
        _node(ax, x, y, f"Bay\n{label}", "#dbeafe", "#2563eb", r=0.026, size=8)
    for _, bx, by in blocks:
        for bay, mx, my in bays[:3]:
            if bay == "24" and by >= 0.78:
                _arrow(ax, (bx + 0.022, by), (mx - 0.027, my), color="#ef4444", lw=1.2, style="--", alpha=0.65, ms=8)
            else:
                _arrow(ax, (bx + 0.022, by), (mx - 0.027, my), color="#f97316", lw=1.0, alpha=0.8, ms=8)
    _arrow(ax, (0.472, 0.79), (0.593, 0.80), color="#dc2626", lw=3.0, ms=13)
    ax.text(0.505, 0.83, "selected edge", fontsize=8, color="#dc2626", weight="bold")
    _label_box(
        ax,
        0.69,
        0.62,
        0.285,
        0.205,
        "Phase 1 node/edge feature",
        [
            "Block node: 강재수량, 절단장, 베벨수량, 장척, 길이, 두께",
            "Bay node: 현재 강재/절단장/베벨/블록 부하",
            "Edge: block 물량 + Bay 현재부하 + 배정 후 gap",
            "Hard mask: 광폭/Bay 가능조건, 장척 Bay24 옵션",
        ],
        title_fc="#f97316",
    )
    _box(ax, 0.69, 0.545, 0.285, 0.055, "Action = SELECT_EDGE(Block, Bay)", fc="#fff7ed", ec="#fdba74", size=10, weight="bold")


def _draw_phase2_graph(ax):
    _box(ax, 0.43, 0.445, 0.24, 0.045, "Phase 2-1: W/O-Machine graph", fc="#2563eb", ec="#2563eb", size=12, weight="bold", color="white")
    wos = [("W1", 0.45, 0.375), ("W2", 0.45, 0.315), ("W3", 0.45, 0.255), ("W4", 0.45, 0.195)]
    machines = [("PLS21", 0.62, 0.385, "22"), ("PLS22", 0.62, 0.325, "22"), ("PLS31", 0.62, 0.265, "23"), ("PLS41", 0.62, 0.205, "24")]
    for label, x, y in wos:
        _node(ax, x, y, label, "#dcfce7", "#16a34a")
    for label, x, y, bay in machines:
        _node(ax, x, y, label, "#dbeafe", "#2563eb", r=0.027, size=7)
        ax.text(x + 0.045, y, f"Bay {bay}", fontsize=8, va="center", color="#475569")
    for i, (_, wx, wy) in enumerate(wos):
        for j, (_, mx, my, bay) in enumerate(machines):
            if (i < 2 and bay == "22") or (i >= 2 and bay in {"23", "24"}):
                _arrow(ax, (wx + 0.022, wy), (mx - 0.028, my), color="#2563eb", lw=1.0, alpha=0.75, ms=8)
    _arrow(ax, (0.472, 0.36), (0.592, 0.39), color="#1d4ed8", lw=3.0, ms=13)
    _label_box(
        ax,
        0.69,
        0.22,
        0.285,
        0.20,
        "Phase 2 node/edge feature",
        [
            "W/O node: TACT_TIME, LTH, CUT_LTH, BV_QTY, THK",
            "Machine node: Bay, 현재 W/O/TACT/절단장/베벨 부하",
            "Edge: 같은 Phase1 Bay, 설비 가능조건, 선호도",
            "Action = SELECT_EDGE(W/O, Machine)",
        ],
        title_fc="#2563eb",
    )
    _box(ax, 0.69, 0.145, 0.285, 0.055, "Phase 1 message가 W/O 후보 machine을 제한", fc="#eff6ff", ec="#93c5fd", size=10, weight="bold")


def _draw_batch_decoder(ax):
    _box(ax, 0.43, 0.085, 0.24, 0.045, "Phase 2-2: Batch / Sequencing decoder", fc="#64748b", ec="#64748b", size=11, weight="bold", color="white")
    for y, m in [(0.02, "PLS21")]:
        ax.text(0.43, y + 0.027, m, fontsize=9, weight="bold", ha="left", va="center", color="#111827")
        _box(ax, 0.49, y, 0.075, 0.052, "Batch 1\nW1+W2", fc="#f8fafc", ec="#94a3b8", size=8)
        _box(ax, 0.58, y, 0.075, 0.052, "Batch 2\nW3", fc="#f8fafc", ec="#94a3b8", size=8)
        _arrow(ax, (0.565, y + 0.026), (0.58, y + 0.026), color="#64748b", ms=8)
    _box(
        ax,
        0.69,
        0.025,
        0.285,
        0.095,
        "Batch hard constraint\nW/O 1~3개, LTH 합 ≤ 55,000\n출력: event log, machine timeline, makespan",
        fc="#f8fafc",
        ec="#94a3b8",
        size=9,
    )


def _legend(ax, x, y):
    _box(ax, x, y + 0.14, 0.2, 0.045, "Node / Edge type", fc="#334155", ec="#334155", size=11, weight="bold", color="white")
    _node(ax, x + 0.025, y + 0.105, "B", "#ffedd5", "#f97316", r=0.014, size=7)
    ax.text(x + 0.05, y + 0.105, "Block node", fontsize=9, va="center", color="#111827")
    _node(ax, x + 0.025, y + 0.075, "W", "#dcfce7", "#16a34a", r=0.014, size=7)
    ax.text(x + 0.05, y + 0.075, "W/O node", fontsize=9, va="center", color="#111827")
    _node(ax, x + 0.025, y + 0.045, "Bay", "#dbeafe", "#2563eb", r=0.014, size=6)
    ax.text(x + 0.05, y + 0.045, "Bay node", fontsize=9, va="center", color="#111827")
    _node(ax, x + 0.025, y + 0.015, "M", "#e0f2fe", "#0284c7", r=0.014, size=7)
    ax.text(x + 0.05, y + 0.015, "Machine node", fontsize=9, va="center", color="#111827")
    _line(ax, (x + 0.115, y + 0.105), (x + 0.155, y + 0.105), color="#f97316", lw=2.0)
    ax.text(x + 0.165, y + 0.105, "Block→Bay candidate", fontsize=8, va="center")
    _line(ax, (x + 0.115, y + 0.075), (x + 0.155, y + 0.075), color="#94a3b8", lw=2.0)
    ax.text(x + 0.165, y + 0.075, "belongs / contains", fontsize=8, va="center")
    _line(ax, (x + 0.115, y + 0.045), (x + 0.155, y + 0.045), color="#2563eb", lw=2.0)
    ax.text(x + 0.165, y + 0.045, "W/O→Machine candidate", fontsize=8, va="center")
    _line(ax, (x + 0.115, y + 0.015), (x + 0.155, y + 0.015), color="#ef4444", lw=2.0, style="--")
    ax.text(x + 0.165, y + 0.015, "masked edge", fontsize=8, va="center")


def _draw_connected_panel(ax, x0, title, selected=False):
    _box(ax, x0, 0.86, 0.41, 0.052, title, fc="#0f172a", ec="#0f172a", size=13, weight="bold", color="white")
    blocks = {"B1": (x0 + 0.055, 0.70), "B2": (x0 + 0.055, 0.54)}
    bays = {"22": (x0 + 0.22, 0.74), "23": (x0 + 0.22, 0.62), "24": (x0 + 0.22, 0.50)}
    wos = {"W1": (x0 + 0.055, 0.32), "W2": (x0 + 0.055, 0.22), "W3": (x0 + 0.055, 0.12)}
    machines = {
        "PLS21": (x0 + 0.34, 0.36, "22"),
        "PLS22": (x0 + 0.34, 0.27, "22"),
        "PLS31": (x0 + 0.34, 0.18, "23"),
        "PLS41": (x0 + 0.34, 0.09, "24"),
    }
    for label, xy in blocks.items():
        _node(ax, *xy, label, "#ffedd5", "#f97316")
    for label, xy in bays.items():
        _node(ax, *xy, f"Bay\n{label}", "#dbeafe", "#2563eb", r=0.026, size=8)
    for label, xy in wos.items():
        _node(ax, *xy, label, "#dcfce7", "#16a34a")
    for label, (x, y, bay) in machines.items():
        _node(ax, x, y, label, "#e0f2fe", "#0284c7", r=0.024, size=7)
        ax.text(x + 0.043, y, f"Bay {bay}", fontsize=8, va="center", color="#475569")

    for block, bxy in blocks.items():
        for bay, bay_xy in bays.items():
            if selected and block == "B1" and bay == "22":
                _arrow(ax, (bxy[0] + 0.022, bxy[1]), (bay_xy[0] - 0.027, bay_xy[1]), color="#dc2626", lw=3.0, ms=12)
            elif selected and block == "B1":
                _arrow(ax, (bxy[0] + 0.022, bxy[1]), (bay_xy[0] - 0.027, bay_xy[1]), color="#f97316", lw=1.0, style="--", alpha=0.35, ms=8)
            else:
                _arrow(ax, (bxy[0] + 0.022, bxy[1]), (bay_xy[0] - 0.027, bay_xy[1]), color="#f97316", lw=1.0, alpha=0.65, ms=8)

    for w in ("W1", "W2"):
        _line(ax, (blocks["B1"][0], blocks["B1"][1] - 0.025), (wos[w][0], wos[w][1] + 0.022), color="#94a3b8", lw=1.6)
    _line(ax, (blocks["B2"][0], blocks["B2"][1] - 0.025), (wos["W3"][0], wos["W3"][1] + 0.022), color="#94a3b8", lw=1.6)
    for bay, bay_xy in bays.items():
        for machine, mxy in machines.items():
            if mxy[2] == bay:
                _line(ax, (bay_xy[0] + 0.027, bay_xy[1]), (mxy[0] - 0.025, mxy[1]), color="#94a3b8", lw=1.5)

    if not selected:
        ax.text(x0 + 0.125, 0.395, "Phase 2 후보 edge는\nPhase 1 배정 전에는 대기", ha="center", va="center", fontsize=9, color="#64748b")
    else:
        for w in ("W1", "W2"):
            for machine in ("PLS21", "PLS22"):
                _arrow(ax, (wos[w][0] + 0.023, wos[w][1]), (machines[machine][0] - 0.026, machines[machine][1]), color="#2563eb", lw=1.4, alpha=0.8, ms=8)
            for machine in ("PLS31", "PLS41"):
                _arrow(ax, (wos[w][0] + 0.023, wos[w][1]), (machines[machine][0] - 0.026, machines[machine][1]), color="#ef4444", lw=1.0, style="--", alpha=0.35, ms=7)
        _box(ax, x0 + 0.11, 0.405, 0.22, 0.055, "B1→Bay22 선택\nW1/W2는 Bay22 설비만 후보", fc="#eef6ff", ec="#60a5fa", size=9)


def draw_connected(output_path: Path) -> None:
    _configure_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.03, 0.955, "PMSP Phase 1→Phase 2 연결 이종 그래프", fontsize=17, weight="bold", color="#0f172a")
    ax.text(
        0.03,
        0.925,
        "Phase 1의 Block→Bay 선택이 Phase 2의 W/O→Machine 후보 edge를 생성/제거한다.",
        fontsize=11,
        color="#475569",
    )
    _draw_connected_panel(ax, 0.03, "Step t: Phase 1 배정 전", selected=False)
    _arrow(ax, (0.465, 0.48), (0.535, 0.48), color="#0f172a", lw=2.8, ms=16)
    ax.text(0.5, 0.52, "select\nB1→Bay22", ha="center", fontsize=10, weight="bold", color="#0f172a")
    _draw_connected_panel(ax, 0.56, "Step t+1: Phase 2 후보 edge 변화", selected=True)
    _box(
        ax,
        0.39,
        0.765,
        0.22,
        0.095,
        "Edge legend\norange: Block→Bay candidate\ngray: belongs/contains\nblue: W/O→Machine candidate\nred dashed: masked edge",
        fc="#f8fafc",
        ec="#cbd5e1",
        size=8,
    )
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[CHECK][draw_phase_graph_environment.draw_connected] output={output_path}")


def draw(output_path: Path) -> None:
    _configure_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    _draw_left_dec_pomdp(ax)
    _draw_phase1_graph(ax)
    _draw_phase2_graph(ax)
    _draw_batch_decoder(ax)
    _line(ax, (0.41, -0.02), (0.41, 0.96), color="#cbd5e1", lw=1.4)
    _line(ax, (0.675, -0.02), (0.675, 0.96), color="#e2e8f0", lw=1.0)
    ax.text(0.43, 0.975, "현재 PMSP 절단 환경의 계층형 이종 그래프 표현", fontsize=15, weight="bold", color="#0f172a")
    ax.text(
        0.43,
        0.945,
        "Block/Bay/W/O/Machine 수가 바뀌어도 node·edge 수만 바뀌고 feature dimension은 유지",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[CHECK][draw_phase_graph_environment.draw] output={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw PMSP hierarchical graph MDP overview PNG")
    parser.add_argument("--output", default="output/phase_graph_environment/pmsp_graph_mdp_overview.png")
    parser.add_argument("--mode", choices=["overview", "connected"], default="overview")
    args = parser.parse_args()
    if args.mode == "connected":
        draw_connected(Path(args.output))
    else:
        draw(Path(args.output))


if __name__ == "__main__":
    main()
