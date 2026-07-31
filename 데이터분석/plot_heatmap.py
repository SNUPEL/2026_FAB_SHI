"""기존/생성 두 데이터셋의 수치형 변수 상관관계 히트맵.

WGT, CURVE_LTH 는 상관분석에서 제외한다.
두 데이터셋의 히트맵 축(컬럼) 순서는 Real 기준으로 통일한다.

사용 예:
    python 데이터분석\\plot_heatmap.py ^
        --real "input\\260724_절단블록_데이터_None.xlsx" ^
        --gen  generated_bth.xlsx ^
        --outdir heatmap_bth
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

# 날짜/식별자 성격의 정수 컬럼 + 분석 제외 요청 변수
DROP_COLS = {
    "ACT_ST_DT", "ACT_ED_DT", "RT_CUT_ST_DTM", "RT_CUT_ED_DTM",
    "SASS_ACT_STDT", "ASS_ACT_STDT", "PAN_ST_DT", "CUT_BAY",
    "WGT", "CURVE_LTH", "CURVE_QTY", "BVL_LTH", "BV_QTY", "PTLST_QTY", "TACT_TIME", "STL_QTY"
}


def setup_font():
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def numeric_frame(path):
    df = pd.read_excel(path, sheet_name=0)
    num = df.select_dtypes(include=[np.number])
    drop_upper = {c.upper() for c in DROP_COLS}
    num = num.drop(columns=[c for c in num.columns if c.upper() in drop_upper],
                   errors="ignore")
    # 상수 컬럼 제거 (상관계수 NaN 방지)
    return num.loc[:, num.std(numeric_only=True) > 0]


def unified_order(real_num, gen_num, common_only=False):
    """두 데이터셋의 히트맵 축 순서를 Real 기준으로 통일한다."""
    ref = list(real_num.columns)
    if common_only:
        return [c for c in ref if c in gen_num.columns]
    return ref + [c for c in gen_num.columns if c not in ref]


def draw(ax, corr, title, n, p, annot=True):
    sns.heatmap(
        corr, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, center=0,
        annot=annot, fmt=".2f", annot_kws={"size": 7}, square=True,
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    ax.set_title(f"{title}  (n={n}, p={p})", fontsize=13, pad=12)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def parse_args():
    p = argparse.ArgumentParser(
        description="기존/생성 데이터의 상관관계 히트맵 생성")
    p.add_argument("--real", required=True, help="기존(실측) xlsx 경로")
    p.add_argument("--gen", required=True, help="생성(합성) xlsx 경로")
    p.add_argument("--outdir", default="heatmap_out",
                   help="출력 폴더명 (이 스크립트 폴더 하위에 생성)")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--no-annot", action="store_true", help="셀 숫자 표기 생략")
    p.add_argument("--common-only", action="store_true",
                   help="두 데이터셋 공통 컬럼만 히트맵 축에 사용")
    return p.parse_args()


def main():
    args = parse_args()
    setup_font()
    annot = not args.no_annot

    base = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(base, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    datasets = [
        ("Real", "real", args.real),
        ("Generated", "gen", args.gen),
    ]

    # 1) 두 파일을 먼저 모두 읽는다
    loaded = []
    for title, slug, path in datasets:
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        num = numeric_frame(path)
        loaded.append((title, slug, num))
        print(f"[load] {title:<10} {path}  ->  n={len(num)}, p={num.shape[1]}")

    # 2) 축 순서 통일 (Real 기준, Gen 전용 컬럼은 뒤에 붙임)
    real_num, gen_num = loaded[0][2], loaded[1][2]
    order = unified_order(real_num, gen_num, common_only=args.common_only)

    only_real = [c for c in real_num.columns
                 if c in order and c not in gen_num.columns]
    only_gen = [c for c in gen_num.columns
                if c in order and c not in real_num.columns]
    if only_real or only_gen:
        print(f"[정렬] Real 전용 {only_real} / Gen 전용 {only_gen}"
              f" -> 상대 히트맵에서는 공백 셀로 표시됩니다.")
    print(f"[정렬] 축 순서({len(order)}개): {', '.join(order)}")

    # 3) 통일된 축으로 상관행렬 산출
    frames = []
    for title, slug, num in loaded:
        corr = num.corr(method="pearson").reindex(index=order, columns=order)
        frames.append((title, slug, num, corr))

    # 1x2 통합본
    fig, axes = plt.subplots(1, 2, figsize=(22, 9.5))
    for ax, (title, _, num, corr) in zip(axes.ravel(), frames):
        draw(ax, corr, title, len(num), num.shape[1], annot)
    fig.suptitle("Numeric Feature Correlation Heatmaps", fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "correlation_heatmaps.png"),
                dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # 개별 2장 + 상관행렬 csv
    for title, slug, num, corr in frames:
        f, a = plt.subplots(figsize=(11, 9.5))
        draw(a, corr, title, len(num), num.shape[1], annot)
        f.tight_layout()
        f.savefig(os.path.join(outdir, f"heatmap_{slug}.png"),
                  dpi=args.dpi, bbox_inches="tight")
        plt.close(f)
        corr.to_csv(os.path.join(outdir, f"corr_{slug}.csv"),
                    encoding="utf-8-sig")

    # 실제 vs 생성 상관구조 차이 요약
    r = frames[0][3]
    g = frames[1][3]
    cols = [c for c in order
            if c in real_num.columns and c in gen_num.columns]
    if not cols:
        print("\n[경고] 두 데이터셋에 공통 수치형 컬럼이 없어 차이 비교를 건너뜁니다.")
    else:
        diff = (r.loc[cols, cols] - g.loc[cols, cols]).abs()
        mask = np.triu(np.ones(diff.shape, dtype=bool), k=1)
        vals = diff.where(mask).stack().sort_values(ascending=False)
        print(f"\n[비교 대상 컬럼] {len(cols)}개: {', '.join(cols)}")
        print(f"[DIFF] 상관계수 절대차 평균 {vals.mean():.3f} / 최대 {vals.max():.3f}")
        for (i, j), v in vals.head(10).items():
            print(f"    {i:<12} - {j:<12} |dr| = {v:.3f} "
                  f"(real {r.loc[i, j]:+.2f} -> gen {g.loc[i, j]:+.2f})")

        # 차분 히트맵
        signed = r.loc[cols, cols] - g.loc[cols, cols]
        f, a = plt.subplots(figsize=(11, 9.5))
        sns.heatmap(signed, ax=a, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                    annot=annot, fmt=".2f", annot_kws={"size": 7}, square=True,
                    linewidths=0.4, linecolor="white",
                    cbar_kws={"shrink": 0.75, "label": "r(real) - r(gen)"})
        a.set_title(f"Correlation Difference  (Real - Generated, p={len(cols)})",
                    fontsize=13, pad=12)
        a.tick_params(axis="x", rotation=90, labelsize=8)
        a.tick_params(axis="y", rotation=0, labelsize=8)
        f.tight_layout()
        f.savefig(os.path.join(outdir, "heatmap_diff.png"),
                  dpi=args.dpi, bbox_inches="tight")
        plt.close(f)
        signed.to_csv(os.path.join(outdir, "corr_diff.csv"),
                      encoding="utf-8-sig")

    print(f"\n[done] 저장 위치: {outdir}")


if __name__ == "__main__":
    main()