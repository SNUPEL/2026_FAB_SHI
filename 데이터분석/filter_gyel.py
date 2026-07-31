import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="계열(GYEL) 기준으로 절단블록 데이터 필터링")
    parser.add_argument("input", help="입력 엑셀 파일 경로")
    parser.add_argument("-g", "--gyel", nargs="+", default=["NP"],
                        help="추출할 계열, 여러 개 가능 (예: -g FN NC). 기본값: NP")
    parser.add_argument("-o", "--output", default=None, help="출력 파일 경로 (미지정 시 자동 생성)")
    args = parser.parse_args()

    in_path = Path(args.input)
    df = pd.read_excel(in_path)

    if "GYEL" not in df.columns:
        raise SystemExit(f"'GYEL' 컬럼이 없습니다. 실제 컬럼: {list(df.columns)}")

    targets = [g.strip() for g in args.gyel]
    filtered = df[df["GYEL"].astype(str).str.strip().isin(targets)]

    if filtered.empty:
        raise SystemExit(f"계열 {targets}에 해당하는 데이터가 없습니다. "
                         f"가능한 값: {sorted(df['GYEL'].dropna().unique())}")

    tag = "_".join(targets)
    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem.replace("None", tag) + in_path.suffix
        if "None" in in_path.stem else f"{in_path.stem}_{tag}{in_path.suffix}"
    )

    filtered.to_excel(out_path, index=False)
    print(f"{len(filtered)} / {len(df)} 행 추출 ({', '.join(targets)}) → {out_path}")


if __name__ == "__main__":
    main()
