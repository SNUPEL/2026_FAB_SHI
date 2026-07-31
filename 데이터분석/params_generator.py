# -*- coding: utf-8 -*-
"""params_generator.py — block_params.json + wo_params.json + 계열조합(joint)으로 합성 W/O 데이터 생성.

전 계열이 동일 생성식을 공유하고 계수만 다르다(FL 블록 사슬만 역방향).
흐름: [1] block_params 사슬로 블록-계열 생성 → [2] wo_params로 블록 총량에 맞춘 W/O 생성.
계수는 fit_blocks_params.py / fit_wo_params.py 산출물에서만 읽는다(하드코딩·자체fit 없음).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLOCK_PARAMS = REPO_ROOT / "block_params.json"
DEFAULT_WO_PARAMS = REPO_ROOT / "wo_params.json"
DEFAULT_JOINT_PROFILE = REPO_ROOT / "Utils" / "data" / "multi_series_generation_profile.json"

SUPPORTED_SERIES = ("NP", "NC", "FN", "FL")
RHO = 7.85e-6
# TACT_TIME = 0.3037*CUT_LTH + 0.1325*MARK_LTH + 0.4790*THK + 0.3840*PTLST_QTY  (고정식)
TACT_COEF = {"cut": 0.3037, "mark": 0.1325, "thk": 0.4790, "ptlst": 0.3840}
BLOCK_CHAIN_ITEMS = ("LTH", "MARK_LTH", "CUT_LTH", "STL_QTY", "THK", "BTH")
WO_OUTPUT_COLUMNS = (
    "PROJ_NO", "BLK_NO", "WK_ORD_NO", "GYEL", "LTH", "BTH", "THK",
    "CUT_LTH", "MARK_LTH", "BVL_LTH", "STL_QTY", "BV_QTY", "TACT_TIME", "PTLST_QTY",
)


# ════════════════════════════════════════════
# 계수/조합 로딩 (엄격)
# ════════════════════════════════════════════
def load_params(
    block_params_path: str | Path = DEFAULT_BLOCK_PARAMS,
    wo_params_path: str | Path = DEFAULT_WO_PARAMS,
    joint_path: str | Path = DEFAULT_JOINT_PROFILE,
) -> dict:
    """계수/조합을 로딩한다. 경로별로 캐시하므로 에피소드마다 재파싱하지 않는다.

    반환 dict는 캐시되어 공유되므로 호출부는 읽기 전용으로만 사용해야 한다.
    """
    return _load_params_cached(
        str(Path(block_params_path).resolve()),
        str(Path(wo_params_path).resolve()),
        str(Path(joint_path).resolve()),
    )


@lru_cache(maxsize=4)
def _load_params_cached(block_params_path: str, wo_params_path: str, joint_path: str) -> dict:
    block = _load_json(block_params_path, "block_params")
    wo = _load_json(wo_params_path, "wo_params")
    thk_specs = block.get("shared", {}).get("thk_specs")
    if not thk_specs:
        print("[ERROR][params_generator.load_params] cause=missing_thk_specs")
        raise RuntimeError("block_params.json missing shared.thk_specs")
    for series in SUPPORTED_SERIES:
        if series not in block.get("series", {}):
            print(f"[ERROR][params_generator.load_params] cause=missing_block_series series={series}")
            raise RuntimeError(f"block_params.json missing series: {series}")
        if series not in wo.get("series", {}):
            print(f"[ERROR][params_generator.load_params] cause=missing_wo_series series={series}")
            raise RuntimeError(f"wo_params.json missing series: {series}")
    combinations, probabilities = _load_joint(joint_path)
    return {
        "block": block["series"],
        "wo": wo["series"],
        "thk_specs": np.asarray(thk_specs, dtype=float),
        "combinations": combinations,
        "probabilities": probabilities,
    }


def _load_json(path: str | Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        print(f"[ERROR][params_generator._load_json] cause=missing_file label={label} path={path}")
        raise FileNotFoundError(f"missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_joint(joint_path: str | Path) -> tuple[list, np.ndarray]:
    payload = _load_json(joint_path, "joint_profile")
    joint = payload.get("physical_block_joint")
    if not isinstance(joint, Mapping) or "combinations" not in joint or "probabilities" not in joint:
        print(f"[ERROR][params_generator._load_joint] cause=missing_physical_block_joint path={joint_path}")
        raise RuntimeError("joint profile missing physical_block_joint.combinations/probabilities")
    combinations = [[str(s).upper() for s in combo] for combo in joint["combinations"]]
    probabilities = np.asarray(joint["probabilities"], dtype=float)
    if len(combinations) != len(probabilities) or probabilities.sum() <= 0:
        print("[ERROR][params_generator._load_joint] cause=invalid_joint_distribution")
        raise RuntimeError("invalid physical_block_joint distribution")
    for combo in combinations:
        for series in combo:
            if series not in SUPPORTED_SERIES:
                print(f"[ERROR][params_generator._load_joint] cause=unknown_series series={series}")
                raise RuntimeError(f"joint combination has unsupported series: {series}")
    return combinations, probabilities / probabilities.sum()


# ════════════════════════════════════════════
# [1단계] 블록 사슬 생성
# ════════════════════════════════════════════
def _snap_grid(v, grid, lo, hi):
    return np.clip(np.round(v / grid) * grid, lo, hi)


def _snap_specs(v, thk_specs):
    return thk_specs[np.abs(np.asarray(v)[:, None] - thk_specs[None, :]).argmin(axis=1)]


def _gen_block_item(item, d, values, rng, n, thk_specs):
    form = d["form"]
    if form == "normal":
        return _snap_grid(rng.normal(d["mu"], d["sd"], n), d["grid"], d["min"], d["max"])
    x = values[d["x"]]
    if form == "linear":
        pred = d["a"] * x + d["b"]
        noise = d["noise"]
        if noise["kind"] == "mul_gamma":
            v = pred * rng.gamma(noise["shape"], noise["scale"], n)
        else:
            v = pred + rng.normal(0, noise["sd"], n)
        v = np.maximum(v, d.get("floor", 0.0))
        if d.get("zero_rate", 0.0) > 0:
            v = np.where(rng.random(n) < d["zero_rate"], 0.0, v)
        if d.get("integer"):
            v = np.clip(np.round(v), d.get("min", 1), d.get("max", np.inf))
        return v
    if form == "log":
        pred = d["a"] * np.log(np.clip(x, 1e-3, None)) + d["b"]
        return _snap_specs(pred + rng.normal(0, d["noise"]["sd"], n), thk_specs)
    if form == "sat":
        pred = d["Vmax"] * x / (d["K"] + x)
        return _snap_grid(pred + rng.normal(0, d["noise"]["sd"], n), d["grid"], d["min"], d["max"])
    print(f"[ERROR][params_generator._gen_block_item] cause=unknown_form item={item} form={form}")
    raise RuntimeError(f"unknown block item form: {form}")


def generate_blocks(series: str, block_spec: Mapping, n_blocks: int, rng, thk_specs) -> pd.DataFrame:
    """block_params 사슬로 n_blocks개 블록-계열 특성을 생성한다."""
    values: dict[str, np.ndarray] = {}
    for item in block_spec["chain"]:            # 계열별 사슬 순서 (FL=역방향)
        values[item] = _gen_block_item(item, block_spec[item], values, rng, n_blocks, thk_specs)
    return pd.DataFrame({item: values[item] for item in BLOCK_CHAIN_ITEMS})


# ════════════════════════════════════════════
# [2단계] 블록에 매칭하는 W/O 생성
# ════════════════════════════════════════════
def _decay(u, n, a, b, p):
    return 1.0 - (a - b / n) * np.power(u, p)


def _wo_linear(spec, vals, rng, n):
    return spec["const"] + sum(spec[k] * vals[k] for k in spec["x"]) + rng.normal(0, spec["noise"]["sd"], n)


def _sum_match(vals, target, floor):
    """W/O 값들의 합을 블록 목표(target)에 정확히 맞춘다 (블록 = Σ W/O)."""
    vals = np.maximum(vals, floor)
    if target <= 0:
        return np.zeros_like(vals)
    total = vals.sum()
    return np.full_like(vals, target / len(vals)) if total <= 0 else vals * (target / total)


def generate_wo_for_block(series: str, wo_spec: Mapping, block: Mapping, rng) -> dict[str, np.ndarray]:
    n = max(1, int(round(float(block["STL_QTY"]))))
    l_b, b_b = float(block["LTH"]), float(block["BTH"])
    u = np.arange(1, n + 1) / n

    q = wo_spec["LTH"]
    length = _snap_grid(l_b * _decay(u, n, q["a"], q["b"], q["p"]), q["grid"], q["min"], q["max"])
    q = wo_spec["BTH"]
    width = _snap_grid(b_b * _decay(u, n, q["a"], q["b"], q["p"]) * np.exp(rng.normal(0, q["noise"]["sd"], n)),
                       q["grid"], q["min"], q["max"])
    q, e = wo_spec["THK"], wo_spec["THK"]["noise"]
    thk = _snap_grid(q["c0"] + q["c1"] * np.exp(-q["c2"] * length / l_b)
                     + skewnorm.rvs(e["shape"], e["loc"], e["scale"], size=n, random_state=rng),
                     q["grid"], q["min"], q["max"])
    vals = {"LTH": length, "BTH": width, "THK": thk}

    # 자유 생성 후 블록 총량에 합 매칭 (블록 CUT/MARK = Σ W/O)
    vals["CUT_LTH"] = _sum_match(_wo_linear(wo_spec["CUT_LTH"], vals, rng, n),
                                 float(block["CUT_LTH"]), wo_spec["CUT_LTH"]["floor"])
    vals["MARK_LTH"] = _sum_match(_wo_linear(wo_spec["MARK_LTH"], vals, rng, n),
                                  float(block["MARK_LTH"]), wo_spec["MARK_LTH"]["floor"])

    # 베벨 2단계 (발생 로지스틱 + 크기 선형)
    q = wo_spec["BVL_LTH"]
    bevel = np.maximum(_wo_linear(q, vals, rng, n), 0.0)
    if q.get("two_stage"):
        o = q["occurrence"]
        eta = o["const"] + sum(o[k] * vals[k] for k in o["x"])
        occurs = rng.random(n) < 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        bevel = np.where(occurs, np.maximum(bevel, 0.1), 0.0)
    vals["BVL_LTH"] = bevel

    bv = wo_spec.get("BV_QTY")
    if bv is None:
        vals["BV_QTY"] = np.zeros(n)
    else:
        vals["BV_QTY"] = np.where(bevel > 0, np.maximum(np.round(_wo_linear(bv, vals, rng, n)), bv["min"]), 0)
    pt = wo_spec["PTLST_QTY"]
    vals["PTLST_QTY"] = np.maximum(np.round(_wo_linear(pt, vals, rng, n)), pt["min"])
    vals["STL_QTY"] = np.ones(n)
    vals["TACT_TIME"] = (TACT_COEF["cut"] * vals["CUT_LTH"] + TACT_COEF["mark"] * vals["MARK_LTH"]
                         + TACT_COEF["thk"] * thk + TACT_COEF["ptlst"] * vals["PTLST_QTY"])
    return vals


# ════════════════════════════════════════════
# 오케스트레이션: 물리블록 × 계열조합 → wo_df
# ════════════════════════════════════════════
def generate_synthetic_wo_df(
    n_physical_blocks: int,
    seed: int,
    params: Mapping | None = None,
) -> pd.DataFrame:
    """joint 계열조합을 표본해 물리블록마다 여러 계열을 생성하고 wo_df로 합친다."""
    if n_physical_blocks <= 0:
        print(f"[ERROR][params_generator.generate_synthetic_wo_df] cause=invalid_n value={n_physical_blocks}")
        raise RuntimeError("n_physical_blocks must be positive")
    if params is None:
        params = load_params()
    rng = np.random.default_rng(int(seed))
    combos = params["combinations"]
    probs = params["probabilities"]
    thk_specs = params["thk_specs"]

    frames: list[pd.DataFrame] = []
    for physical_index in range(1, n_physical_blocks + 1):
        proj_no, blk_no = f"PROJ_{physical_index}", f"BLK_{physical_index}"
        combo = combos[int(rng.choice(len(combos), p=probs))]      # 계열 조합 표본
        for series in combo:
            block = generate_blocks(series, params["block"][series], 1, rng, thk_specs).iloc[0]
            wo_vals = generate_wo_for_block(series, params["wo"][series], block, rng)
            n = len(wo_vals["LTH"])
            frame = pd.DataFrame(wo_vals)
            frame.insert(0, "PROJ_NO", proj_no)
            frame.insert(1, "BLK_NO", blk_no)
            frame.insert(2, "WK_ORD_NO", [f"{proj_no}_{blk_no}_{series}_{i:03d}" for i in range(1, n + 1)])
            frame.insert(3, "GYEL", series)
            frames.append(frame)

    if not frames:
        print("[ERROR][params_generator.generate_synthetic_wo_df] cause=no_blocks_generated")
        raise RuntimeError("params-driven generation produced no blocks")
    wo_df = pd.concat(frames, ignore_index=True)[list(WO_OUTPUT_COLUMNS)]
    return wo_df


BLOCK_OUTPUT_COLUMNS = (
    "PROJ_NO", "BLK_NO", "GYEL", "LTH", "BTH", "THK", "CUT_LTH", "MARK_LTH",
    "BVL_LTH", "STL_QTY", "WO_QTY", "BV_QTY", "TACT_TIME", "PTLST_QTY",
)


def load_joint_reference(joint_path: str | Path = DEFAULT_JOINT_PROFILE) -> dict:
    """joint의 실적 계열조합 분포(조합·count·확률)를 shipyard 로딩 없이 읽는다.

    감사/리포트용. 전체 profile을 로드하지 않으므로 empirical_series(shipyard)를
    트리거하지 않는다.
    """
    payload = _load_json(joint_path, "joint_profile")
    joint = payload.get("physical_block_joint")
    if not isinstance(joint, Mapping) or not {"combinations", "counts", "probabilities"} <= set(joint):
        print(f"[ERROR][params_generator.load_joint_reference] cause=missing_physical_block_joint path={joint_path}")
        raise RuntimeError("joint profile missing physical_block_joint fields")
    return {
        "combinations": [[str(s).upper() for s in combo] for combo in joint["combinations"]],
        "counts": [int(c) for c in joint["counts"]],
        "probabilities": [float(p) for p in joint["probabilities"]],
    }


def block_df_from_wo(wo_df: pd.DataFrame) -> pd.DataFrame:
    """생성 W/O를 블록-계열로 집계한다. LTH/BTH/THK=max, 나머지 수량/길이=합, WO_QTY=W/O수."""
    grouped = wo_df.groupby(["PROJ_NO", "BLK_NO", "GYEL"], sort=False)
    block = grouped.agg(
        LTH=("LTH", "max"), BTH=("BTH", "max"), THK=("THK", "max"),
        CUT_LTH=("CUT_LTH", "sum"), MARK_LTH=("MARK_LTH", "sum"), BVL_LTH=("BVL_LTH", "sum"),
        STL_QTY=("STL_QTY", "sum"), BV_QTY=("BV_QTY", "sum"),
        TACT_TIME=("TACT_TIME", "max"), PTLST_QTY=("PTLST_QTY", "sum"),
    ).reset_index()
    block["WO_QTY"] = grouped.size().reset_index(name="WO_QTY")["WO_QTY"].to_numpy()
    return block[list(BLOCK_OUTPUT_COLUMNS)]


def series_combinations_from_wo(wo_df: pd.DataFrame) -> list[list[str]]:
    """물리블록별 계열 조합 목록(감사용)."""
    combos = wo_df.groupby(["PROJ_NO", "BLK_NO"], sort=False)["GYEL"].apply(
        lambda s: sorted(set(s))
    )
    return [list(combo) for combo in combos.tolist()]


__all__ = [
    "load_params", "generate_synthetic_wo_df", "block_df_from_wo",
    "series_combinations_from_wo", "load_joint_reference",
    "generate_blocks", "generate_wo_for_block",
]
