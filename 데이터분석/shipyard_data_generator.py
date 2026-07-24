# -*- coding: utf-8 -*-
"""
조선소 블록·W/O 합성 데이터 생성기
====================================================

NP는 `[제출본]가공공장_중간보고.pdf` 고정 산식 생성기를 그대로 사용한다.
FN/FL/NC만 실적 엑셀(WO/블록)에서 계열별 파라미터를 적합한다.

모드 (mode 인자)
  · 'spearman' (기본, 길이 직접) : 로그-로그·멱함수·코퓰러로 순위 구조·분포 보존
        → WO Spearman MAE 0.084 / 블록 0.179
  · 'pearson'  (선형식)          : 선형회귀+정규잔차로 선형 상관(Pearson) 보존
        → WO Pearson MAE 0.114 / 블록 0.123 (Spearman MAE 0.104 / 0.152)
    두 모드는 생성 흐름을 공유하고, 모든 계수와 잔차분포는 선택한 계열에서 적합한다.

공통 생성 구조
  1. 블록 속성   : 길이 ~ N → 마킹 → 절단 → W/O 수 사슬,
                   블록두께 = a·ln(W/O 수) + b + 잔차 → 규격 스냅
  2. WO 길이     : 함수형(병합 패턴 + 멱감쇠), max = 블록길이 보존
  3. WO 두께     : μ(rL)+σ(rL)·SkewNormal, 0.5mm 빈도가중 스냅 (+모드별 ρ 코퓰러)
  4. WO 마킹·절단: WO 길이로 자유 생성 → 입력 실적의 block 집계 계약에 맞게 보정
                   (신규 다계열 MARK=MAX, 과거 260618 NP MARK=SUM, CUT=SUM)
  5. WO 베벨·부재: WO 직접 생성 (베벨길이 ← 두께+길이+마킹, 부재 ← 절단+길이+마킹)
  6. BTH·STL_QTY: BTH는 계열별 로그선형식, STL_QTY는 조건부 범주확률과 계열별 주변분포로 생성
  7. TACT_TIME: 발표자료 Case 6 고정식(CUT/MARK/THK/PTLST)을 전 계열에 공통 적용
     → 블록값: 베벨길이·베벨수량·부재·강재 = Σ WO, BTH·TACT_TIME = MAX WO

계열별 공통 식(식의 형태만 공유하고 값은 독립 적합)
  · 블록길이 ~ N(μ_GYEL, σ_GYEL²)
  · 블록MARK = a_GYEL·길이 + b_GYEL + N(0,σ_MARK,GYEL²)
  · 블록CUT  = c_GYEL·MARK + d_GYEL + N(0,σ_CUT,GYEL²)
  · W/O 수   = round(e_GYEL·CUT + f_GYEL + N(0,σ_WO,GYEL²))
  · 블록두께 = g_GYEL·ln(W/O 수) + h_GYEL + N(0,σ_THK,GYEL²)

`TACT_TIME`은 발표자료의 NP Case 6 고정식을 NP/FN/FL/NC에 적용한다.

선형식(pearson 모드 — 확정식 제외, 실적에서 적합)
  · 베벨길이 = a·두께 + b + N(0,σ)            (허들 유지)
  · 베벨수량 = round(a·베벨길이 + b + N(0,σ))
  · 부재수량 = round(a·절단 + b·길이 + c + N(0,σ))
  · 마킹·절단 = 길이 비례 선형 생성 + 정규노이즈 (입력 block 집계 계약 보존)
  · 두께 = Pearson ρ 직접 부여(선형 조건부)

사용법
  from shipyard_data_generator import ShipyardGenerator
  gen = ShipyardGenerator(wo_xlsx, blk_xlsx, mode='spearman', series='FN')
  gen.fit()
  wo_df, blk_df = gen.generate(n_blocks=800, seed=2026)
  gen.compare(wo_df, blk_df, method='spearman')   # 'pearson'도 가능

  NP 고정식 CLI: python shipyard_data_generator.py --series NP --n 800
  다계열 적합 CLI: python shipyard_data_generator.py --wo_xlsx WO.xlsx --blk_xlsx 블록.xlsx --series FN --mode pearson --n 800
"""
import argparse
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata, norm, skewnorm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.data.report_formula_data_generator import (
    BthFormulaProfile,
    BTH_FORMULA_FEATURES,
    TACT_A_CUT,
    TACT_A_MARK,
    TACT_A_PTLST,
    TACT_A_THK,
    fit_bth_formula,
    generate_report_formula_data,
    sample_bth_formula,
)

CONDITIONAL_FEATURES = ['LTH', 'THK', 'MARK_LTH', 'CUT_LTH', 'BVL_LTH', 'BV_QTY', 'PTLST_QTY']
WCOLS = CONDITIONAL_FEATURES + ['BTH', 'STL_QTY', 'TACT_TIME']
BCOLS = CONDITIONAL_FEATURES + ['BTH', 'STL_QTY', 'WO_QTY', 'TACT_TIME']
COMPARISON_WCOLS = CONDITIONAL_FEATURES + ['BTH', 'STL_QTY']
COMPARISON_BCOLS = CONDITIONAL_FEATURES + ['BTH', 'STL_QTY', 'WO_QTY']

# 발표자료 27쪽 Case 6 계수를 canonical NP 생성기에서 직접 공유한다.
TACT_CUT = TACT_A_CUT
TACT_MARK = TACT_A_MARK
TACT_THK = TACT_A_THK
TACT_PT = TACT_A_PTLST
TACT_FORMULA_SCOPE = 'all_series_shared_np_ppt_case6'
STL_LOCAL_WEIGHT = 0.75
EMPIRICAL_SERIES = ('FN', 'FL', 'NC')
EMPIRICAL_GENERATION_PROFILE_SCHEMA = 'shipyard_empirical_generation_profile_v1'

# FL 계열 MARK_LTH 생성 방법 선택지(다른 계열엔 영향 없음).
#   'chain'     : 블록 사슬을 길이→CUT→MARK로 두고 MARK를 CUT에서 뽑는다.
#                 (구 shipyard_data_generator_fl.py 로직. 블록 MARK~CUT 상관 보존)
#   'dirichlet' : 실적 (블록 MARK 총량, W/O 수) 쌍을 재추출(±jitter)하고 블록 총량을
#                 대칭 Dirichlet(alpha(n))로 W/O에 배분한다. (구 fl_mark_lth.py 로직.
#                 합계 계약이 구조적으로 보장되고 블록 내 분산을 실적에서 역산)
FL_MARK_METHODS = ('chain', 'dirichlet')
DEFAULT_FL_MARK_METHOD = 'chain'
# Dirichlet 총량 재추출 시 총량에 곱하는 승법 로그정규 노이즈 표준편차.
FL_DIRICHLET_TOTAL_JITTER = 0.10


def _profile_json_value(value):
    """numpy 값을 손실 없이 JSON 기본형으로 변환한다."""

    if isinstance(value, np.ndarray):
        return _profile_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _profile_json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, tuple):
        return [_profile_json_value(item) for item in value]
    if isinstance(value, list):
        return [_profile_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _profile_json_value(item) for key, item in value.items()}
    return value


def _require_profile_keys(profile, required, context):
    """고정 profile에 필요한 필드가 없으면 fallback 없이 실패한다."""

    missing = sorted(set(required) - set(profile))
    if missing:
        print(
            "[ERROR][shipyard_data_generator._require_profile_keys] "
            f"cause=missing_profile_keys context={context} keys={missing}"
        )
        raise RuntimeError(f"missing generation profile keys: {context}: {missing}")


def _powf(u, a, b):
    """레벨 멱감쇠 형태: 1 - a·u^b"""
    return 1 - a * np.power(np.clip(u, 0, 1), b)


def _cluster_rel(vs, rel=0.05):
    """내림차순 길이 벡터를 상대 5% 이내로 묶어 그룹 인덱스 리스트 반환"""
    idx = [[0]]
    for i in range(1, len(vs)):
        top = vs[idx[-1][-1]]
        if top > 0 and (top - vs[i]) / top <= rel:
            idx[-1].append(i)
        else:
            idx.append([i])
    return idx


def _adjacent_merge_bin(adjacent_index, wo_count):
    """인접 W/O 위치를 적합·생성에서 공통으로 쓰는 10구간 index로 변환한다."""

    if wo_count < 2 or adjacent_index < 0 or adjacent_index >= wo_count - 1:
        print(
            "[ERROR][shipyard_data_generator._adjacent_merge_bin] "
            f"cause=invalid_adjacent_position index={adjacent_index} wo_count={wo_count}"
        )
        raise RuntimeError("invalid_adjacent_merge_position")
    return min(int(adjacent_index / (wo_count - 1) * 10), 9)


def _select_series_rows(wo, block, series):
    """다계열 입력을 한 계열로 엄격히 분리한다."""

    wo_has_series = 'GYEL' in wo.columns
    block_has_series = 'GYEL' in block.columns
    if wo_has_series != block_has_series:
        print(
            "[ERROR][shipyard_data_generator._select_series_rows] "
            f"cause=inconsistent_gyel_columns wo_has_gyel={wo_has_series} "
            f"block_has_gyel={block_has_series}"
        )
        raise RuntimeError("inconsistent_gyel_columns")
    if not wo_has_series:
        if series is not None:
            print(
                "[ERROR][shipyard_data_generator._select_series_rows] "
                f"cause=series_column_missing requested_series={series}"
            )
            raise RuntimeError("series_column_missing")
        return wo.copy(), block.copy(), None

    wo_series = set(wo['GYEL'].dropna().astype(str).str.strip())
    block_series = set(block['GYEL'].dropna().astype(str).str.strip())
    available = sorted(wo_series | block_series)
    if series is None:
        if len(available) != 1:
            print(
                "[ERROR][shipyard_data_generator._select_series_rows] "
                f"cause=series_required_for_multi_series_input available={available}"
            )
            raise RuntimeError("series_required_for_multi_series_input")
        series = available[0]
    series = str(series).strip()
    if series not in wo_series or series not in block_series:
        print(
            "[ERROR][shipyard_data_generator._select_series_rows] "
            f"cause=requested_series_missing requested={series} "
            f"wo_series={sorted(wo_series)} block_series={sorted(block_series)}"
        )
        raise RuntimeError("requested_series_missing")

    selected_wo = wo[wo['GYEL'].astype(str).str.strip().eq(series)].copy()
    selected_block = block[block['GYEL'].astype(str).str.strip().eq(series)].copy()
    print(
        "[CHECK][shipyard_data_generator._select_series_rows] "
        f"series={series} wo_rows={len(selected_wo)} block_rows={len(selected_block)}"
    )
    return selected_wo, selected_block, series


def _resolve_mark_aggregation(work_orders, blocks, block_keys):
    """입력 block의 MARK_LTH가 W/O 최댓값인지 합계인지 전수검사한다."""

    required = set(block_keys) | {'MARK_LTH'}
    for label, frame in (('wo', work_orders), ('block', blocks)):
        missing = sorted(required - set(frame.columns))
        if missing:
            print(
                "[ERROR][shipyard_data_generator._resolve_mark_aggregation] "
                f"cause=missing_columns table={label} columns={missing}"
            )
            raise RuntimeError(f"missing_mark_aggregation_columns[{label}]: {missing}")
    duplicate_blocks = blocks.duplicated(block_keys, keep=False)
    if duplicate_blocks.any():
        examples = blocks.loc[duplicate_blocks, block_keys].head(5).to_dict('records')
        print(
            "[ERROR][shipyard_data_generator._resolve_mark_aggregation] "
            f"cause=duplicate_block_keys input={examples}"
        )
        raise RuntimeError(f"duplicate_mark_aggregation_block_keys: {examples}")

    aggregate = (
        work_orders.groupby(block_keys, as_index=False, dropna=False)
        .agg(MARK_MAX=('MARK_LTH', 'max'), MARK_SUM=('MARK_LTH', 'sum'))
    )
    compared = blocks[block_keys + ['MARK_LTH']].merge(
        aggregate,
        on=block_keys,
        how='outer',
        validate='one_to_one',
        indicator=True,
    )
    unmatched = compared['_merge'].ne('both')
    if unmatched.any():
        examples = compared.loc[unmatched, block_keys + ['_merge']].head(5).to_dict('records')
        print(
            "[ERROR][shipyard_data_generator._resolve_mark_aggregation] "
            f"cause=block_reference_mismatch input={examples}"
        )
        raise RuntimeError(f"mark_aggregation_block_reference_mismatch: {examples}")
    numeric = compared[['MARK_LTH', 'MARK_MAX', 'MARK_SUM']].apply(pd.to_numeric, errors='coerce')
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        print(
            "[ERROR][shipyard_data_generator._resolve_mark_aggregation] "
            "cause=non_numeric_or_missing_mark_length"
        )
        raise RuntimeError("invalid_mark_aggregation_values")

    actual = numeric['MARK_LTH'].to_numpy(dtype=float)
    max_matches = np.isclose(actual, numeric['MARK_MAX'], rtol=1e-9, atol=1e-6)
    sum_matches = np.isclose(actual, numeric['MARK_SUM'], rtol=1e-9, atol=1e-6)
    max_all = bool(max_matches.all())
    sum_all = bool(sum_matches.all())
    if max_all == sum_all:
        cause = 'ambiguous_contract' if max_all else 'unsupported_contract'
        print(
            "[ERROR][shipyard_data_generator._resolve_mark_aggregation] "
            f"cause={cause} blocks={len(compared)} max_matches={int(max_matches.sum())} "
            f"sum_matches={int(sum_matches.sum())}"
        )
        raise RuntimeError(f"mark_aggregation_{cause}")
    contract = 'max' if max_all else 'sum'
    print(
        "[VALIDATION][shipyard_data_generator._resolve_mark_aggregation] "
        f"passed=true contract={contract} blocks={len(compared)} "
        f"max_matches={int(max_matches.sum())} sum_matches={int(sum_matches.sum())}"
    )
    return contract


def _fit_block_chain_parameters(blocks, chain_order='mark_first'):
    """공통 블록 생성식의 계수와 정규 잔차를 선택 계열에서 적합한다.

    chain_order='mark_first'(기본): 길이→MARK→CUT (MARK=a·길이, CUT=c·MARK).
    chain_order='cut_first'       : 길이→CUT→MARK (CUT=c·길이, MARK=a·CUT).
        FL 'chain' 방법에서 MARK를 CUT에서 뽑기 위해 쓴다. 반환 key는 동일하다.
    """

    if chain_order not in ('mark_first', 'cut_first'):
        print(
            "[ERROR][shipyard_data_generator._fit_block_chain_parameters] "
            f"cause=unsupported_chain_order chain_order={chain_order}"
        )
        raise RuntimeError(f"unsupported_chain_order: {chain_order}")
    required = {'LTH', 'MARK_LTH', 'CUT_LTH', 'WO_QTY'}
    missing = sorted(required - set(blocks.columns))
    if missing:
        print(
            "[ERROR][shipyard_data_generator._fit_block_chain_parameters] "
            f"cause=missing_columns columns={missing}"
        )
        raise RuntimeError(f"missing_block_chain_columns: {missing}")
    values = blocks[list(sorted(required))].apply(pd.to_numeric, errors='coerce')
    invalid_rows = int(values.isna().any(axis=1).sum())
    if invalid_rows:
        print(
            "[ERROR][shipyard_data_generator._fit_block_chain_parameters] "
            f"cause=non_numeric_or_missing rows={invalid_rows}"
        )
        raise RuntimeError("invalid_block_chain_rows")
    if len(values) < 3:
        print(
            "[ERROR][shipyard_data_generator._fit_block_chain_parameters] "
            f"cause=insufficient_blocks rows={len(values)} required=3"
        )
        raise RuntimeError("insufficient_block_chain_rows")
    if (values['LTH'] <= 0).any() or (values['CUT_LTH'] < 0).any() or (values['WO_QTY'] < 1).any():
        print(
            "[ERROR][shipyard_data_generator._fit_block_chain_parameters] "
            "cause=out_of_domain_block_chain_value"
        )
        raise RuntimeError("out_of_domain_block_chain_value")

    def fit_linear(y_name, x_name):
        x = values[x_name].to_numpy(dtype=float)
        y = values[y_name].to_numpy(dtype=float)
        design = np.column_stack([x, np.ones(len(x))])
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        residual_sd = float(np.std(y - (slope * x + intercept), ddof=0))
        return float(slope), float(intercept), residual_sd

    if chain_order == 'cut_first':
        # 길이→CUT→MARK: CUT을 길이에서, MARK를 CUT에서 뽑는다.
        cut_a, cut_b, cut_sd = fit_linear('CUT_LTH', 'LTH')
        mark_a, mark_b, mark_sd = fit_linear('MARK_LTH', 'CUT_LTH')
    else:
        # 길이→MARK→CUT (기본): MARK를 길이에서, CUT을 MARK에서 뽑는다.
        mark_a, mark_b, mark_sd = fit_linear('MARK_LTH', 'LTH')
        cut_a, cut_b, cut_sd = fit_linear('CUT_LTH', 'MARK_LTH')
    count_a, count_b, count_sd = fit_linear('WO_QTY', 'CUT_LTH')
    return {
        'length_mean': float(values['LTH'].mean()),
        'length_sd': float(values['LTH'].std(ddof=0)),
        'length_min': float(values['LTH'].min()),
        'length_max': float(values['LTH'].max()),
        'mark_a': mark_a,
        'mark_b': mark_b,
        'mark_residual_sd': mark_sd,
        'mark_min': float(values['MARK_LTH'].min()),
        'cut_a': cut_a,
        'cut_b': cut_b,
        'cut_residual_sd': cut_sd,
        'cut_min': float(values['CUT_LTH'].min()),
        'wo_count_a': count_a,
        'wo_count_b': count_b,
        'wo_count_residual_sd': count_sd,
        'wo_count_min': int(values['WO_QTY'].min()),
        'wo_count_max': int(values['WO_QTY'].max()),
    }


def _sample_wo_count(expected_count, residual_sd, minimum, maximum, rng):
    """계열별 W/O 수 회귀식에서 정수 개수를 생성하고 실적 지원 범위로 제한한다."""

    sampled = expected_count + rng.normal(0.0, residual_sd)
    return int(np.clip(round(sampled), minimum, maximum))


def _fit_conditional_bth_stl_model(actual_wos):
    """BTH 로그선형식과 STL_QTY 조건부 범주확률을 계열별로 적합한다."""

    required = set(CONDITIONAL_FEATURES) | {'BTH', 'STL_QTY'}
    missing = sorted(required - set(actual_wos.columns))
    if missing:
        print(
            "[ERROR][shipyard_data_generator._fit_conditional_bth_stl_model] "
            f"cause=missing_columns columns={missing}"
        )
        raise RuntimeError(f"missing_conditional_bth_stl_columns: {missing}")

    numeric = actual_wos[list(CONDITIONAL_FEATURES) + ['BTH', 'STL_QTY']].apply(
        pd.to_numeric,
        errors='coerce',
    )
    invalid_rows = int(numeric.isna().any(axis=1).sum())
    if invalid_rows:
        print(
            "[ERROR][shipyard_data_generator._fit_conditional_bth_stl_model] "
            f"cause=non_numeric_or_missing rows={invalid_rows}"
        )
        raise RuntimeError("invalid_conditional_bth_stl_rows")
    if (numeric['BTH'] <= 0).any() or (numeric['STL_QTY'] < 0).any():
        print(
            "[ERROR][shipyard_data_generator._fit_conditional_bth_stl_model] "
            "cause=out_of_domain_target"
        )
        raise RuntimeError("out_of_domain_conditional_bth_stl_target")
    if not np.allclose(numeric['STL_QTY'], np.round(numeric['STL_QTY'])):
        print(
            "[ERROR][shipyard_data_generator._fit_conditional_bth_stl_model] "
            "cause=non_integer_stl_quantity"
        )
        raise RuntimeError("non_integer_stl_quantity")

    series_values = actual_wos['GYEL'].astype(str).str.strip().str.upper().unique()
    if len(series_values) != 1:
        print(
            "[ERROR][shipyard_data_generator._fit_conditional_bth_stl_model] "
            f"cause=non_unique_series values={series_values.tolist()}"
        )
        raise RuntimeError("conditional BTH/STL model requires exactly one series")
    bth_formula = fit_bth_formula(actual_wos, series=str(series_values[0]))
    feature_values = numeric[CONDITIONAL_FEATURES].to_numpy(dtype=float)
    feature_mean = feature_values.mean(axis=0)
    feature_scale = feature_values.std(axis=0, ddof=0)
    active = feature_scale > 0
    if not active.any():
        print(
            "[ERROR][shipyard_data_generator._fit_conditional_bth_stl_model] "
            "cause=all_conditioning_features_constant"
        )
        raise RuntimeError("all_conditioning_features_constant")
    active_features = np.asarray(CONDITIONAL_FEATURES)[active].tolist()
    excluded_features = np.asarray(CONDITIONAL_FEATURES)[~active].tolist()
    if excluded_features:
        print(
            "[CHECK][shipyard_data_generator._fit_conditional_bth_stl_model] "
            f"excluded_constant_features={excluded_features}"
        )
    normalized_features = (feature_values[:, active] - feature_mean[active]) / feature_scale[active]
    stl_quantity = numeric['STL_QTY'].to_numpy(dtype=int)
    stl_classes = np.unique(stl_quantity)
    stl_priors = np.array([(stl_quantity == value).mean() for value in stl_classes])
    return {
        'features': active_features,
        'mean': feature_mean[active],
        'scale': feature_scale[active],
        'tree': cKDTree(normalized_features),
        'bth_formula': bth_formula,
        'stl_quantity': stl_quantity,
        'stl_classes': stl_classes,
        'stl_priors': stl_priors,
    }


def _sample_conditional_bth_stl(generated_features, model, rng, neighbor_count=32):
    """계열별 식의 BTH와 조건부 STL_QTY를 별도로 생성한다."""

    if neighbor_count < 1:
        print(
            "[ERROR][shipyard_data_generator._sample_conditional_bth_stl] "
            f"cause=invalid_neighbor_count value={neighbor_count}"
        )
        raise RuntimeError("invalid_neighbor_count")
    missing = sorted(set(model['features']) - set(generated_features.columns))
    if missing:
        print(
            "[ERROR][shipyard_data_generator._sample_conditional_bth_stl] "
            f"cause=missing_generated_features columns={missing}"
        )
        raise RuntimeError(f"missing_generated_conditioning_features: {missing}")

    target = generated_features[model['features']].apply(pd.to_numeric, errors='coerce')
    if target.isna().any(axis=None):
        print(
            "[ERROR][shipyard_data_generator._sample_conditional_bth_stl] "
            f"cause=non_numeric_or_missing rows={int(target.isna().any(axis=1).sum())}"
        )
        raise RuntimeError("invalid_generated_conditioning_features")
    target_normalized = (target.to_numpy(dtype=float) - model['mean']) / model['scale']
    k = min(int(neighbor_count), len(model['stl_quantity']))
    _, nearest = model['tree'].query(target_normalized, k=k)
    nearest = np.asarray(nearest, dtype=int)
    if nearest.ndim == 1:
        nearest = nearest.reshape(-1, 1)
    local_probabilities = np.column_stack([
        (model['stl_quantity'][nearest] == value).mean(axis=1)
        for value in model['stl_classes']
    ])
    probabilities = (
        STL_LOCAL_WEIGHT * local_probabilities
        + (1.0 - STL_LOCAL_WEIGHT) * model['stl_priors']
    )
    expected_counts = model['stl_priors'] * len(nearest)
    target_counts = np.floor(expected_counts).astype(int)
    remainder = len(nearest) - int(target_counts.sum())
    if remainder:
        fractional_order = np.argsort(-(expected_counts - target_counts), kind='stable')
        target_counts[fractional_order[:remainder]] += 1

    majority_class = int(np.argmax(target_counts))
    class_indices = np.full(len(nearest), majority_class, dtype=int)
    available = np.ones(len(nearest), dtype=bool)
    minority_classes = [
        index for index in np.argsort(target_counts)
        if index != majority_class and target_counts[index] > 0
    ]
    for class_index in minority_classes:
        candidate_rows = np.flatnonzero(available)
        score = (
            np.log(probabilities[candidate_rows, class_index] + 1e-12)
            - np.log(probabilities[candidate_rows, majority_class] + 1e-12)
            + rng.gumbel(size=len(candidate_rows))
        )
        count = int(target_counts[class_index])
        selected_rows = candidate_rows[np.argpartition(score, -count)[-count:]]
        class_indices[selected_rows] = class_index
        available[selected_rows] = False
    stl_quantity = model['stl_classes'][class_indices]
    bth = sample_bth_formula(
        generated_features[list(BTH_FORMULA_FEATURES)],
        model['bth_formula'],
        rng,
    )
    return bth, stl_quantity


def _calculate_tact_time(cut_length, mark_length, thickness, part_quantity, bevel_quantity):
    """발표자료 27쪽 Case 6 고정식으로 W/O 택트타임(분)을 계산한다."""

    values = [
        np.asarray(cut_length, dtype=float),
        np.asarray(mark_length, dtype=float),
        np.asarray(thickness, dtype=float),
        np.asarray(part_quantity, dtype=float),
        np.asarray(bevel_quantity, dtype=float),
    ]
    if any((~np.isfinite(value)).any() or (value < 0).any() for value in values):
        print(
            "[ERROR][shipyard_data_generator._calculate_tact_time] "
            "cause=non_finite_or_negative_input"
        )
        raise RuntimeError("invalid_tact_formula_input")
    return (
        TACT_CUT * values[0]
        + TACT_MARK * values[1]
        + TACT_THK * values[2]
        + TACT_PT * values[3]
    )


def _scale_to_max(values, maximum, lower_bound):
    """값의 순서를 유지하면서 최댓값을 목표 block MARK_LTH에 정확히 맞춘다."""

    values = np.asarray(values, dtype=float).copy()
    maximum = float(maximum)
    lower_bound = float(lower_bound)
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or not np.isfinite(maximum)
        or not np.isfinite(lower_bound)
        or maximum < 0.0
        or lower_bound < 0.0
    ):
        print(
            "[ERROR][shipyard_data_generator._scale_to_max] "
            f"cause=invalid_input size={values.size} maximum={maximum} lower_bound={lower_bound}"
        )
        raise RuntimeError("invalid_scale_to_max_input")
    # 실적에 MARK_LTH=0인 블록이 있으므로 목표 최대값 0은 W/O 전체 0으로 보존한다.
    if maximum == 0.0:
        values.fill(0.0)
        return values
    if maximum < lower_bound:
        print(
            "[ERROR][shipyard_data_generator._scale_to_max] "
            f"cause=maximum_below_positive_floor maximum={maximum} lower_bound={lower_bound}"
        )
        raise RuntimeError("maximum_below_scale_to_max_floor")
    values = np.maximum(values, lower_bound)
    current_maximum = float(values.max())
    maximum_index = int(np.argmax(values))
    if current_maximum <= lower_bound:
        values.fill(lower_bound)
    else:
        values = lower_bound + (
            (values - lower_bound)
            * (maximum - lower_bound)
            / (current_maximum - lower_bound)
        )
    values[maximum_index] = maximum
    return values


def _apply_zero_inflated_floor(value, positive_floor):
    """0과 관측 양수 지원범위를 분리해 블록 마킹길이를 제한한다."""

    value = float(value)
    positive_floor = float(positive_floor)
    if not np.isfinite(value) or not np.isfinite(positive_floor) or positive_floor <= 0.0:
        print(
            "[ERROR][shipyard_data_generator._apply_zero_inflated_floor] "
            f"cause=invalid_input value={value} positive_floor={positive_floor}"
        )
        raise RuntimeError("invalid_zero_inflated_floor_input")
    return 0.0 if value <= 0.0 else max(value, positive_floor)


def _aggregate_generated_block(generated_wos, mark_aggregation):
    """W/O 생성값을 현장 계층 계약에 맞게 블록 특성으로 집계한다."""

    required = set(WCOLS)
    missing = sorted(required - set(generated_wos.columns))
    if missing:
        print(
            "[ERROR][shipyard_data_generator._aggregate_generated_block] "
            f"cause=missing_columns columns={missing}"
        )
        raise RuntimeError(f"missing_generated_wo_columns: {missing}")
    if generated_wos.empty:
        print(
            "[ERROR][shipyard_data_generator._aggregate_generated_block] "
            "cause=empty_generated_wos"
        )
        raise RuntimeError("empty_generated_wos")
    if mark_aggregation not in {'max', 'sum'}:
        print(
            "[ERROR][shipyard_data_generator._aggregate_generated_block] "
            f"cause=unsupported_mark_aggregation input={mark_aggregation}"
        )
        raise RuntimeError(f"unsupported_mark_aggregation: {mark_aggregation}")
    stl_quantity = pd.to_numeric(generated_wos['STL_QTY'], errors='coerce')
    if stl_quantity.isna().any() or not np.allclose(stl_quantity, np.round(stl_quantity)):
        print(
            "[ERROR][shipyard_data_generator._aggregate_generated_block] "
            "cause=invalid_stl_quantity"
        )
        raise RuntimeError("invalid_generated_stl_quantity")
    return {
        'LTH': float(generated_wos['LTH'].max()),
        'THK': float(generated_wos['THK'].max()),
        'MARK_LTH': float(getattr(generated_wos['MARK_LTH'], mark_aggregation)()),
        'CUT_LTH': float(generated_wos['CUT_LTH'].sum()),
        'BVL_LTH': float(generated_wos['BVL_LTH'].sum()),
        'BV_QTY': float(generated_wos['BV_QTY'].sum()),
        'PTLST_QTY': float(generated_wos['PTLST_QTY'].sum()),
        'BTH': float(generated_wos['BTH'].max()),
        'STL_QTY': int(stl_quantity.sum()),
        'WO_QTY': int(len(generated_wos)),
        'TACT_TIME': float(generated_wos['TACT_TIME'].max()),
    }


def _fit_thickness_ratio_profiles(centers, means, standard_deviations):
    """길이비 구간 중 실제 관측치가 있는 구간만 두께 곡선에 적합한다.

    희소 계열에서 빈 구간의 통계량을 임의 값으로 대체하지 않는다. 관측 구간이
    삼 개 미만이면 지수 평균 곡선과 2차 표준편차 곡선을 식별할 수 없으므로 실패시킨다.
    """

    centers = np.asarray(centers, dtype=float)
    means = np.asarray(means, dtype=float)
    standard_deviations = np.asarray(standard_deviations, dtype=float)
    observed = np.isfinite(centers) & np.isfinite(means) & np.isfinite(standard_deviations)
    observed_count = int(observed.sum())
    if observed_count < 3:
        print(
            "[ERROR][shipyard_data_generator._fit_thickness_ratio_profiles] "
            f"cause=insufficient_observed_thickness_ratio_bins observed={observed_count} required=3"
        )
        raise RuntimeError("insufficient_observed_thickness_ratio_bins")

    mean_parameters, _ = optimize.curve_fit(
        lambda x, inf, amp, k: inf + amp * np.exp(-k * x),
        centers[observed],
        means[observed],
        p0=[12, 6, 3],
        maxfev=5000,
    )
    standard_deviation_parameters = np.polyfit(
        centers[observed],
        standard_deviations[observed],
        2,
    )
    return mean_parameters, standard_deviation_parameters, observed_count


class ShipyardGenerator:
    def __init__(self, wo_xlsx, blk_xlsx, mode='spearman', series=None,
                 fl_mark_method=DEFAULT_FL_MARK_METHOD):
        if mode not in ('spearman', 'pearson'):
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.__init__] "
                f"cause=unsupported_mode mode={mode}"
            )
            raise RuntimeError(f"unsupported_mode: {mode}")
        if fl_mark_method not in FL_MARK_METHODS:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.__init__] "
                f"cause=unsupported_fl_mark_method method={fl_mark_method} allowed={FL_MARK_METHODS}"
            )
            raise RuntimeError(f"unsupported_fl_mark_method: {fl_mark_method}")
        self.fl_mark_method = fl_mark_method
        normalized_series = str(series or '').strip().upper()
        if normalized_series == 'NP':
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.__init__] "
                "cause=np_fixed_formula_required use=generate_report_formula_data"
            )
            raise RuntimeError("np_fixed_formula_required")
        if normalized_series not in EMPIRICAL_SERIES:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.__init__] "
                f"cause=empirical_series_required allowed={EMPIRICAL_SERIES} input={series}"
            )
            raise RuntimeError("empirical_series_required")
        self.mode = mode
        raw_wo = pd.read_excel(wo_xlsx)
        raw_block = pd.read_excel(blk_xlsx)
        self.wo, self.blk, self.series = _select_series_rows(
            raw_wo,
            raw_block,
            normalized_series,
        )
        self.block_keys = ['PROJ_NO', 'BLK_NO']
        if 'GYEL' in self.wo.columns:
            self.block_keys.insert(1, 'GYEL')
        # MARK_LTH은 전 계열에서 W/O 최댓값이 아닌 합계로 정의한다(블록 마킹 총량 = Σ W/O).
        self.mark_aggregation = 'sum'
        if self.series == 'FL':
            print(
                "[CHECK][shipyard_data_generator.ShipyardGenerator.__init__] "
                f"series=FL mark_aggregation=sum fl_mark_method={self.fl_mark_method}"
            )
        g = self.wo.groupby(self.block_keys)
        self.wo['n_wo'] = g['LTH'].transform('size')
        self.woF = self.wo[self.wo['n_wo'] >= 2].copy()
        self._fitted = False

    # ================= 적합 =================
    def fit(self):
        woF = self.woF
        g = woF.groupby(self.block_keys)

        # 블록 식의 형태는 모든 계열이 공유하고 계수·잔차는 현재 계열에서만 적합한다.
        block_chain = self.wo.groupby(self.block_keys).agg(
            LTH=('LTH', 'max'),
            MARK_LTH=('MARK_LTH', self.mark_aggregation),
            CUT_LTH=('CUT_LTH', 'sum'),
            WO_QTY=('LTH', 'size'),
        ).reset_index()
        # FL 'chain' 방법은 MARK를 CUT에서 뽑으므로 블록 사슬을 길이→CUT→MARK로 둔다.
        chain_order = 'cut_first' if self.series == 'FL' else 'mark_first'
        self.block_chain = _fit_block_chain_parameters(block_chain, chain_order=chain_order)
        self.conditional_bth_stl = _fit_conditional_bth_stl_model(self.wo)
        # FL은 두 MARK 방법을 모두 쓸 수 있도록 Dirichlet 파라미터(총량 풀 + 농도)를 함께 적합한다.
        if self.series == 'FL':
            self._fit_fl_dirichlet_marking()
        print(
            "[CHECK][shipyard_data_generator.fit] "
            f"series={self.series} block_chain_rows={len(block_chain)} "
            f"wo_count_range={self.block_chain['wo_count_min']}..{self.block_chain['wo_count_max']}"
        )

        # --- WO 길이: 블록별 (a,b) 적합 후 n에 대한 경향 ---
        NB = []
        fit_failures = []
        fit_warnings = []
        for block_id, b in g:
            vs = np.sort(b['LTH'].values)[::-1]
            n = len(vs)
            if n < 3 or len(np.unique(vs)) < 2:
                continue
            idx = _cluster_rel(vs)
            gpos = np.array([np.mean(grp) / (n - 1) for grp in idx])
            gm = np.array([vs[grp].mean() for grp in idx])
            M = vs.max()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', optimize.OptimizeWarning)
                    fab, _ = optimize.curve_fit(
                        lambda u, a, bb: M * _powf(u, a, bb), gpos, gm,
                        p0=[0.7, 1.0], bounds=([0.01, 0.1], [1.5, 3.0]), maxfev=5000)
                NB.append((n, fab[0], fab[1]))
                fit_warnings.extend((block_id, str(item.message)) for item in caught)
            except (RuntimeError, ValueError) as exc:
                fit_failures.append((block_id, str(exc)))
        if fit_failures:
            print(
                "[CHECK][shipyard_data_generator.fit] "
                f"wo_length_curve_excluded={len(fit_failures)} examples={fit_failures[:3]}"
            )
        if fit_warnings:
            print(
                "[CHECK][shipyard_data_generator.fit] "
                f"wo_length_curve_warnings={len(fit_warnings)} examples={fit_warnings[:3]}"
            )
        if len(NB) < 3:
            print(
                "[ERROR][shipyard_data_generator.fit] "
                f"cause=insufficient_wo_length_curve_fits fitted={len(NB)} required=3 series={self.series}"
            )
            raise RuntimeError("insufficient_wo_length_curve_fits")
        NB = np.array(NB)
        nns, aa, bbv = NB[:, 0], NB[:, 1], NB[:, 2]
        self.pa, _ = optimize.curve_fit(lambda n, inf, c: inf - c / n, nns, aa, p0=[0.9, 1])
        self.bconst = float(np.median(bbv))
        zA = (aa - (self.pa[0] - self.pa[1] / nns)) / np.array([self._sigA(n) for n in nns])
        zA = (zA - zA.mean()) / zA.std()
        self.sAp = skewnorm.fit(zA)

        # 병합 확률 패턴 (위치 10구간)
        mp = {i: [] for i in range(10)}
        for _, b in g:
            vs = np.sort(b['LTH'].values)[::-1]
            nn = len(vs)
            for i in range(nn - 1):
                mp[_adjacent_merge_bin(i, nn)].append(
                    (vs[i] - vs[i + 1]) / max(vs[i], 1e-9) <= 0.05)
        reachable_bins = {
            _adjacent_merge_bin(index, wo_count)
            for wo_count in range(
                max(2, int(self.block_chain['wo_count_min'])),
                int(self.block_chain['wo_count_max']) + 1,
            )
            for index in range(wo_count - 1)
        }
        missing_bins = sorted(index for index in reachable_bins if not mp[index])
        if missing_bins:
            print(
                "[ERROR][shipyard_data_generator.fit] "
                f"cause=missing_reachable_merge_bins series={self.series} bins={missing_bins}"
            )
            raise RuntimeError("missing_reachable_merge_bins")
        self.puL = np.array([np.mean(mp[index]) if mp[index] else np.nan for index in range(10)])
        self.floorL = max(int(woF['LTH'].quantile(.005)), 1)

        # --- WO 두께: μ(rL), σ(rL), 잔차 분포 ---
        woF = woF.assign(ML=g['LTH'].transform('max'))
        rLv = (woF['LTH'] / woF['ML']).values
        THKv = woF['THK'].values
        edg = np.linspace(0, 1, 11)
        cen = (edg[:-1] + edg[1:]) / 2
        ratio_bin_values = [
            THKv[(rLv >= edg[i]) & ((rLv < edg[i + 1]) if i < 9 else (rLv <= 1))]
            for i in range(10)
        ]
        mu_pts = np.array([values.mean() if len(values) else np.nan for values in ratio_bin_values])
        sd_pts = np.array([values.std() if len(values) else np.nan for values in ratio_bin_values])
        self.pmu, self.psd, observed_ratio_bins = _fit_thickness_ratio_profiles(cen, mu_pts, sd_pts)
        if observed_ratio_bins < len(cen):
            excluded = np.flatnonzero(~(np.isfinite(mu_pts) & np.isfinite(sd_pts))).tolist()
            print(
                "[CHECK][shipyard_data_generator.fit] "
                f"thickness_ratio_observed_bins={observed_ratio_bins} "
                f"total_bins={len(cen)} excluded_empty_bins={excluded}"
            )
        self.zTp = skewnorm.fit((THKv - self._muf(rLv)) / self._sdf(rLv))
        # 0.5mm 격자 (빈도가중 스냅용)
        vc = pd.Series(THKv).value_counts()
        self.thk_grid = np.array(sorted(vc.index))
        f = np.array([vc[v] for v in self.thk_grid], float)
        self.thk_freq = f / f.sum()

        # --- 조건부 ρ (n, logLr) → 두께-길이 상관 분포 (모드별 척도) ---
        def _blk_rho(b):
            if self.mode == 'pearson':
                return np.corrcoef(b['LTH'], b['THK'])[0, 1]
            return spearmanr(b['LTH'], b['THK']).correlation
        rr = [[_blk_rho(b), len(b), np.log(b['LTH'].max() / max(b['LTH'].min(), 1))]
              for _, b in g if len(b) >= 4 and b.LTH.nunique() > 1 and b.THK.nunique() > 1]
        rdf = pd.DataFrame(rr, columns=['rho', 'n', 'logLr']).dropna()
        Xr = np.column_stack([np.ones(len(rdf)), rdf['n'], rdf['logLr']])
        self.cmu = np.linalg.lstsq(Xr, rdf['rho'].values, rcond=None)[0]
        self.csg = np.linalg.lstsq(Xr, np.abs(rdf['rho'].values - Xr @ self.cmu) * 1.2533, rcond=None)[0]
        zr = (rdf['rho'].values - Xr @ self.cmu) / np.array(
            [max(0.12, self.csg[0] + self.csg[1] * n + self.csg[2] * l) for n, l in zip(rdf['n'], rdf['logLr'])])
        self.zRp = skewnorm.fit(zr)

        # --- WO 마킹·절단: WO 길이로 자유 생성 (배분 대신 후처리 스케일) ---
        Lw = self.wo['LTH'].values.astype(float)
        MKw = self.wo['MARK_LTH'].values.astype(float)
        CTw = self.wo['CUT_LTH'].values.astype(float)
        # block 집계 보정 시 사용할 W/O 양수 지원범위 하한
        self.mk_lo = float(MKw[MKw > 0].min())
        self.ct_lo = float(CTw[CTw > 0].min())
        if self.mode == 'pearson':
            # 선형: 마킹/절단 = a·길이 + b + 정규잔차
            self.mkA, self.mkB = np.polyfit(Lw, MKw, 1); self.mkS = float((MKw - (self.mkA * Lw + self.mkB)).std())
            self.ctA, self.ctB = np.polyfit(Lw, CTw, 1); self.ctS = float((CTw - (self.ctA * Lw + self.ctB)).std())
        else:
            # 로그-로그: 마킹/절단 = exp(a·ln길이 + b)·로그정규
            mm = MKw > 0; bm = np.polyfit(np.log(Lw[mm]), np.log(MKw[mm]), 1)
            self.mkA, self.mkB = bm[0], bm[1]; self.mkS = float((np.log(MKw[mm]) - np.polyval(bm, np.log(Lw[mm]))).std())
            mc = CTw > 0; bc = np.polyfit(np.log(Lw[mc]), np.log(CTw[mc]), 1)
            self.ctA, self.ctB = bc[0], bc[1]; self.ctS = float((np.log(CTw[mc]) - np.polyval(bc, np.log(Lw[mc]))).std())

        # --- WO 베벨·부재 회귀 ---
        L = self.wo['LTH'].values.astype(float)
        T = self.wo['THK'].values.astype(float)
        CUT = self.wo['CUT_LTH'].values.astype(float)
        MK = self.wo['MARK_LTH'].values.astype(float)
        BVL = self.wo['BVL_LTH'].values.astype(float)
        BVQ = self.wo['BV_QTY'].values.astype(float)
        PT = self.wo['PTLST_QTY'].values.astype(float)
        self.bH, self.Tm, self.Ts = self._logit_fit(T, (BVL > 0).astype(float))   # 베벨 발생 허들 ← 두께 (공통)
        m = BVL > 0
        if self.mode == 'spearman':
            # 로그-로그 회귀. 베벨길이 ← 두께+길이+마킹, 베벨수량 ← 베벨길이+절단, 부재 ← 절단+길이+마킹
            #   길이항: 베벨↔길이 음수 방지 / 마킹항: 마킹↔베벨(+)·마킹↔부재(−) / 절단항: 절단↔베벨수량(+)
            self.cB, self.sB = self._fit_ll(BVL[m], T[m], L[m], MK[m])                 # 베벨길이 ← 두께+길이+마킹
            self.cQ, self.sQ = self._fit_ll(np.maximum(BVQ[m], 0.5), BVL[m], CUT[m])   # 베벨수량 ← 베벨길이+절단
            self.cP, self.sP = self._fit_ll(PT, CUT, L, MK)                            # 부재수량 ← 절단+길이+마킹
        else:
            # 선형 회귀 + 정규잔차. 베벨길이 ← 두께+길이+마킹, 베벨수량 ← 베벨길이+절단, 부재 ← 절단+길이+마킹
            self.cBlin, self.sBlin = self._fit_lin(BVL[m], T[m], L[m], MK[m])          # 베벨길이 = a·두께+b·길이+c·마킹+d
            self.cQlin, self.sQlin = self._fit_lin(BVQ[m], BVL[m], CUT[m])             # 베벨수량 = a·베벨길이+b·절단+c
            self.cPlin, self.sPlin = self._fit_lin(PT, CUT, L, MK)                    # 부재수량 = a·절단+b·길이+c·마킹+d

        # 블록 두께 = f(W/O 수 n). W/O 수가 사슬 끝에 있으므로 두께도
        # 길이·마킹·절단과 연결된다. W/O 수를 강재수량과 혼용하지 않는다.
        bagg = self.wo.groupby(self.block_keys).agg(THK=('THK', 'max'), STL=('LTH', 'size'))
        vc = bagg['THK'].value_counts(normalize=True)
        self.blk_thk_vals = np.array(sorted(vc.index))     # 두께 규격 격자 (스냅용)
        # 두께 ~ a·ln(강재) + b + N(0, σ²)
        self.thk_a, self.thk_b = np.polyfit(np.log(bagg['STL']), bagg['THK'], 1)
        self.thk_sig = float((bagg['THK'] - (self.thk_a * np.log(bagg['STL']) + self.thk_b)).std())
        self.n_min = self.block_chain['wo_count_min']
        self.n_max = self.block_chain['wo_count_max']
        self._fitted = True
        return self

    def to_generation_profile(self):
        """Excel 적합 결과 중 생성에 필요한 상태만 JSON 호환 dict로 내보낸다."""

        if not self._fitted:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.to_generation_profile] "
                f"cause=generator_not_fitted series={self.series}"
            )
            raise RuntimeError("generator must be fitted before profile export")
        common_parameters = (
            'pa', 'bconst', 'sAp', 'puL', 'floorL', 'pmu', 'psd', 'zTp',
            'thk_grid', 'thk_freq', 'cmu', 'csg', 'zRp', 'mk_lo', 'ct_lo',
            'mkA', 'mkB', 'mkS', 'ctA', 'ctB', 'ctS', 'bH', 'Tm', 'Ts',
            'blk_thk_vals', 'thk_a', 'thk_b', 'thk_sig', 'n_min', 'n_max',
        )
        mode_parameters = (
            ('cB', 'sB', 'cQ', 'sQ', 'cP', 'sP')
            if self.mode == 'spearman'
            else ('cBlin', 'sBlin', 'cQlin', 'sQlin', 'cPlin', 'sPlin')
        )
        required_attributes = common_parameters + mode_parameters
        if self.series == 'FL':
            # FL은 두 MARK 방법을 모두 재현할 수 있도록 Dirichlet 파라미터도 함께 싣는다.
            required_attributes = required_attributes + (
                'fl_mark_total_pool', 'fl_mark_count_pool', 'fl_dirichlet_alpha_coef',
            )
        missing = [name for name in required_attributes if not hasattr(self, name)]
        if missing:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.to_generation_profile] "
                f"cause=missing_fitted_attributes series={self.series} attributes={missing}"
            )
            raise RuntimeError(f"missing fitted generation attributes: {missing}")

        conditional = self.conditional_bth_stl
        _require_profile_keys(
            conditional,
            ('features', 'mean', 'scale', 'tree', 'bth_formula', 'stl_quantity', 'stl_classes', 'stl_priors'),
            'conditional_bth_stl',
        )
        bth = conditional['bth_formula']
        profile = {
            'schema': EMPIRICAL_GENERATION_PROFILE_SCHEMA,
            'series': self.series,
            'mode': self.mode,
            'mark_aggregation': self.mark_aggregation,
            'fl_mark_method': self.fl_mark_method,
            'block_chain': _profile_json_value(self.block_chain),
            'parameters': {
                name: _profile_json_value(getattr(self, name))
                for name in required_attributes
            },
            'conditional_bth_stl': {
                'features': list(conditional['features']),
                'mean': _profile_json_value(conditional['mean']),
                'scale': _profile_json_value(conditional['scale']),
                'normalized_actual_features': _profile_json_value(conditional['tree'].data),
                'bth_formula': {
                    'series': bth.series,
                    'coefficients': list(bth.coefficients),
                    'residual_std': bth.residual_std,
                    'r_squared': bth.r_squared,
                    'observed_specs': list(bth.observed_specs),
                },
                'stl_quantity': _profile_json_value(conditional['stl_quantity']),
                'stl_classes': _profile_json_value(conditional['stl_classes']),
                'stl_priors': _profile_json_value(conditional['stl_priors']),
            },
        }
        return profile

    @classmethod
    def from_generation_profile(cls, profile):
        """고정 JSON profile을 Excel 없이 생성 가능한 객체로 복원한다."""

        if not isinstance(profile, dict):
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.from_generation_profile] "
                f"cause=invalid_profile_type type={type(profile).__name__}"
            )
            raise RuntimeError("generation profile must be a dictionary")
        _require_profile_keys(
            profile,
            ('schema', 'series', 'mode', 'mark_aggregation', 'block_chain', 'parameters', 'conditional_bth_stl'),
            'root',
        )
        if profile['schema'] != EMPIRICAL_GENERATION_PROFILE_SCHEMA:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.from_generation_profile] "
                f"cause=schema_mismatch actual={profile['schema']} "
                f"expected={EMPIRICAL_GENERATION_PROFILE_SCHEMA}"
            )
            raise RuntimeError("empirical generation profile schema mismatch")
        series = str(profile['series']).strip().upper()
        mode = str(profile['mode']).strip().lower()
        mark_aggregation = str(profile['mark_aggregation']).strip().lower()
        fl_mark_method = str(profile.get('fl_mark_method', DEFAULT_FL_MARK_METHOD)).strip().lower()
        if (
            series not in EMPIRICAL_SERIES
            or mode not in ('spearman', 'pearson')
            or mark_aggregation not in ('max', 'sum')
            or fl_mark_method not in FL_MARK_METHODS
        ):
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.from_generation_profile] "
                f"cause=invalid_contract series={series} mode={mode} mark={mark_aggregation} "
                f"fl_mark_method={fl_mark_method}"
            )
            raise RuntimeError("invalid empirical generation profile contract")

        common_arrays = ('pa', 'puL', 'pmu', 'psd', 'thk_grid', 'thk_freq', 'cmu', 'csg', 'bH', 'blk_thk_vals')
        common_tuples = ('sAp', 'zTp', 'zRp')
        common_floats = (
            'bconst', 'mk_lo', 'ct_lo', 'mkA', 'mkB', 'mkS', 'ctA', 'ctB',
            'ctS', 'Tm', 'Ts', 'thk_a', 'thk_b', 'thk_sig',
        )
        common_ints = ('floorL', 'n_min', 'n_max')
        mode_arrays = ('cB', 'cQ', 'cP') if mode == 'spearman' else ('cBlin', 'cQlin', 'cPlin')
        mode_floats = ('sB', 'sQ', 'sP') if mode == 'spearman' else ('sBlin', 'sQlin', 'sPlin')
        required_parameters = common_arrays + common_tuples + common_floats + common_ints + mode_arrays + mode_floats
        if series == 'FL':
            required_parameters = required_parameters + (
                'fl_mark_total_pool', 'fl_mark_count_pool', 'fl_dirichlet_alpha_coef',
            )
        parameters = profile['parameters']
        if not isinstance(parameters, dict):
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.from_generation_profile] "
                "cause=invalid_parameters_type"
            )
            raise RuntimeError("generation profile parameters must be a dictionary")
        _require_profile_keys(parameters, required_parameters, 'parameters')

        instance = cls.__new__(cls)
        instance.fl_mark_method = fl_mark_method
        instance.series = series
        instance.mode = mode
        instance.mark_aggregation = mark_aggregation
        instance.block_chain = {str(key): value for key, value in profile['block_chain'].items()}
        if series == 'FL':
            instance.fl_mark_total_pool = np.asarray(parameters['fl_mark_total_pool'], dtype=float)
            instance.fl_mark_count_pool = np.asarray(parameters['fl_mark_count_pool'], dtype=int)
            coef = parameters['fl_dirichlet_alpha_coef']
            instance.fl_dirichlet_alpha_coef = (
                None if coef is None else (float(coef[0]), float(coef[1]))
            )
        for name in common_arrays + mode_arrays:
            setattr(instance, name, np.asarray(parameters[name], dtype=float))
        for name in common_tuples:
            setattr(instance, name, tuple(float(value) for value in parameters[name]))
        for name in common_floats + mode_floats:
            setattr(instance, name, float(parameters[name]))
        for name in common_ints:
            setattr(instance, name, int(parameters[name]))

        conditional = profile['conditional_bth_stl']
        _require_profile_keys(
            conditional,
            ('features', 'mean', 'scale', 'normalized_actual_features', 'bth_formula', 'stl_quantity', 'stl_classes', 'stl_priors'),
            'conditional_bth_stl',
        )
        bth = conditional['bth_formula']
        _require_profile_keys(
            bth,
            ('series', 'coefficients', 'residual_std', 'r_squared', 'observed_specs'),
            'conditional_bth_stl.bth_formula',
        )
        normalized_features = np.asarray(conditional['normalized_actual_features'], dtype=float)
        mean = np.asarray(conditional['mean'], dtype=float)
        scale = np.asarray(conditional['scale'], dtype=float)
        stl_quantity = np.asarray(conditional['stl_quantity'], dtype=int)
        stl_classes = np.asarray(conditional['stl_classes'], dtype=int)
        stl_priors = np.asarray(conditional['stl_priors'], dtype=float)
        features = [str(value) for value in conditional['features']]
        if (
            normalized_features.ndim != 2
            or normalized_features.shape != (len(stl_quantity), len(features))
            or mean.shape != (len(features),)
            or scale.shape != (len(features),)
            or (scale <= 0).any()
            or stl_classes.shape != stl_priors.shape
            or not np.isclose(stl_priors.sum(), 1.0)
        ):
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.from_generation_profile] "
                f"cause=invalid_conditional_shape series={series}"
            )
            raise RuntimeError("invalid conditional BTH/STL generation profile")
        bth_profile = BthFormulaProfile(
            series=str(bth['series']).strip().upper(),
            coefficients=tuple(float(value) for value in bth['coefficients']),
            residual_std=float(bth['residual_std']),
            r_squared=float(bth['r_squared']),
            observed_specs=tuple(float(value) for value in bth['observed_specs']),
        )
        if bth_profile.series != series:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.from_generation_profile] "
                f"cause=bth_series_mismatch profile={series} bth={bth_profile.series}"
            )
            raise RuntimeError("BTH profile series mismatch")
        instance.conditional_bth_stl = {
            'features': features,
            'mean': mean,
            'scale': scale,
            'tree': cKDTree(normalized_features),
            'bth_formula': bth_profile,
            'stl_quantity': stl_quantity,
            'stl_classes': stl_classes,
            'stl_priors': stl_priors,
        }
        instance._fitted = True
        return instance

    # ---- 적합 보조 ----
    @staticmethod
    def _sigA(n):
        return max(0.04, 0.476 / np.sqrt(n) - 0.050)

    def _muf(self, x):
        return self.pmu[0] + self.pmu[1] * np.exp(-self.pmu[2] * x)

    def _sdf(self, x):
        return np.polyval(self.psd, x)

    def _fit_beta(self, var, pc):
        """블록 내 점유율 log-log 회귀: log(share_var) ~ β·log(share_pred)"""
        bx, by = [], []
        for _, b in self.woF.groupby(self.block_keys):
            P = b[pc].values.astype(float)
            v = b[var].values.astype(float)
            if v.sum() <= 0 or P.sum() <= 0 or len(b) < 2:
                continue
            sv, sP = v / v.sum(), P / P.sum()
            mm = (sv > 0) & (sP > 0)
            bx.extend(np.log(sP[mm]))
            by.extend(np.log(sv[mm]))
        bx, by = np.array(bx), np.array(by)
        be, c = np.polyfit(bx, by, 1)
        return be, (by - (c + be * bx)).std()

    @staticmethod
    def _fit_ll(y, *xs):
        """로그-로그 다중 선형회귀. 반환: (계수[절편,β...], 잔차표준편차)"""
        m = np.all([x > 0 for x in xs] + [y > 0], axis=0)
        X = np.column_stack([np.ones(m.sum())] + [np.log(x[m]) for x in xs])
        ly = np.log(y[m])
        c = np.linalg.lstsq(X, ly, rcond=None)[0]
        return c, (ly - X @ c).std()

    @staticmethod
    def _fit_lin(y, *xs):
        """선형 다중회귀 (원공간). 반환: (계수[절편,a...], 잔차표준편차)"""
        X = np.column_stack([np.ones(len(y))] + [x for x in xs])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return c, (y - X @ c).std()

    @staticmethod
    def _scale_to_sum(v, total, L, lo, iters=6):
        """길이가중 가법으로 합을 total에 맞추되 하한 lo 보장 (음수·0 방지).
        하한에 걸린 원소는 고정하고 나머지로 부족분을 재분배 (반복)."""
        v = np.maximum(v, lo)
        for _ in range(iters):
            deficit = total - v.sum()
            if abs(deficit) < 1e-9:
                break
            free = (v > lo) if deficit < 0 else np.ones(len(v), bool)  # 줄일 땐 하한 아닌 것만
            w = L * free
            sw = w.sum()
            if sw <= 0:
                break
            v = np.maximum(v + deficit * w / sw, lo)
        return v

    @staticmethod
    def _logit_fit(x, y, it=60):
        """IRLS 로지스틱 회귀 (표준화 x). 반환: (계수, x평균, x표준편차)"""
        xm = (x - x.mean()) / x.std()
        b = np.array([0., 0.])
        for _ in range(it):
            p = 1 / (1 + np.exp(-(b[0] + b[1] * xm)))
            W = p * (1 - p) + 1e-6
            z = b[0] + b[1] * xm + (y - p) / W
            X = np.column_stack([np.ones(len(xm)), xm])
            b = np.linalg.lstsq(X * W[:, None], z * W, rcond=None)[0]
        return b, x.mean(), x.std()

    def _snap_thk(self, t, MT, rng):
        """연속 두께 t를 실제 0.5mm 격자로 빈도가중 스냅 (상한 MT)"""
        sel = (self.thk_grid >= 6) & (self.thk_grid <= MT)
        cand, cf = self.thk_grid[sel], self.thk_freq[sel]
        if len(cand) == 0:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator._snap_thk] "
                f"cause=no_allowed_thickness_spec target={t} maximum={MT}"
            )
            raise RuntimeError("no_allowed_thickness_spec")
        log_weights = np.log(cf) - ((cand - t) / 0.6) ** 2
        weights = np.exp(log_weights - np.max(log_weights))
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator._snap_thk] "
                f"cause=invalid_thickness_weights target={t} maximum={MT}"
            )
            raise RuntimeError("invalid_thickness_weights")
        return float(rng.choice(cand, p=weights / weights.sum()))

    # ================= FL Dirichlet(구 fl_mark_lth.py) 마킹 =================
    def _fit_fl_dirichlet_marking(self):
        """fl_mark_lth.py 'Dirichlet 2단계' 파라미터를 적합한다.

        · 총량 풀 : 실제 블록의 (MARK 총량 Σ, W/O 수) 쌍. 생성 때 쌍을 통째로 재추출한다.
        · 농도 alpha(n) : 블록 내 share의 변동계수에서 CV²=(n-1)/(n·alpha+1)로 블록크기
          n별 alpha를 역산하고, 로그-로그 회귀 alpha(n)=exp(b)·n^a 로 적합한다(빈도 가중).
          관측이 3개 미만이면 상수 alpha=2.5로 폴백한다.
        """

        totals, counts = [], []
        size_shares = {}
        for _, block in self.wo.groupby(self.block_keys):
            mark = block['MARK_LTH'].to_numpy(dtype=float)
            total = float(mark.sum())
            n = int(len(mark))
            if not np.isfinite(total) or total <= 0.0 or n < 1:
                continue
            totals.append(total)
            counts.append(n)
            if n >= 2:
                size_shares.setdefault(n, []).extend((mark / total).tolist())
        if len(totals) < 3:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator._fit_fl_dirichlet_marking] "
                f"cause=insufficient_fl_marking_blocks totals={len(totals)}"
            )
            raise RuntimeError("insufficient_fl_dirichlet_marking_data")
        self.fl_mark_total_pool = np.asarray(totals, dtype=float)
        self.fl_mark_count_pool = np.asarray(counts, dtype=int)

        rows = []  # (n, 표본수, alpha)
        for n, shares in size_shares.items():
            shares = np.asarray(shares, dtype=float)
            if len(shares) < 20:
                continue
            cv = float(shares.std() / shares.mean())
            if cv <= 0.0:
                continue
            alpha = (n - 1) / (n * cv ** 2) - 1.0 / n
            if np.isfinite(alpha) and alpha > 0.0:
                rows.append((float(n), float(len(shares)), float(alpha)))
        if len(rows) >= 3:
            ns = np.array([row[0] for row in rows], dtype=float)
            weights = np.array([row[1] for row in rows], dtype=float)
            alphas = np.array([row[2] for row in rows], dtype=float)
            coef = np.polyfit(np.log(ns), np.log(alphas), 1, w=np.sqrt(weights))
            self.fl_dirichlet_alpha_coef = (float(coef[0]), float(coef[1]))
        else:
            self.fl_dirichlet_alpha_coef = None  # 상수 폴백
        print(
            "[CHECK][shipyard_data_generator.ShipyardGenerator._fit_fl_dirichlet_marking] "
            f"series=FL pool_blocks={len(totals)} alpha_coef={self.fl_dirichlet_alpha_coef} "
            f"alpha(2)={self._fl_alpha_of(2):.2f} alpha(7)={self._fl_alpha_of(7):.2f}"
        )

    def _fl_alpha_of(self, n):
        """블록 크기 n의 대칭 Dirichlet 농도. 계수가 없으면 상수 2.5."""
        coef = getattr(self, 'fl_dirichlet_alpha_coef', None)
        if coef is None:
            return 2.5
        value = np.exp(coef[0] * np.log(max(float(n), 2.0)) + coef[1])
        return float(np.clip(value, 0.3, 50.0))

    def _sample_fl_dirichlet_total(self, rng, n):
        """count≈n인 실제 블록에서 MARK 총량을 뽑아 ±jitter(승법 로그정규)를 준다."""
        counts = self.fl_mark_count_pool
        totals = self.fl_mark_total_pool
        candidates = np.flatnonzero(counts == int(n))
        if candidates.size == 0:
            unique_counts = np.unique(counts)
            nearest = int(unique_counts[np.abs(unique_counts - int(n)).argmin()])
            candidates = np.flatnonzero(counts == nearest)
        index = int(rng.choice(candidates))
        total = float(totals[index]) * float(np.exp(rng.normal(0.0, FL_DIRICHLET_TOTAL_JITTER)))
        return max(total, self.mk_lo)

    def _allocate_fl_dirichlet(self, rng, total, n):
        """블록 MARK 총량을 대칭 Dirichlet(alpha(n))로 W/O에 배분한다(합계 정확 보존)."""
        if n <= 1:
            return np.array([float(total)], dtype=float)
        weights = rng.gamma(self._fl_alpha_of(n), 1.0, n)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights, weight_sum = np.ones(n), float(n)
        return float(total) * weights / weight_sum

    # ================= 생성 =================
    def _gen_one(self, rng, wo_count=None):
        # 1) 블록 속성: 식의 형태는 공통이고 모든 값은 선택 계열의 실적 적합값이다.
        p = self.block_chain
        ML = float(np.clip(rng.normal(p['length_mean'], p['length_sd']), p['length_min'], p['length_max']))
        if self.series == 'FL':
            # FL 공통 사슬: 길이→CUT→W/O 수. 블록 MARK 총량만 방법에 따라 다르게 만든다.
            bCUT = max(p['cut_a'] * ML + p['cut_b'] + rng.normal(0, p['cut_residual_sd']), p['cut_min'])
            n = (
                int(wo_count)
                if wo_count is not None
                else _sample_wo_count(
                    expected_count=p['wo_count_a'] * bCUT + p['wo_count_b'],
                    residual_sd=p['wo_count_residual_sd'],
                    minimum=self.n_min,
                    maximum=self.n_max,
                    rng=rng,
                )
            )
            if self.fl_mark_method == 'dirichlet':
                # fl_mark_lth: 실적 (총량, W/O 수) 쌍을 재추출(±jitter)해 블록 MARK 총량을 뽑는다.
                bMARK = self._sample_fl_dirichlet_total(rng, n)
            else:
                # chain: 블록 MARK를 CUT에서 뽑는다(블록 MARK~CUT 상관 보존).
                bMARK = max(p['mark_a'] * bCUT + p['mark_b'] + rng.normal(0, p['mark_residual_sd']), p['mark_min'])
                bMARK = _apply_zero_inflated_floor(bMARK, self.mk_lo)
        else:
            bMARK = max(p['mark_a'] * ML + p['mark_b'] + rng.normal(0, p['mark_residual_sd']), p['mark_min'])
            bMARK = _apply_zero_inflated_floor(bMARK, self.mk_lo)
            bCUT = max(p['cut_a'] * bMARK + p['cut_b'] + rng.normal(0, p['cut_residual_sd']), p['cut_min'])
            n = (
                int(wo_count)
                if wo_count is not None
                else _sample_wo_count(
                    expected_count=p['wo_count_a'] * bCUT + p['wo_count_b'],
                    residual_sd=p['wo_count_residual_sd'],
                    minimum=self.n_min,
                    maximum=self.n_max,
                    rng=rng,
                )
            )
        # 외부 count는 물리 블록 수식에서 먼저 확정한 값이므로 실적 지원범위로
        # 자르지 않는다. 생성기 자체가 count를 뽑을 때만 적합 지원범위를 강제한다.
        if n <= 0 or (wo_count is None and (n < self.n_min or n > self.n_max)):
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator._gen_one] "
                f"cause=invalid_generated_wo_count series={self.series} value={n} "
                f"support={self.n_min}..{self.n_max} prescribed={wo_count is not None}"
            )
            raise RuntimeError("invalid_generated_wo_count")
        # 블록 두께 = f(W/O 수 n): a·ln(n)+b + 잔차 → 가장 가까운 규격으로 스냅
        thk_cont = self.thk_a * np.log(max(n, 1)) + self.thk_b + rng.normal(0, self.thk_sig)
        MT = float(self.blk_thk_vals[np.argmin(np.abs(self.blk_thk_vals - thk_cont))])

        # 2) WO 길이 (함수형, max=ML)
        if n < 2:
            L = np.array([ML])
        else:
            sizes = [1]
            for i in range(1, n):
                merge_bin = _adjacent_merge_bin(i - 1, n)
                merge_probability = self.puL[merge_bin]
                if not np.isfinite(merge_probability):
                    print(
                        "[ERROR][shipyard_data_generator.ShipyardGenerator._gen_one] "
                        f"cause=unfitted_merge_probability series={self.series} bin={merge_bin} n={n}"
                    )
                    raise RuntimeError("unfitted_merge_probability")
                if rng.random() < merge_probability:
                    sizes[-1] += 1
                else:
                    sizes.append(1)
            a = float(np.clip((self.pa[0] - self.pa[1] / n) + self._sigA(n) * skewnorm.rvs(*self.sAp, random_state=rng), 0.05, 1.4))
            st = np.cumsum([0] + sizes[:-1])
            ce = st + (np.array(sizes) - 1) / 2
            rel = _powf(ce / (n - 1), a, self.bconst)
            rel[0] = 1.0
            vals = []
            for gi, sz in enumerate(sizes):
                vals.extend(np.maximum(ML * rel[gi] * (1 + rng.normal(0, 0.0072, sz)), self.floorL))
            L = np.sort(np.clip(np.round(vals), self.floorL, ML))[::-1]
            L[0] = ML
        n = len(L)

        # 3) WO 두께 (값 + 빈도가중 스냅 + 조건부 ρ 코퓰러)
        r = L / L.max()
        vals = np.array([self._snap_thk(self._muf(r[i]) + self._sdf(r[i]) * skewnorm.rvs(*self.zTp, random_state=rng), MT, rng)
                         for i in range(n)])
        if vals.max() < MT:
            vals[np.argmin(L)] = MT
        logLr = np.log(L.max() / max(L.min(), 1))
        rlo, rhi = (-.95, .7)
        rho = float(np.clip((self.cmu[0] + self.cmu[1] * n + self.cmu[2] * logLr)
                            + max(0.12, self.csg[0] + self.csg[1] * n + self.csg[2] * logLr) * skewnorm.rvs(*self.zRp, random_state=rng),
                            rlo, rhi))
        if self.mode == 'spearman':
            x = norm.ppf((rankdata(L) - 0.5) / n)        # 순위 기반 코퓰러 (Spearman ρ)
        else:
            x = (L - L.mean()) / (L.std() + 1e-9)         # 선형 기반 (Pearson ρ 직접)
        y = rho * x + np.sqrt(max(1 - rho ** 2, 0)) * rng.normal(0, 1, n)
        THK = np.sort(vals)[np.argsort(np.argsort(y))]
        if THK.max() < MT:
            THK[np.argmin(L)] = MT
        else:
            THK[np.argmax(THK)] = MT

        # 4) WO 마킹·절단: WO 길이로 자유 생성한 뒤 입력 실적의 block 집계 계약을 보존한다.
        #    FL 'dirichlet' 방법만 MARK를 Dirichlet 배분으로 대체하고, 그 외(FL 'chain'·타
        #    계열)는 기존 길이-회귀 + 합계 스케일을 쓴다. CUT은 전 계열 공통이다.
        fl_dirichlet = self.series == 'FL' and self.fl_mark_method == 'dirichlet'
        if self.mode == 'pearson':
            if fl_dirichlet:
                MARK = self._allocate_fl_dirichlet(rng, bMARK, n)
            else:
                MARK = self.mkA * L + self.mkB + rng.normal(0, self.mkS, n)
                MARK = (
                    _scale_to_max(MARK, bMARK, self.mk_lo)
                    if self.mark_aggregation == 'max'
                    else self._scale_to_sum(MARK, bMARK, L, self.mk_lo)
                )
            CUT = self.ctA * L + self.ctB + rng.normal(0, self.ctS, n)
            CUT = self._scale_to_sum(CUT, bCUT, L, self.ct_lo)
        else:
            if fl_dirichlet:
                MARK = self._allocate_fl_dirichlet(rng, bMARK, n)
            else:
                MARK = np.exp(self.mkA * np.log(L) + self.mkB + rng.normal(0, self.mkS, n))
                MARK = (
                    _scale_to_max(MARK, bMARK, self.mk_lo)
                    if self.mark_aggregation == 'max'
                    else self._scale_to_sum(MARK * bMARK / MARK.sum(), bMARK, L, self.mk_lo)
                )
            CUT = np.exp(self.ctA * np.log(L) + self.ctB + rng.normal(0, self.ctS, n))
            CUT = self._scale_to_sum(CUT * bCUT / CUT.sum(), bCUT, L, self.ct_lo)

        # 5) WO 베벨·부재 (WO 직접 생성)
        pH = 1 / (1 + np.exp(-(self.bH[0] + self.bH[1] * (THK - self.Tm) / self.Ts)))
        hv = rng.random(n) < pH                            # 베벨 발생 허들 (공통)
        if self.mode == 'spearman':
            # 로그-로그. 베벨길이 ← 두께+길이+마킹, 부재 ← 절단+길이+마킹
            lnMK = np.log(np.maximum(MARK, 1e-6))
            BVL = np.where(hv, np.exp(self.cB[0] + self.cB[1] * np.log(THK) + self.cB[2] * np.log(L) + self.cB[3] * lnMK + rng.normal(0, self.sB, n)), 0.0)
            BVQ = np.where(BVL > 0, np.maximum(np.round(np.exp(self.cQ[0] + self.cQ[1] * np.log(np.maximum(BVL, 0.1)) + self.cQ[2] * np.log(CUT) + rng.normal(0, self.sQ, n))), 1), 0.0)
            PT = np.maximum(np.round(np.exp(self.cP[0] + self.cP[1] * np.log(CUT) + self.cP[2] * np.log(L) + self.cP[3] * lnMK + rng.normal(0, self.sP, n))), 1)
        else:
            # 선형. 베벨길이 ← 두께+길이+마킹, 부재 ← 절단+길이+마킹
            BVL = np.where(hv, np.maximum(self.cBlin[0] + self.cBlin[1] * THK + self.cBlin[2] * L + self.cBlin[3] * MARK + rng.normal(0, self.sBlin, n), 0.1), 0.0)
            BVQ = np.where(BVL > 0, np.maximum(np.round(self.cQlin[0] + self.cQlin[1] * BVL + self.cQlin[2] * CUT + rng.normal(0, self.sQlin, n)), 1), 0.0)
            PT = np.maximum(np.round(self.cPlin[0] + self.cPlin[1] * CUT + self.cPlin[2] * L + self.cPlin[3] * MARK + rng.normal(0, self.sPlin, n)), 1)
        wodf = pd.DataFrame({'LTH': L, 'THK': THK, 'MARK_LTH': MARK, 'CUT_LTH': CUT,
                             'BVL_LTH': BVL, 'BV_QTY': BVQ, 'PTLST_QTY': PT})
        wodf['TACT_TIME'] = _calculate_tact_time(CUT, MARK, THK, PT, BVQ)
        return wodf

    def generate(self, n_blocks=800, seed=2026, wo_counts=None, block_seeds=None):
        """합성 블록 n_blocks개 생성. 반환: (wo_df, blk_df). wo_df엔 BLK_ID 부여."""
        if not self._fitted:
            self.fit()
        resolved_wo_counts = _validate_optional_positive_integer_sequence(
            wo_counts,
            int(n_blocks),
            'wo_counts',
        )
        resolved_block_seeds = _validate_optional_positive_integer_sequence(
            block_seeds,
            int(n_blocks),
            'block_seeds',
        )
        rng = np.random.default_rng(seed)
        wos = []
        for bid in range(n_blocks):
            block_rng = (
                np.random.default_rng(resolved_block_seeds[bid])
                if resolved_block_seeds is not None
                else rng
            )
            requested_count = (
                resolved_wo_counts[bid]
                if resolved_wo_counts is not None
                else None
            )
            w = self._gen_one(block_rng, wo_count=requested_count)
            w = w.copy()
            w.insert(0, 'BLK_ID', bid)
            if self.series is not None:
                w.insert(1, 'GYEL', self.series)
            wos.append(w)
        wo_df = pd.concat(wos, ignore_index=True)
        BTH, STL = _sample_conditional_bth_stl(
            wo_df,
            self.conditional_bth_stl,
            rng,
        )
        wo_df['BTH'] = BTH
        wo_df['STL_QTY'] = STL

        bls = []
        for bid, block_wos in wo_df.groupby('BLK_ID', sort=True):
            block = _aggregate_generated_block(block_wos, self.mark_aggregation)
            block['BLK_ID'] = bid
            if self.series is not None:
                block['GYEL'] = self.series
            bls.append(block)
        identity_columns = ['BLK_ID'] + (['GYEL'] if self.series is not None else [])
        wo_df = wo_df[identity_columns + WCOLS]
        blk_df = pd.DataFrame(bls)[identity_columns + BCOLS]
        return wo_df, blk_df

    # ================= 검증 =================
    def _actuals(self):
        act_wo = self.wo[COMPARISON_WCOLS]
        act_blk = self.wo.groupby(self.block_keys).agg(
            LTH=('LTH', 'max'), THK=('THK', 'max'), MARK_LTH=('MARK_LTH', self.mark_aggregation),
            CUT_LTH=('CUT_LTH', 'sum'), BVL_LTH=('BVL_LTH', 'sum'), BV_QTY=('BV_QTY', 'sum'),
            PTLST_QTY=('PTLST_QTY', 'sum'), BTH=('BTH', 'max'), STL_QTY=('STL_QTY', 'sum'),
            WO_QTY=('LTH', 'size')
        ).reset_index()
        return act_wo, act_blk

    def compare(self, wo_df, blk_df, method='spearman'):
        """생성 vs 실적 상관행렬 MAE(상삼각) 출력 및 반환."""
        act_wo, act_blk = self._actuals()
        cwa = act_wo[COMPARISON_WCOLS].corr(method)
        cwg = wo_df[COMPARISON_WCOLS].corr(method)
        cba = act_blk[COMPARISON_BCOLS].corr(method)
        cbg = blk_df[COMPARISON_BCOLS].corr(method)
        wmae = np.abs((cwg.values - cwa.values)[np.triu_indices(len(COMPARISON_WCOLS), 1)]).mean()
        bmae = np.abs((cbg.values - cba.values)[np.triu_indices(len(COMPARISON_BCOLS), 1)]).mean()
        print(f"[{method}] WO 상관행렬 MAE={wmae:.3f} / 블록 MAE={bmae:.3f}")
        return {'wo_mae': wmae, 'blk_mae': bmae,
                'wo_actual': cwa, 'wo_gen': cwg, 'blk_actual': cba, 'blk_gen': cbg}


def _validate_optional_positive_integer_sequence(values, expected_length, field_name):
    if values is None:
        return None
    raw_values = tuple(values)
    resolved = []
    invalid_value = None
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)):
            invalid_value = value
            break
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid_value = value
            break
        if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            invalid_value = value
            break
        resolved.append(int(numeric))
    if len(raw_values) != expected_length or invalid_value is not None:
        print(
            "[ERROR][shipyard_data_generator._validate_optional_positive_integer_sequence] "
            f"cause=invalid_sequence field={field_name} expected={expected_length} "
            f"actual={len(raw_values)} invalid_value={invalid_value} values={raw_values[:10]}"
        )
        raise RuntimeError(f"invalid_{field_name}")
    return tuple(resolved)


def main():
    ap = argparse.ArgumentParser(description="조선소 블록·W/O 합성 데이터 생성")
    ap.add_argument('--wo_xlsx', help='실적 WO 엑셀 경로')
    ap.add_argument('--blk_xlsx', help='실적 블록 엑셀 경로')
    ap.add_argument('--mode', choices=['spearman', 'pearson'], default='spearman',
                    help="생성 모드: spearman(길이직접,기본) / pearson(선형식)")
    ap.add_argument('--series', help='적합할 계열: NP, FN, FL, NC. 다계열 입력에서는 필수')
    ap.add_argument('--fl-mark-method', dest='fl_mark_method',
                    choices=list(FL_MARK_METHODS), default=DEFAULT_FL_MARK_METHOD,
                    help="FL MARK_LTH 생성 방법: chain(CUT→MARK) / dirichlet(fl_mark_lth 2단계). FL에만 적용")
    ap.add_argument('--n', type=int, default=800, help='생성 블록 수 (기본 800)')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', default='generated', help='출력 파일 접두사')
    args = ap.parse_args()

    normalized_series = str(args.series or '').strip().upper()
    print(
        "[CHECK][shipyard_data_generator.main] "
        f"wo={args.wo_xlsx} block={args.blk_xlsx} series={normalized_series} "
        f"mode={args.mode} blocks={args.n} output={args.out}"
    )
    if normalized_series == 'NP':
        generated = generate_report_formula_data(n_blocks=args.n, seed=args.seed, gyel='NP')
        wo_df = generated.wo_df
        blk_df = generated.block_df
        wo_df.to_csv(f'{args.out}_wo.csv', index=False, encoding='utf-8-sig')
        blk_df.to_csv(f'{args.out}_blk.csv', index=False, encoding='utf-8-sig')
        with open(f'{args.out}_metadata.json', 'w', encoding='utf-8') as file:
            json.dump(
                {
                    'series': 'NP',
                    'mode': 'np_ppt_fixed_formula',
                    'generated_blocks': len(blk_df),
                    'generated_wos': len(wo_df),
                    'tact_formula_scope': TACT_FORMULA_SCOPE,
                    'tact_formula_coefficients': {
                        'CUT_LTH': TACT_CUT,
                        'MARK_LTH': TACT_MARK,
                        'THK': TACT_THK,
                        'PTLST_QTY': TACT_PT,
                    },
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        print(
            "[VALIDATION][shipyard_data_generator.main] passed=true "
            f"series=NP generated_wos={len(wo_df)} generated_blocks={len(blk_df)} "
            f"source=np_ppt_fixed_formula tact_formula_scope={TACT_FORMULA_SCOPE}"
        )
        return
    if not args.wo_xlsx or not args.blk_xlsx:
        print(
            "[ERROR][shipyard_data_generator.main] "
            f"cause=missing_empirical_source series={normalized_series}"
        )
        raise RuntimeError("FN/FL/NC generation requires --wo_xlsx and --blk_xlsx")
    gen = ShipyardGenerator(
        args.wo_xlsx, args.blk_xlsx, mode=args.mode, series=args.series,
        fl_mark_method=args.fl_mark_method,
    ).fit()
    wo_df, blk_df = gen.generate(n_blocks=args.n, seed=args.seed)
    wo_df.to_csv(f'{args.out}_wo.csv', index=False, encoding='utf-8-sig')
    blk_df.to_csv(f'{args.out}_blk.csv', index=False, encoding='utf-8-sig')
    with open(f'{args.out}_metadata.json', 'w', encoding='utf-8') as file:
        json.dump(
            {
                'series': gen.series,
                'mode': gen.mode,
                'generated_blocks': len(blk_df),
                'generated_wos': len(wo_df),
                'bth_source': 'series_log_linear_formula_observed_spec_rounding',
                'stl_quantity_source': (
                    f'series_conditional_class_probability_local_weight_{STL_LOCAL_WEIGHT}_exact_marginal'
                ),
                'tact_formula_scope': TACT_FORMULA_SCOPE,
                'mark_aggregation': gen.mark_aggregation,
                'fl_mark_method': gen.fl_mark_method,
                'tact_formula_coefficients': {
                    'CUT_LTH': TACT_CUT,
                    'MARK_LTH': TACT_MARK,
                    'THK': TACT_THK,
                    'PTLST_QTY': TACT_PT,
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(
        "[VALIDATION][shipyard_data_generator.main] passed=true "
        f"series={gen.series} generated_wos={len(wo_df)} generated_blocks={len(blk_df)} "
        "tact_time_generated=true bth_generated=true stl_qty_generated=true "
        f"tact_formula_scope={TACT_FORMULA_SCOPE}"
    )
    gen.compare(wo_df, blk_df, 'spearman')
    gen.compare(wo_df, blk_df, 'pearson')


if __name__ == '__main__':
    main()
