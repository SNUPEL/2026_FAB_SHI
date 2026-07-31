# -*- coding: utf-8 -*-
"""
fit_blocks_params.py — 블록 산출식 계수 적합 → block_params.json
==============================================================
발표자료(PPT) 블록 산출식의 **형태를 고정**하고 계열(NP/NC/FN/FL)별로
계수만 재추정한다. shipyard_data_generator 가 읽는 JSON 을 만든다.

고정 산출식 (PPT 10쪽)
    길이     L_b   ~ N(mu, sd)                          Clip[min, max], 5mm 격자
    마킹길이 M_b   = (a·L_b + b) × eps                  eps ~ Gamma(k, th), b >= 0
                                                        Zero-Inflation
    절단길이 C_b   = (a·M_b + b) × eps                  eps ~ Gamma(k, th)
    강재수량 n_stl = (a·C_b + b) × eps                  eps ~ Gamma(k, th), 정수, min=1
    두께     H_b   = a·ln(n_stl) + b + eps              eps ~ N(0, sd²), 규격 스냅
    폭       B_b   = Vmax·C_b/(K + C_b) + eps           eps ~ N(0, sd²), 5mm 격자   [신규]

계열 예외 — FL 마킹길이
    FL 은 마킹길이가 블록길이와 사실상 무관하다(Spearman 0.221, R² 0.053).
    반면 절단길이와는 0.487 로 가장 강하다. 따라서 FL 만 마킹·절단 순서를
    뒤집어 절단길이에서 마킹길이를 뽑는다.
        FL   L_b → C_b = (a·L_b + b)×eps → M_b = (a·C_b + b)×eps
    나머지 계열은 PPT 순서 그대로다. 폭·강재수량·두께 경로는 전 계열 동일.

사용법
    python fit_blocks_params.py <블록엑셀> [출력경로]
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

SERIES = ('NP', 'NC', 'FN', 'FL')
LTH_GRID = 5.0
BTH_GRID = 5.0
BLOCK_PARAMS_SCHEMA = 'block_formula_params_v3'

FORWARD_CHAIN = ('LTH', 'MARK_LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH')
REVERSED_CHAIN = ('LTH', 'CUT_LTH', 'MARK_LTH', 'STL_QTY', 'THK', 'BTH')

# 계열별로 바꾸는 것은 '사슬 방향' 하나뿐이다. 나머지 형태·오차형은 PPT 고정.
SERIES_CHAIN = {'NP': 'forward', 'NC': 'forward', 'FN': 'forward', 'FL': 'reversed'}

# 마킹길이는 전 계열 곱셈오차 + 절편 비음수로 통일한다.
# 가산오차(PPT 원안)를 정방향에 그대로 쓰면 짧은 블록에서 예측 평균이 음수가 되어
# 생성 마킹의 12~19%가 하한에 눌린다(실적 영비율은 0~2.2%). PPT 10쪽 자체가
# '예측값이 클수록 오차도 커질 때 곱셈'이라고 적고 있고, 마킹의 조건부 산포는
# 실제로 길이에 비례해 커지므로 곱셈이 형태 규칙에 맞는 선택이다.
MARK_NOISE = 'mul_gamma'


# ════════════════════════════════════════════
# 적합 보조
# ════════════════════════════════════════════
def _r2(y, pred):
    y = np.asarray(y, dtype=float)
    return float(1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _fit_noise(kind, y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    resid = y - pred
    if kind == 'normal':
        return dict(kind='normal', sd=float(resid.std(ddof=1)))
    if kind == 'add_gamma':
        shift = float(-resid.min() + 1e-3)
        shape, _, scale = stats.gamma.fit(resid + shift, floc=0)
        return dict(kind='add_gamma', shape=float(shape), scale=float(scale), shift=shift)
    if kind == 'mul_gamma':
        ratio = y / np.clip(pred, 1e-6, None)
        ratio = ratio[(ratio > 0) & np.isfinite(ratio)]
        shape, _, scale = stats.gamma.fit(ratio, floc=0)
        return dict(kind='mul_gamma', shape=float(shape), scale=float(scale))
    raise ValueError(f'unsupported noise kind: {kind}')


def _fit_linear(noise_kind, predictor, x, y, nonneg=False, **post):
    """y = a·x + b (+/× eps). PPT 의 유일한 평균함수 형태.

    nonneg=True 면 절편을 0 이상으로 제약한다. 곱셈오차는 예측 평균이
    양수여야 정의되므로 마킹길이에만 건다.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if nonneg:
        objective = lambda p: float(np.sum((np.clip(p[0] * x + p[1], 1e-6, None) - y) ** 2))
        result = optimize.minimize(objective, [y.mean() / x.mean(), 0.0],
                                   bounds=[(0.0, None), (0.0, None)])
        a, b = float(result.x[0]), float(result.x[1])
    else:
        a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    spec = dict(form='linear', x=predictor, a=float(a), b=float(b))
    spec['noise'] = _fit_noise(noise_kind, y, pred)
    spec['r2'] = _r2(y, pred)
    spec.update(post)
    return spec


def _fit_log(predictor, x, y, **post):
    """y = a·ln(x) + b + N(0, sd²)."""
    x = np.log(np.clip(np.asarray(x, dtype=float), 1e-3, None))
    y = np.asarray(y, dtype=float)
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    spec = dict(form='log', x=predictor, a=float(a), b=float(b))
    spec['noise'] = _fit_noise('normal', y, pred)
    spec['r2'] = _r2(y, pred)
    spec.update(post)
    return spec


def _fit_saturating(predictor, x, y, **post):
    """y = Vmax·x/(K + x) + N(0, sd²)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    par, _ = optimize.curve_fit(lambda v, vmax, k: vmax * v / (k + v), x, y,
                                p0=[y.max(), np.median(x)], maxfev=40000)
    pred = par[0] * x / (par[1] + x)
    spec = dict(form='sat', x=predictor, Vmax=float(par[0]), K=float(par[1]))
    spec['noise'] = _fit_noise('normal', y, pred)
    spec['r2'] = _r2(y, pred)
    spec.update(post)
    return spec


# ════════════════════════════════════════════
# 계열 하나 적합
# ════════════════════════════════════════════
def fit_series(rows, chain_name):
    columns = {name: rows[name].to_numpy(dtype=float)
               for name in ('LTH', 'MARK_LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH')}
    is_reversed = chain_name == 'reversed'
    chain = REVERSED_CHAIN if is_reversed else FORWARD_CHAIN

    # 마킹길이 0인 블록은 회귀에서 제외한다(영과잉 확률로 따로 재현).
    mark = columns['MARK_LTH']
    positive = mark > 0
    positive_columns = {name: values[positive] for name, values in columns.items()}

    if is_reversed:
        cut_spec = _fit_linear('mul_gamma', 'LTH', columns['LTH'], columns['CUT_LTH'],
                               floor=float(columns['CUT_LTH'].min()))
        mark_spec = _fit_linear(MARK_NOISE, 'CUT_LTH',
                                positive_columns['CUT_LTH'], positive_columns['MARK_LTH'],
                                nonneg=True, zero_rate=float((mark == 0).mean()),
                                floor=float(mark[positive].min()))
    else:
        mark_spec = _fit_linear(MARK_NOISE, 'LTH',
                                positive_columns['LTH'], positive_columns['MARK_LTH'],
                                nonneg=True, zero_rate=float((mark == 0).mean()),
                                floor=float(mark[positive].min()))
        cut_spec = _fit_linear('mul_gamma', 'MARK_LTH', columns['MARK_LTH'], columns['CUT_LTH'],
                               floor=float(columns['CUT_LTH'].min()))

    return {
        'chain': list(chain),
        'n_rows': int(len(rows)),
        # 길이는 전 계열 정규 고정. 형태를 계열마다 바꾸지 않는다.
        'LTH': dict(form='normal',
                    mu=float(columns['LTH'].mean()), sd=float(columns['LTH'].std(ddof=1)),
                    min=float(columns['LTH'].min()), max=float(columns['LTH'].max()),
                    grid=LTH_GRID),
        'MARK_LTH': mark_spec,
        'CUT_LTH': cut_spec,
        'STL_QTY': _fit_linear('mul_gamma', 'CUT_LTH', columns['CUT_LTH'], columns['STL_QTY'],
                               integer=True, min=int(columns['STL_QTY'].min()),
                               max=int(columns['STL_QTY'].max())),
        # 두께 규격 격자(specs)는 계열별로 두지 않는다. 계열 표본이 작으면
        # 관측 종수가 줄어들 뿐 실제 자재 사양이 좁아지는 것이 아니므로,
        # 전 계열 합집합을 shared 로 공유한다.
        'THK': _fit_log('STL_QTY', columns['STL_QTY'], columns['THK']),
        'BTH': _fit_saturating('CUT_LTH', columns['CUT_LTH'], columns['BTH'],
                               min=float(columns['BTH'].min()), max=float(columns['BTH'].max()),
                               grid=BTH_GRID),
    }


def fit_block_params(blocks):
    required = {'GYEL', 'LTH', 'MARK_LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH'}
    missing = sorted(required - set(blocks.columns))
    if missing:
        print(f'[ERROR][fit_block_params] cause=missing_columns columns={missing}')
        raise RuntimeError(f'missing_block_columns: {missing}')

    out = {}
    for series in SERIES:
        rows = blocks[blocks['GYEL'].astype(str).str.upper() == series]
        if len(rows) < 10:
            print(f'[WARN][fit_block_params] series={series} rows={len(rows)} → 건너뜀')
            continue
        out[series] = fit_series(rows, SERIES_CHAIN[series])
        _print_series(series, out[series])
    thk_specs = [float(v) for v in sorted(blocks['THK'].dropna().unique())]
    print(f'\n[공유] 두께 규격 격자 {len(thk_specs)}종 '
          f'({thk_specs[0]:g}~{thk_specs[-1]:g}mm)')
    return {'schema': BLOCK_PARAMS_SCHEMA,
            'shared': {'thk_specs': thk_specs},
            'series': out}


def _print_series(series, spec):
    print(f"[{series}] n={spec['n_rows']}  사슬: {' → '.join(spec['chain'])}")
    for item in ('MARK_LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH'):
        detail = spec[item]
        print(f"    {item:9s} {detail['form']:4s} ← {detail['x']:9s} "
              f"{detail['noise']['kind']:10s} R²={detail['r2']:.3f}")


def main():
    if len(sys.argv) < 2:
        print('사용법: python fit_blocks_params.py <블록엑셀> [출력경로]')
        sys.exit(1)
    source = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else 'block_params.json'
    blocks = pd.read_excel(source)
    print(f'입력: {source} ({len(blocks)}행)\n')
    params = fit_block_params(blocks)
    params['source'] = str(source)
    with open(target, 'w', encoding='utf-8') as handle:
        json.dump(params, handle, ensure_ascii=False, indent=1)
    print(f'\n저장: {target}')


if __name__ == '__main__':
    main()
