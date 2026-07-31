# -*- coding: utf-8 -*-
"""
fit_wo_params.py — W/O 산출식 계수 적합 → wo_params.json
========================================================
발표자료의 산출식 형태를 그대로 유지하고, 계열(NP/NC/FN/FL)별로 계수만 재추정한다.
블록 앵커(L_b, B_b)는 절단블록 데이터의 실측값을 쓴다.

산출식 (u_i = i / n_stl,b,  i 는 W/O 를 길이 내림차순 정렬한 순번)

  길이      L_bi     = L_b * {1 - (a - b/n) * u_i^p}                     오차 없음
  폭  [신규] B_bi     = B_b * {1 - (a - b/n) * u_i^p} * exp(eps)          eps ~ N(0, s^2)
  두께      H_bi     = c0 + c1 * exp(-c2 * L_bi / L_b) + eps             eps ~ SkewNormal
  절단 길이  L_cut,bi = k1 * L_bi + k0 + eps                              eps ~ N
  마킹 길이  L_mark,bi= k1 * L_bi + k0 + eps                              eps ~ N
  베벨 길이  L_bvl,bi = k0 + k1*H_bi + k2*L_bi + k3*L_mark,bi + eps       eps ~ N, 음수는 0
  베벨 수량  n_bvl,bi = k1*L_bvl,bi + k2*L_cut,bi + k0 + eps              eps ~ N, L_bvl>0 일 때만
  부재 수량  n_ptlst  = k0 + k1*L_cut + k2*L_bi + k3*L_mark + eps         eps ~ N
  무게      WGT      = round(7.85e-6 * L_bi * B_bi * H_bi)               항등식

베벨 발생 조건 (BEVEL_TWO_STAGE=True)
    실적 베벨길이는 계열별로 55~77%가 0인데, 선형식 결과를 0에서 자르는 것만으로는
    그 비율이 재현되지 않는다. 발표자료의 "베벨 길이가 0보다 크게 계산된 경우"
    조건을 확률식으로 명시해 발생 여부를 먼저 정하고, 발생분에만 위 선형식을 적용한다.
        P(L_bvl > 0) = sigmoid(g0 + g1*H_bi + g2*L_bi + g3*L_mark,bi)
    산출식 형태 자체는 그대로이며 조건부 확률 한 줄이 추가될 뿐이다.
    False 로 두면 발표자료 원본대로 단일 선형식 + 0 절단으로 적합한다.

사용법
    python fit_wo_params.py <W/O엑셀> <블록엑셀> [출력경로] [--no-validate]
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import skewnorm, ks_2samp

SERIES = ('NP', 'NC', 'FN', 'FL')
KEY = ['PROJ_NO', 'BLK_NO', 'GYEL']          # 블록 키는 계열까지 포함해야 한다
RHO = 7.85e-6
LTH_GRID = 5.0
BTH_GRID = 5.0
THK_GRID = 0.5
BEVEL_TWO_STAGE = True
WO_PARAMS_SCHEMA = 'wo_formula_params_v3'

GEN_COLUMNS = ('LTH', 'BTH', 'THK', 'MARK_LTH', 'CUT_LTH',
               'BVL_LTH', 'BV_QTY', 'PTLST_QTY')


# ════════════════════════════════════════════
# 입력
# ════════════════════════════════════════════
def load(wo_path, block_path):
    wo = pd.read_excel(wo_path)
    blocks = pd.read_excel(block_path)

    missing = sorted({'PROJ_NO', 'BLK_NO', 'GYEL', 'LTH', 'BTH', 'STL_QTY'} - set(blocks.columns))
    if missing:
        raise RuntimeError(f'missing_block_columns: {missing}')

    anchors = (blocks.set_index(KEY)[['LTH', 'BTH']]
               .rename(columns={'LTH': 'Lb', 'BTH': 'Bb'}))
    if anchors.index.has_duplicates:
        raise RuntimeError('block_key_not_unique: PROJ_NO+BLK_NO+GYEL 중복')

    n_before = len(wo)
    wo = wo.join(anchors, on=KEY)
    if len(wo) != n_before:
        raise RuntimeError(f'join_changed_row_count: {n_before} → {len(wo)}')
    if wo['Lb'].isna().any():
        raise RuntimeError(f'unmatched_wo_rows: {int(wo["Lb"].isna().sum())}건')

    # 길이 내림차순 순번 i, 블록 강재 수량 n, 정규화 인덱스 u
    wo = wo.sort_values(KEY + ['LTH'], ascending=[True, True, True, False]).copy()
    wo['i'] = wo.groupby(KEY).cumcount() + 1
    wo['n'] = wo.groupby(KEY)['LTH'].transform('size')
    wo['u'] = wo['i'] / wo['n']
    print(f'입력: W/O {len(wo)}행 / 블록 {len(blocks)}행, 조인 정상')
    return wo, blocks


# ════════════════════════════════════════════
# 적합 보조
# ════════════════════════════════════════════
def _r2(y, pred):
    y = np.asarray(y, dtype=float)
    return float(1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def _decay(u, n, a, b, p):
    """발표자료 감쇠항 1 - (a - b/n) * u^p"""
    return 1.0 - (a - b / n) * np.power(u, p)


def _fit_decay(target_ratio, u, n, transform=None):
    best = None
    for p0 in (0.5, 0.89, 1.5, 2.5):
        for a0 in (0.3, 0.6, 0.9):
            def residual(th, _t=transform):
                model = _decay(u, n, *th)
                if _t == 'log':
                    return target_ratio - np.log(np.clip(model, 1e-3, None))
                return target_ratio - model
            result = least_squares(residual, [a0, 0.5, p0],
                                   bounds=([0.0, -2.0, 0.05], [0.999, 3.0, 6.0]))
            if best is None or result.cost < best.cost:
                best = result
    return best.x


def ols(columns, predictors, y):
    """상수항 포함 최소제곱 → {x, const, 계수…, noise, r2}"""
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(y))] + [columns[name] for name in predictors])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    dof = max(len(y) - design.shape[1], 1)
    spec = {'x': list(predictors), 'const': float(beta[0])}
    spec.update({name: float(value) for name, value in zip(predictors, beta[1:])})
    spec['noise'] = dict(kind='normal', sd=float(np.sqrt((residual ** 2).sum() / dof)))
    spec['r2'] = _r2(y, design @ beta)
    return spec


def logistic(columns, predictors, y, iterations=200):
    """IRLS 로지스틱 회귀 → {x, const, 계수…, accuracy}"""
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(y))] + [columns[name] for name in predictors])
    scale = np.maximum(np.abs(design).max(axis=0), 1e-9)
    scaled = design / scale
    beta = np.zeros(scaled.shape[1])
    for _ in range(iterations):
        mu = 1.0 / (1.0 + np.exp(-np.clip(scaled @ beta, -30, 30)))
        weights = np.maximum(mu * (1.0 - mu), 1e-6)
        working = scaled @ beta + (y - mu) / weights
        updated = np.linalg.solve((scaled.T * weights) @ scaled + 1e-8 * np.eye(len(beta)),
                                  (scaled.T * weights) @ working)
        if np.max(np.abs(updated - beta)) < 1e-9:
            beta = updated
            break
        beta = updated
    beta = beta / scale
    mu = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
    spec = {'x': list(predictors), 'const': float(beta[0])}
    spec.update({name: float(value) for name, value in zip(predictors, beta[1:])})
    spec['accuracy'] = float(((mu > 0.5) == (y > 0.5)).mean())
    return spec


# ════════════════════════════════════════════
# 항목별 적합
# ════════════════════════════════════════════
def fit_length(rows):
    """L_bi = L_b * {1 - (a - b/n) u^p}"""
    ratio = (rows['LTH'] / rows['Lb']).to_numpy(dtype=float)
    u, n = rows['u'].to_numpy(dtype=float), rows['n'].to_numpy(dtype=float)
    a, b, p = _fit_decay(ratio, u, n)
    pred = rows['Lb'].to_numpy(dtype=float) * _decay(u, n, a, b, p)
    return dict(form='decay', a=float(a), b=float(b), p=float(p), grid=LTH_GRID,
                min=float(rows['LTH'].min()), max=float(rows['LTH'].max()),
                r2=_r2(rows['LTH'], pred))


def fit_width(rows, rng):
    """B_bi = B_b * {1 - (a - b/n) u^p} * exp(eps)

    감쇠항은 로그공간 최소제곱, sigma 는 생성 표본의 주변분포 표준편차가
    실적과 맞도록 모멘트 보정한다(로그 잔차 sd 를 그대로 쓰면 반올림·절단
    때문에 과산포가 된다).
    """
    width = rows['BTH'].to_numpy(dtype=float)
    anchor = rows['Bb'].to_numpy(dtype=float)
    u, n = rows['u'].to_numpy(dtype=float), rows['n'].to_numpy(dtype=float)
    low, high = float(width.min()), float(width.max())

    a, b, p = _fit_decay(np.log(np.clip(width / anchor, 1e-6, None)), u, n, transform='log')
    decay = _decay(u, n, a, b, p)
    target = float(width.std())

    def simulated_sd(sigma):
        return float(np.mean([
            np.clip(np.round(anchor * decay * np.exp(rng.normal(0, sigma, len(u)))
                             / BTH_GRID) * BTH_GRID, low, high).std()
            for _ in range(8)]))

    lo, hi = 0.0, 1.5
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if simulated_sd(mid) < target else (lo, mid)

    return dict(form='decay_mul', a=float(a), b=float(b), p=float(p),
                noise=dict(kind='lognormal', sd=float(0.5 * (lo + hi))),
                grid=BTH_GRID, min=low, max=high, r2=_r2(width, anchor * decay))


def fit_thickness(rows):
    """H_bi = c0 + c1 * exp(-c2 * L_bi/L_b) + eps,  eps ~ SkewNormal"""
    x = (rows['LTH'] / rows['Lb']).to_numpy(dtype=float)
    y = rows['THK'].to_numpy(dtype=float)
    result = least_squares(lambda th: y - (th[0] + th[1] * np.exp(-th[2] * x)),
                           [y.min(), max(y.max() - y.min(), 1e-3), 2.85],
                           bounds=([0, 0, 0.01], [np.inf, np.inf, 20]))
    c0, c1, c2 = result.x
    pred = c0 + c1 * np.exp(-c2 * x)
    shape, loc, scale = skewnorm.fit(y - pred)
    return dict(form='exp_decay', c0=float(c0), c1=float(c1), c2=float(c2),
                noise=dict(kind='skewnorm', shape=float(shape),
                           loc=float(loc), scale=float(scale)),
                grid=THK_GRID, min=float(y.min()), max=float(y.max()), r2=_r2(y, pred))


def fit_bevel(rows, columns):
    """베벨 길이. BEVEL_TWO_STAGE 면 발생 여부(로지스틱) + 발생분 선형식."""
    predictors = ('THK', 'LTH', 'MARK_LTH')
    zero_rate = float((columns['BVL_LTH'] == 0).mean())
    if not BEVEL_TWO_STAGE:
        spec = ols(columns, predictors, columns['BVL_LTH'])
        spec.update(form='linear', floor=0.0, zero_rate=zero_rate, two_stage=False)
        return spec

    positive = rows['BVL_LTH'].to_numpy(dtype=float) > 0
    occurrence = logistic(columns, predictors, positive.astype(float))
    positive_columns = {name: values[positive] for name, values in columns.items()}
    magnitude = ols(positive_columns, predictors, positive_columns['BVL_LTH'])
    return dict(form='linear', two_stage=True, floor=0.0, zero_rate=zero_rate,
                occurrence=occurrence, **magnitude)


# ════════════════════════════════════════════
# 계열 하나 적합
# ════════════════════════════════════════════
def fit_series(rows, rng):
    columns = {name: rows[name].to_numpy(dtype=float) for name in GEN_COLUMNS}

    spec = {
        'n_wo': int(len(rows)),
        'n_block': int(rows.groupby(KEY).ngroups),
        'LTH': fit_length(rows),
        'BTH': fit_width(rows, rng),
        'THK': fit_thickness(rows),
        'CUT_LTH': ols(columns, ('LTH',), columns['CUT_LTH']),
        'MARK_LTH': ols(columns, ('LTH',), columns['MARK_LTH']),
        'BVL_LTH': fit_bevel(rows, columns),
    }
    spec['CUT_LTH'].update(form='linear', floor=float(columns['CUT_LTH'].min()))
    spec['MARK_LTH'].update(form='linear', floor=0.0)

    positive = rows[rows['BVL_LTH'] > 0]
    if len(positive) > 10:
        positive_columns = {name: positive[name].to_numpy(dtype=float)
                            for name in ('BVL_LTH', 'CUT_LTH')}
        quantity = ols(positive_columns, ('BVL_LTH', 'CUT_LTH'),
                       positive['BV_QTY'].to_numpy(dtype=float))
        quantity.update(form='linear', integer=True, min=1, gate='BVL_LTH > 0')
        spec['BV_QTY'] = quantity
    else:
        print(f'[WARN] 베벨 표본 {len(positive)}건 → BV_QTY 생략')
        spec['BV_QTY'] = None

    parts = ols(columns, ('CUT_LTH', 'LTH', 'MARK_LTH'), columns['PTLST_QTY'])
    parts.update(form='linear', integer=True, min=1)
    spec['PTLST_QTY'] = parts

    spec['WGT'] = dict(form='identity', rho=RHO,
                       note='WGT = round(rho * LTH * BTH * THK)')
    return spec


def fit_wo_params(wo):
    out = {}
    rng = np.random.default_rng(0)
    for series in SERIES:
        rows = wo[wo['GYEL'].astype(str).str.upper() == series]
        if len(rows) < 30:
            print(f'[WARN] series={series} rows={len(rows)} → 건너뜀')
            continue
        out[series] = fit_series(rows, rng)
        _print_series(series, out[series])
    return {'schema': WO_PARAMS_SCHEMA,
            'options': dict(bevel_two_stage=BEVEL_TWO_STAGE),
            'series': out}


def _print_series(series, spec):
    print(f"\n[{series}] W/O n={spec['n_wo']}  블록 {spec['n_block']}개")
    lth, bth, thk = spec['LTH'], spec['BTH'], spec['THK']
    print(f"    LTH       L_b*(1-({lth['a']:.3f}-{lth['b']:.3f}/n)u^{lth['p']:.3f})"
          f"{'':16s}R²={lth['r2']:.3f}")
    print(f"    BTH       B_b*(1-({bth['a']:.3f}-{bth['b']:.3f}/n)u^{bth['p']:.3f})"
          f"*exp(N(0,{bth['noise']['sd']:.3f}²))  R²={bth['r2']:.3f}")
    print(f"    THK       {thk['c0']:.2f}+{thk['c1']:.2f}exp(-{thk['c2']:.2f}x)"
          f"+SkewNorm({thk['noise']['shape']:.2f},{thk['noise']['loc']:.2f},"
          f"{thk['noise']['scale']:.2f}){'':4s}R²={thk['r2']:.3f}")
    for item in ('CUT_LTH', 'MARK_LTH', 'BVL_LTH', 'BV_QTY', 'PTLST_QTY'):
        detail = spec[item]
        if detail is None:
            print(f'    {item:9s} (표본 부족)')
            continue
        terms = '  '.join(f"{name}={detail[name]:+.5f}" for name in detail['x'])
        print(f"    {item:9s} const={detail['const']:+.4f}  {terms}"
              f"  sd={detail['noise']['sd']:.3f}  R²={detail['r2']:.3f}")
        if detail.get('two_stage'):
            occ = detail['occurrence']
            occ_terms = '  '.join(f"{name}={occ[name]:+.5f}" for name in occ['x'])
            print(f"    {'':9s} P(>0)=sig(const={occ['const']:+.4f}  {occ_terms})"
                  f"  적중률={occ['accuracy']:.3f}  실적 0비율={detail['zero_rate']:.3f}")


# ════════════════════════════════════════════
# 생성 (검증용. shipyard_data_generator 와 동일한 규칙)
# ════════════════════════════════════════════
def generate_block(block, spec, rng):
    n = int(block['STL_QTY'])
    u = np.arange(1, n + 1) / n
    anchor_l, anchor_b = float(block['LTH']), float(block['BTH'])

    q = spec['LTH']
    length = np.round(anchor_l * _decay(u, n, q['a'], q['b'], q['p']) / q['grid']) * q['grid']
    length = np.clip(length, q['min'], q['max'])

    q = spec['BTH']
    width = anchor_b * _decay(u, n, q['a'], q['b'], q['p']) \
        * np.exp(rng.normal(0, q['noise']['sd'], n))
    width = np.clip(np.round(width / q['grid']) * q['grid'], q['min'], q['max'])

    q, e = spec['THK'], spec['THK']['noise']
    thickness = q['c0'] + q['c1'] * np.exp(-q['c2'] * length / anchor_l) \
        + skewnorm.rvs(e['shape'], e['loc'], e['scale'], size=n, random_state=rng)
    thickness = np.clip(np.round(thickness / q['grid']) * q['grid'], q['min'], q['max'])

    values = {'LTH': length, 'BTH': width, 'THK': thickness}

    def linear(item):
        q = spec[item]
        return q['const'] + sum(q[name] * values[name] for name in q['x']) \
            + rng.normal(0, q['noise']['sd'], n)

    values['MARK_LTH'] = np.maximum(linear('MARK_LTH'), spec['MARK_LTH']['floor'])
    values['CUT_LTH'] = np.maximum(linear('CUT_LTH'), spec['CUT_LTH']['floor'])

    q = spec['BVL_LTH']
    bevel = np.maximum(linear('BVL_LTH'), 0.0)
    if q.get('two_stage'):
        occ = q['occurrence']
        eta = occ['const'] + sum(occ[name] * values[name] for name in occ['x'])
        occurs = rng.random(n) < 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        bevel = np.where(occurs, np.maximum(bevel, 0.1), 0.0)
    values['BVL_LTH'] = bevel

    if spec['BV_QTY'] is None:
        values['BV_QTY'] = np.zeros(n)
    else:
        raw = np.round(linear('BV_QTY'))
        values['BV_QTY'] = np.where(bevel > 0, np.maximum(raw, spec['BV_QTY']['min']), 0)

    values['PTLST_QTY'] = np.maximum(np.round(linear('PTLST_QTY')), spec['PTLST_QTY']['min'])
    values['STL_QTY'] = np.ones(n)
    values['WGT'] = np.round(RHO * length * width * thickness)
    for column in KEY:
        values[column] = block[column]
    return pd.DataFrame(values)


def validate(wo, blocks, params, seed=0):
    """생성 W/O 를 블록으로 합산해 블록 파일과 비교하고, 주변분포 KS 를 낸다."""
    rng = np.random.default_rng(seed)
    generated = pd.concat(
        [generate_block(row, params['series'][row['GYEL']], rng)
         for _, row in blocks.iterrows()
         if row['GYEL'] in params['series']], ignore_index=True)

    rollup = generated.groupby(KEY).agg(
        LTH=('LTH', 'max'), BTH=('BTH', 'max'), THK=('THK', 'max'),
        MARK_LTH=('MARK_LTH', 'sum'), CUT_LTH=('CUT_LTH', 'sum'),
        BVL_LTH=('BVL_LTH', 'sum'), BV_QTY=('BV_QTY', 'sum'),
        PTLST_QTY=('PTLST_QTY', 'sum'))
    actual = blocks.set_index(KEY)[list(rollup.columns)].join(rollup, rsuffix='_gen')

    print('\n[검증] 블록값 재현 (생성 W/O 합산 vs 블록 파일, 중앙 상대오차)')
    for column in rollup.columns:
        ratio = (actual[column + '_gen'] - actual[column]) / actual[column].replace(0, np.nan)
        print(f'    {column:10s} {100 * ratio.median():+7.1f}%')

    print('\n[검증] W/O 주변분포 KS')
    print(f"    {'':4s}" + ''.join(f'{c:>11s}' for c in GEN_COLUMNS))
    for series in params['series']:
        left = wo[wo['GYEL'] == series]
        right = generated[generated['GYEL'] == series]
        stats = (ks_2samp(left[c], right[c]).statistic for c in GEN_COLUMNS)
        print(f'    {series:4s}' + ''.join(f'{v:11.3f}' for v in stats))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) < 2:
        print('사용법: python fit_wo_params.py <W/O엑셀> <블록엑셀> [출력경로] [--no-validate]')
        sys.exit(1)
    wo_path, block_path = args[0], args[1]
    target = args[2] if len(args) > 2 else 'wo_params.json'

    wo, blocks = load(wo_path, block_path)
    params = fit_wo_params(wo)
    params['source'] = dict(wo=str(wo_path), block=str(block_path))
    with open(target, 'w', encoding='utf-8') as handle:
        json.dump(params, handle, ensure_ascii=False, indent=1)
    print(f'\n저장: {target}')

    if '--no-validate' not in sys.argv:
        validate(wo, blocks, params)


if __name__ == '__main__':
    main()