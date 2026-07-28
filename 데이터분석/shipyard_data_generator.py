# -*- coding: utf-8 -*-
"""
절단 블록·W/O 합성 데이터 생성기
================================

`블록_산출식_정리.md` / `W_O_산출식_정리.md` 의 산출식을 그대로 구현한다.
계열별 계수는 이 파일에 상수로 박혀 있다(외부 JSON·모듈 의존 없음).

생성 방향
    블록 6항목 → 강재수량만큼의 W/O → W/O 8항목 + 무게

    NP·NC·FN   길이 → 마킹 → 절단 ┬ 강재 → 두께
                                   └ 폭
    FL         길이 → 절단 ┬ 마킹
                           ├ 강재 → 두께
                           └ 폭

    W/O        길이 ┬ 폭
                    ├ 두께 ┐
                    ├ 마킹 ┴ 베벨길이 → 베벨수량
                    └ 절단 → 부재수량,  무게 = ρ·L·B·H

집계 계약(블록 = MAX/Σ W/O)은 강제하지 않는다. 산출식대로 뽑고 얼마나 어긋나는지
--report 로 확인한다. 강제 정합이 필요하면 그때 별도 단계로 붙이면 된다.

사용법
    python cut_data_generator.py --n 800 --series ALL --out generated
    python cut_data_generator.py --n 1137 --split proportional --report
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

# 학습 파이프라인에서 importlib로 로드될 때도 저장소 루트를 import 경로에 넣어
# 확정 TACT_TIME 계수를 canonical NP 생성기에서 직접 공유한다.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.data.report_formula_data_generator import (  # noqa: E402
    TACT_A_CUT,
    TACT_A_MARK,
    TACT_A_PTLST,
    TACT_A_THK,
)

SERIES = ('NP', 'NC', 'FN', 'FL')
RHO = 7.85e-6                     # kg/mm^3
BVL_FLOOR = 0.1                   # 베벨 발생분의 하한 (산출식 정리 6절)

# 발표자료 27쪽 Case 6 고정식을 전 계열 공통으로 적용한다(베벨 수량은 제외).
TACT_CUT = TACT_A_CUT
TACT_MARK = TACT_A_MARK
TACT_THK = TACT_A_THK
TACT_PT = TACT_A_PTLST
TACT_FORMULA_SCOPE = 'all_series_shared_np_ppt_case6'

# FN/FL/NC empirical 생성기가 학습 profile로 직렬화될 때 쓰는 schema 식별자.
EMPIRICAL_SERIES = ('FN', 'FL', 'NC')
EMPIRICAL_GENERATION_PROFILE_SCHEMA = 'shipyard_formula_generation_profile_v2'

# 두께 규격 격자 — 전 계열 합집합 42종
THK_SPECS = np.array([
    10.0, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
    16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0,
    23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 28.0, 30.0, 32.0,
    33.0, 34.0, 34.5, 35.0, 36.0, 37.0,
])

# ════════════════════════════════════════════════════════════════════
# 블록 계수
#   LTH       : N(mu, sd)                        → clip, 5mm 격자
#   MARK_LTH  : (k1·x + k0)·Gamma(k,θ)/(kθ)      → zero_rate, floor
#   CUT_LTH   : (k1·x + k0)·Gamma(k,θ)/(kθ)      → floor
#   STL_QTY   : (k1·CUT + k0)·Gamma(k,θ)/(kθ)    → 정수, clip
#   THK       : k1·ln(STL) + k0 + N(0,σ)         → 규격 스냅
#   BTH       : Vmax·CUT/(K+CUT) + N(0,σ)        → clip, 5mm 격자
# ════════════════════════════════════════════════════════════════════
BLOCK = {
    'NP': {
        'n_rows': 419,
        'chain': ('LTH', 'MARK_LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH'),
        'LTH': dict(mu=12705.023866348449, sd=5645.241009209492,
                    lo=3000.0, hi=21950.0),
        'MARK_LTH': dict(x='LTH', k1=0.018841405971306607, k0=0.0,
                         shape=1.0112728314212696, scale=0.7365187075280042,
                         zero_rate=0.021479713603818614, floor=0.2),
        'CUT_LTH': dict(x='MARK_LTH', k1=1.521092762704303, k0=73.0299174554929,
                        shape=3.4228594700019452, scale=0.27870322427782374,
                        floor=4.982),
        'STL_QTY': dict(k1=0.012105259379394915, k0=1.3330886209911543,
                        shape=7.740908060877767, scale=0.12732161457937424,
                        lo=1, hi=39),
        'THK': dict(k1=3.041222203979256, k0=13.97537966779211,
                    sd=5.100654798600746),
        'BTH': dict(vmax=3507.7359998045335, kk=40.58132412741576,
                    sd=624.7890979548806, lo=1000.0, hi=4385.0),
    },
    'NC': {
        'n_rows': 93,
        'chain': ('LTH', 'MARK_LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH'),
        'LTH': dict(mu=11112.58064516129, sd=4025.5164138675395,
                    lo=3080.0, hi=21720.0),
        'MARK_LTH': dict(x='LTH', k1=0.025251844123455245, k0=47.89634135515564,
                         shape=2.7460589083562366, scale=0.3550566084752878,
                         zero_rate=0.0, floor=18.959),
        'CUT_LTH': dict(x='MARK_LTH', k1=0.26675190270488947,
                        k0=19.066525421517163, shape=9.81475474013204,
                        scale=0.09627702260917517, floor=9.499),
        'STL_QTY': dict(k1=0.023130035730658832, k0=0.3600577846447488,
                        shape=11.044911859322092, scale=0.09164126567658469,
                        lo=1, hi=12),
        'THK': dict(k1=3.217420685458979, k0=19.60662304618185,
                    sd=5.363464885869228),
        'BTH': dict(vmax=4013.953944452499, kk=17.304190439100495,
                    sd=601.340628288398, lo=1115.0, hi=4375.0),
    },
    'FN': {
        'n_rows': 161,
        'chain': ('LTH', 'MARK_LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH'),
        'LTH': dict(mu=11939.968944099379, sd=5761.3174626364325,
                    lo=3000.0, hi=21605.0),
        'MARK_LTH': dict(x='LTH', k1=0.012324788708706525, k0=0.0,
                         shape=1.4089262396985942, scale=0.6547835860725644,
                         zero_rate=0.018633540372670808, floor=0.52),
        'CUT_LTH': dict(x='MARK_LTH', k1=0.3349939563697306,
                        k0=29.699512796298436, shape=4.464867237324649,
                        scale=0.21616960396629278, floor=3.222),
        'STL_QTY': dict(k1=0.02003068933479665, k0=0.42297623940407236,
                        shape=6.176052122655171, scale=0.17358574670282342,
                        lo=1, hi=8),
        'THK': dict(k1=1.2656840052953482, k0=17.95367870913191,
                    sd=6.384060965436009),
        'BTH': dict(vmax=3653.7845452542615, kk=13.279496762788536,
                    sd=733.9414229204657, lo=1000.0, hi=4400.0),
    },
    'FL': {
        'n_rows': 464,
        # FL 예외 — 마킹 길이를 절단 길이에서 산출한다
        'chain': ('LTH', 'CUT_LTH', 'MARK_LTH', 'STL_QTY', 'THK', 'BTH'),
        'LTH': dict(mu=17256.46551724138, sd=3293.8545446372364,
                    lo=6950.0, hi=22025.0),
        'CUT_LTH': dict(x='LTH', k1=0.024368302571041816,
                        k0=-104.05272777226699, shape=5.377022846618899,
                        scale=0.18674110290806478, floor=46.539),
        'MARK_LTH': dict(x='CUT_LTH', k1=0.427651674350216,
                         k0=38.27613151520023, shape=2.03276803528414,
                         scale=0.4900108874366913, zero_rate=0.0, floor=1.92),
        'STL_QTY': dict(k1=0.019111555245158382, k0=1.430442861127106,
                        shape=18.861445219682405, scale=0.05242449377986381,
                        lo=1, hi=19),
        'THK': dict(k1=4.697328476364492, k0=7.638849452734483,
                    sd=4.503393357198271),
        'BTH': dict(vmax=3789.0500056967226, kk=14.599358504692377,
                    sd=374.70871216042127, lo=2565.0, hi=4395.0),
    },
}

# ════════════════════════════════════════════════════════════════════
# W/O 계수
#   LTH   : L_b·{1-(a-b/n)·u^p}                       (오차 없음)
#   BTH   : B_b·{1-(a-b/n)·u^p}·exp(N(0,σ))
#   THK   : c0 + c1·exp(-c2·L_bi/L_b) + SkewNormal
#   CUT   : k1·L_bi + k0 + N(0,σ)                     → floor
#   MARK  : k1·L_bi + k0 + N(0,σ)                     → 0 절단
#   BVL   : sigmoid(g·[1,H,L,Mark]) 허들 → k·[1,H,L,Mark] + N(0,σ)
#   BV_Q  : k1·BVL + k2·CUT + k0 + N(0,σ)             → BVL>0 에서만, 정수·min 1
#   PTLST : k0 + k1·CUT + k2·L + k3·Mark + N(0,σ)     → 정수·min 1
# ════════════════════════════════════════════════════════════════════
WO = {
    'NP': {
        'LTH': dict(a=0.8753528919207403, b=1.0119117122960395,
                    p=1.0495237159643231, lo=3000.0, hi=21950.0),
        'BTH': dict(a=0.5696190762081534, b=0.6553358365604571,
                    p=0.8938961906461587, sigma=0.2167264788877219,
                    lo=1000.0, hi=4385.0),
        'THK': dict(c0=13.525157873645988, c1=21.054870040841383,
                    c2=10.121859777812901, shape=5.76907516003843,
                    loc=-4.402475110098419, scale=5.9989944319056425,
                    lo=6.0, hi=36.0),
        'CUT_LTH': dict(k0=18.96218250964598, k1=0.005031054122097634,
                        sd=41.22853629053636, floor=2.426),
        'MARK_LTH': dict(k0=-12.103769385809954, k1=0.005151987475644843,
                         sd=17.352985997535058),
        'BVL_OCC': dict(g0=-4.259981961387792, g_thk=0.2224275562208059,
                        g_lth=-2.109327889847194e-05,
                        g_mark=0.006398903458756121),
        'BVL_LTH': dict(k0=-16.424980825105333, k_thk=0.9975177274301635,
                        k_lth=0.0007769361615634907,
                        k_mark=0.007648605658697198, sd=8.785568110321982),
        'BV_QTY': dict(k0=1.3980036898636952, k_bvl=0.08228793414903826,
                       k_cut=0.01837309404728331, sd=2.426773778578712),
        'PTLST_QTY': dict(k0=6.467079089185945, k_cut=0.17605049596193798,
                          k_lth=-0.0005709879526274164,
                          k_mark=-0.10314494110554369, sd=7.848088192204777),
    },
    'NC': {
        'LTH': dict(a=0.5003466067415824, b=0.5668583351174328,
                    p=1.138163060436232, lo=3080.0, hi=21720.0),
        'BTH': dict(a=0.25873015189251447, b=0.29733683121414956,
                    p=0.05000000000000002, sigma=0.16113281180150807,
                    lo=1000.0, hi=4375.0),
        'THK': dict(c0=4.22087207887508e-10, c1=29.00415366650975,
                    c2=0.3451812899040317, shape=2.9983394893917974,
                    loc=-5.920410148927829, scale=7.888281995952374,
                    lo=12.0, hi=36.0),
        'CUT_LTH': dict(k0=19.499036017818028, k1=0.0018632019101319367,
                        sd=14.681377108810134, floor=8.137),
        'MARK_LTH': dict(k0=49.78715544614589, k1=0.006784031740620893,
                         sd=42.18034348907734),
        'BVL_OCC': dict(g0=-8.605559033407271, g_thk=0.382100757448087,
                        g_lth=-6.805984785002518e-06,
                        g_mark=0.002223706347539143),
        'BVL_LTH': dict(k0=-4.336887801523974, k_thk=0.4403805810853394,
                        k_lth=-0.0007047506559974548,
                        k_mark=0.08250345580260077, sd=6.799530258601785),
        'BV_QTY': dict(k0=-0.4676293892709408, k_bvl=0.09125128996330487,
                       k_cut=0.061915727385374014, sd=1.4698068278982814),
        'PTLST_QTY': dict(k0=1.6616238465954192, k_cut=0.12540546165646943,
                          k_lth=-0.0003732505062562508,
                          k_mark=0.0018035271802951634, sd=1.5071589286278329),
    },
    'FN': {
        'LTH': dict(a=0.3526051588766051, b=0.375929028037439,
                    p=1.874847970147295, lo=3000.0, hi=21605.0),
        'BTH': dict(a=0.20495295641079891, b=0.21639558005926343,
                    p=0.796020560549716, sigma=0.1118747724685818,
                    lo=1000.0, hi=4400.0),
        'THK': dict(c0=17.434206882015328, c1=30.121898003064075,
                    c2=6.936870365888734, shape=12.426530070639048,
                    loc=-6.689944272367315, scale=8.707203979465998,
                    lo=10.0, hi=37.0),
        'CUT_LTH': dict(k0=15.402129867501175, k1=0.0020037954169036445,
                        sd=15.523637079925717, floor=1.831),
        'MARK_LTH': dict(k0=16.65350313061878, k1=0.004726899597854636,
                         sd=38.13290351702391),
        'BVL_OCC': dict(g0=-3.4967986744084656, g_thk=0.22128030687522096,
                        g_lth=-1.6920841040567186e-06,
                        g_mark=-0.01449198516584997),
        'BVL_LTH': dict(k0=-14.0296353965498, k_thk=0.7075718031435543,
                        k_lth=0.0016875103866008656,
                        k_mark=-0.05273766200063947, sd=8.789147852935416),
        'BV_QTY': dict(k0=2.477241724247859, k_bvl=0.0020894139724280045,
                       k_cut=-0.015741842337670384, sd=1.4250176171521685),
        'PTLST_QTY': dict(k0=1.7447504788239252, k_cut=0.07509552211201778,
                          k_lth=-0.00017243688077966945,
                          k_mark=-0.0097468308352835, sd=1.1397706995951753),
    },
    'FL': {
        'LTH': dict(a=0.38654610176126064, b=0.9157716106308318,
                    p=2.0827141066573933, lo=3000.0, hi=22025.0),
        'BTH': dict(a=0.21461685909114495, b=0.3419377450706159,
                    p=0.31437742947899855, sigma=0.11713189235888422,
                    lo=1000.0, hi=4395.0),
        'THK': dict(c0=12.03791755005579, c1=8.466161336862635,
                    c2=1.3055165440992553, shape=5.778285577922832,
                    loc=-3.9588941865533496, scale=5.223872491131868,
                    lo=10.0, hi=35.0),
        'CUT_LTH': dict(k0=9.020846019293094, k1=0.002096713305117194,
                        sd=7.545854874513823, floor=4.093),
        'MARK_LTH': dict(k0=10.658686212090693, k1=0.0007906961404045494,
                         sd=20.663875077603436),
        'BVL_OCC': dict(g0=-5.382546107549673, g_thk=0.4077724555189376,
                        g_lth=-0.00014221047189543192,
                        g_mark=0.0013221692631045247),
        'BVL_LTH': dict(k0=-7.471147380715477, k_thk=0.6910584474717639,
                        k_lth=9.579953756400084e-05,
                        k_mark=0.1493255558451073, sd=7.771602903000136),
        'BV_QTY': dict(k0=1.0890631927919727, k_bvl=0.03447068088533162,
                       k_cut=-0.0030099647674790946, sd=0.712113821936909),
        'PTLST_QTY': dict(k0=1.321722957248037, k_cut=0.02391866881820579,
                          k_lth=-7.399266139166208e-05,
                          k_mark=-0.0007985922007822014, sd=0.4427593455129519),
    },
}

BLOCK_COLUMNS = ['PROJ_NO', 'BLK_NO', 'GYEL',
                 'LTH', 'MARK_LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH']
WO_COLUMNS = ['WO_NO', 'PROJ_NO', 'BLK_NO', 'GYEL',
              'LTH', 'BTH', 'THK', 'MARK_LTH', 'CUT_LTH',
              'BVL_LTH', 'BV_QTY', 'PTLST_QTY', 'STL_QTY', 'WGT']

PROJECT_NO_START = 2601
DEFAULT_BLOCKS_PER_PROJECT = 30


# ════════════════════════════════════════════════════════════════════
# 수치 유틸
# ════════════════════════════════════════════════════════════════════
def round_half_up(values, grid=1.0):
    """np.round 의 은행가 반올림을 피한다. 0.5 는 위로 올린다."""
    return np.floor(np.asarray(values, dtype=float) / grid + 0.5) * grid


def gamma_unit(rng, shape, scale, size):
    """평균 1로 정규화한 곱셈형 감마 오차."""
    return rng.gamma(shape, scale, size) / max(shape * scale, 1e-12)


def snap_specs(values, specs=THK_SPECS):
    values = np.clip(np.asarray(values, dtype=float), specs.min(), specs.max())
    return specs[np.abs(values[:, None] - specs[None, :]).argmin(axis=1)]


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _calculate_tact_time(cut_length, mark_length, thickness, part_quantity, bevel_quantity):
    """발표자료 27쪽 Case 6 고정식으로 W/O 택트타임(분)을 계산한다. 베벨 수량은 식에 들어가지 않는다."""

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


def _validate_optional_positive_integer_sequence(values, expected_length, field_name):
    """wo_counts/block_seeds처럼 블록당 하나씩 오는 양의 정수 시퀀스를 fallback 없이 검증한다."""

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


# ════════════════════════════════════════════════════════════════════
# 블록 생성
# ════════════════════════════════════════════════════════════════════
def generate_blocks(block_params, n_blocks, rng, series, thk_specs=THK_SPECS):
    """블록 계수 dict를 받아 블록 n_blocks 개를 벡터화 생성한다."""
    par = block_params
    thk_specs = np.asarray(thk_specs, dtype=float)
    size = int(n_blocks)
    out = {}

    for item in par['chain']:
        spec = par[item]

        if item == 'LTH':
            value = rng.normal(spec['mu'], spec['sd'], size)
            value = np.clip(value, spec['lo'], spec['hi'])
            out['LTH'] = round_half_up(value, 5.0)

        elif item in ('MARK_LTH', 'CUT_LTH'):
            x = out[spec['x']]
            value = (spec['k1'] * x + spec['k0']) * gamma_unit(
                rng, spec['shape'], spec['scale'], size)
            value = np.maximum(value, spec['floor'])
            if spec.get('zero_rate', 0.0) > 0.0:
                value = np.where(rng.random(size) < spec['zero_rate'], 0.0, value)
            out[item] = value

        elif item == 'STL_QTY':
            value = (spec['k1'] * out['CUT_LTH'] + spec['k0']) * gamma_unit(
                rng, spec['shape'], spec['scale'], size)
            out['STL_QTY'] = np.clip(round_half_up(value), spec['lo'],
                                     spec['hi']).astype(int)

        elif item == 'THK':
            value = (spec['k1'] * np.log(np.maximum(out['STL_QTY'], 1))
                     + spec['k0'] + rng.normal(0.0, spec['sd'], size))
            out['THK'] = snap_specs(value, thk_specs)

        elif item == 'BTH':
            cut = out['CUT_LTH']
            value = (spec['vmax'] * cut / (spec['kk'] + cut)
                     + rng.normal(0.0, spec['sd'], size))
            value = np.clip(value, spec['lo'], spec['hi'])
            out['BTH'] = round_half_up(value, 5.0)

    frame = pd.DataFrame({key: out[key] for key in
                          ('LTH', 'MARK_LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH')})
    frame['GYEL'] = series
    return frame


# ════════════════════════════════════════════════════════════════════
# W/O 생성
# ════════════════════════════════════════════════════════════════════
def generate_wos(wo_params, blocks, rng, series):
    """W/O 계수 dict와 블록 표를 받아 블록 STL_QTY 만큼의 W/O 를 벡터화 생성한다."""
    par = wo_params
    counts = blocks['STL_QTY'].to_numpy().astype(int)
    total = int(counts.sum())

    blk_index = np.repeat(np.arange(len(blocks)), counts)
    n = counts[blk_index].astype(float)                       # 블록 강재수량
    start = np.repeat(np.cumsum(counts) - counts, counts)
    order = np.arange(total) - start + 1                      # i = 1..n (길이 내림차순)
    u = order / n

    block_lth = blocks['LTH'].to_numpy()[blk_index]
    block_bth = blocks['BTH'].to_numpy()[blk_index]

    # 1. 길이 — 오차 없음
    spec = par['LTH']
    decay = 1.0 - (spec['a'] - spec['b'] / n) * np.power(u, spec['p'])
    lth = round_half_up(np.clip(block_lth * decay, spec['lo'], spec['hi']), 5.0)

    # 2. 폭 — 같은 감쇠 형태 + 곱셈형 오차
    spec = par['BTH']
    decay = 1.0 - (spec['a'] - spec['b'] / n) * np.power(u, spec['p'])
    bth = block_bth * decay * np.exp(rng.normal(0.0, spec['sigma'], total))
    bth = round_half_up(np.clip(bth, spec['lo'], spec['hi']), 5.0)

    # 3. 두께 — 길이비 지수감쇠 + 왜정규 오차
    spec = par['THK']
    ratio = lth / np.maximum(block_lth, 1e-9)
    thk = (spec['c0'] + spec['c1'] * np.exp(-spec['c2'] * ratio)
           + skewnorm.rvs(spec['shape'], loc=spec['loc'], scale=spec['scale'],
                          size=total, random_state=rng))
    thk = round_half_up(np.clip(thk, spec['lo'], spec['hi']), 0.5)

    # 4. 절단 길이
    spec = par['CUT_LTH']
    cut = spec['k1'] * lth + spec['k0'] + rng.normal(0.0, spec['sd'], total)
    cut = np.maximum(cut, spec['floor'])

    # 5. 마킹 길이 — 음수는 0
    spec = par['MARK_LTH']
    mark = spec['k1'] * lth + spec['k0'] + rng.normal(0.0, spec['sd'], total)
    mark = np.maximum(mark, 0.0)

    # 6. 베벨 길이 — 2단(발생 여부 → 발생분 길이)
    occ = par['BVL_OCC']
    probability = sigmoid(occ['g0'] + occ['g_thk'] * thk
                          + occ['g_lth'] * lth + occ['g_mark'] * mark)
    occurs = rng.random(total) < probability
    spec = par['BVL_LTH']
    bvl = (spec['k0'] + spec['k_thk'] * thk + spec['k_lth'] * lth
           + spec['k_mark'] * mark + rng.normal(0.0, spec['sd'], total))
    bvl = np.where(occurs, np.maximum(bvl, BVL_FLOOR), 0.0)

    # 7. 베벨 수량 — 베벨 길이 > 0 인 W/O 에만
    spec = par['BV_QTY']
    bv_qty = (spec['k0'] + spec['k_bvl'] * bvl + spec['k_cut'] * cut
              + rng.normal(0.0, spec['sd'], total))
    bv_qty = np.where(bvl > 0, np.maximum(round_half_up(bv_qty), 1.0), 0.0)

    # 8. 부재 수량
    spec = par['PTLST_QTY']
    ptlst = (spec['k0'] + spec['k_cut'] * cut + spec['k_lth'] * lth
             + spec['k_mark'] * mark + rng.normal(0.0, spec['sd'], total))
    ptlst = np.maximum(round_half_up(ptlst), 1.0)

    frame = pd.DataFrame({
        'BLK_INDEX': blk_index,
        'WO_SEQ': order.astype(int),
        'LTH': lth, 'BTH': bth, 'THK': thk,
        'MARK_LTH': mark, 'CUT_LTH': cut,
        'BVL_LTH': bvl, 'BV_QTY': bv_qty.astype(int),
        'PTLST_QTY': ptlst.astype(int),
    })
    frame['STL_QTY'] = 1
    frame['WGT'] = round_half_up(RHO * lth * bth * thk).astype(int)   # 항등식
    frame['GYEL'] = series
    return frame


# ════════════════════════════════════════════════════════════════════
# 식별자
# ════════════════════════════════════════════════════════════════════
def assign_identifiers(blocks, wos, blocks_per_project=DEFAULT_BLOCKS_PER_PROJECT):
    blocks = blocks.reset_index(drop=True)
    position = np.arange(len(blocks))
    blocks['PROJ_NO'] = [f'H{PROJECT_NO_START + int(v):04d}'
                         for v in position // blocks_per_project]
    blocks['BLK_NO'] = [f'{series}{int(serial):03d}' for series, serial
                        in zip(blocks['GYEL'], position % blocks_per_project + 1)]

    lookup = blocks[['PROJ_NO', 'BLK_NO']].to_numpy()
    wos = wos.copy()
    wos['PROJ_NO'] = lookup[wos['BLK_INDEX'].to_numpy(), 0]
    wos['BLK_NO'] = lookup[wos['BLK_INDEX'].to_numpy(), 1]
    wos['WO_NO'] = [f'{proj}{blk}-{int(seq):02d}' for proj, blk, seq
                    in zip(wos['PROJ_NO'], wos['BLK_NO'], wos['WO_SEQ'])]

    if blocks[['PROJ_NO', 'BLK_NO']].duplicated().any() \
            or wos['WO_NO'].duplicated().any():
        raise RuntimeError('duplicated_generated_identifier')
    return blocks[BLOCK_COLUMNS], wos[WO_COLUMNS]


# ════════════════════════════════════════════════════════════════════
# 정합 리포트 — 강제하지 않고 어긋난 정도만 본다
# ════════════════════════════════════════════════════════════════════
def contract_report(blocks, wos):
    """블록값 대비 W/O 집계값의 중앙 상대오차(%)."""
    keyed = wos.copy()
    keyed['KEY'] = keyed['PROJ_NO'] + '|' + keyed['BLK_NO']
    base = blocks.copy()
    base['KEY'] = base['PROJ_NO'] + '|' + base['BLK_NO']
    base = base.set_index('KEY')

    aggregated = keyed.groupby('KEY').agg(
        LTH=('LTH', 'max'), BTH=('BTH', 'max'), THK=('THK', 'max'),
        MARK_LTH=('MARK_LTH', 'sum'), CUT_LTH=('CUT_LTH', 'sum'),
        STL_QTY=('STL_QTY', 'sum'))
    rows = []
    for item, rule in (('LTH', 'MAX'), ('BTH', 'MAX'), ('THK', 'MAX'),
                       ('MARK_LTH', 'SUM'), ('CUT_LTH', 'SUM'),
                       ('STL_QTY', 'COUNT')):
        left = aggregated[item].reindex(base.index).to_numpy(dtype=float)
        right = base[item].to_numpy(dtype=float)
        valid = np.abs(right) > 1e-9
        error = np.median((left[valid] - right[valid]) / right[valid]) * 100.0
        rows.append({'항목': item, '계약': rule, '중앙 상대오차(%)': round(error, 2)})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════════════════
def allocate(series_names, total, split):
    if split == 'proportional':
        weights = np.array([BLOCK[name]['n_rows'] for name in series_names], float)
    else:
        weights = np.ones(len(series_names))
    share = weights / weights.sum() * int(total)
    base = np.floor(share).astype(int)
    remainder = int(total) - int(base.sum())
    if remainder > 0:
        base[np.argsort(-(share - base))[:remainder]] += 1
    return {name: int(value) for name, value in zip(series_names, base)}


def generate(series_names, counts, seed=2026, blocks_per_project=DEFAULT_BLOCKS_PER_PROJECT):
    block_frames, wo_frames, offset = [], [], 0
    for index, name in enumerate(series_names):
        count = int(counts[name])
        if count <= 0:
            continue
        rng = np.random.default_rng(seed + index)
        blocks = generate_blocks(BLOCK[name], count, rng, name)
        wos = generate_wos(WO[name], blocks, rng, name)
        wos['BLK_INDEX'] += offset
        offset += len(blocks)
        block_frames.append(blocks)
        wo_frames.append(wos)
    blocks = pd.concat(block_frames, ignore_index=True)
    wos = pd.concat(wo_frames, ignore_index=True)
    return assign_identifiers(blocks, wos, blocks_per_project)


# ════════════════════════════════════════════════════════════════════
# 학습 파이프라인 어댑터
#   MIXED Phase 1/2 학습은 FN/FL/NC W/O를 이 어댑터로 생성한다. 파이프라인이
#   물리 블록 공동분포에서 정한 블록당 W/O 수(wo_counts)와 블록 seed(block_seeds)를
#   그대로 존중하고, BLK_ID 그룹키와 확정 TACT_TIME(Case6)까지 붙여 반환한다.
#   NP는 이 어댑터를 쓰지 않고 report_formula 고정식으로 간다.
# ════════════════════════════════════════════════════════════════════
ADAPTER_WO_IDENTITY = ('BLK_ID', 'GYEL')
ADAPTER_WO_FEATURES = (
    'LTH', 'BTH', 'THK', 'CUT_LTH', 'MARK_LTH', 'BVL_LTH',
    'STL_QTY', 'BV_QTY', 'TACT_TIME', 'PTLST_QTY',
)
ADAPTER_BLOCK_COLUMNS = (
    'BLK_ID', 'GYEL', 'LTH', 'BTH', 'THK', 'CUT_LTH', 'MARK_LTH', 'BVL_LTH',
    'STL_QTY', 'WO_QTY', 'BV_QTY', 'TACT_TIME', 'PTLST_QTY',
)


class ShipyardGenerator:
    """FN/FL/NC 계열의 고정 산출식 생성기를 학습 profile 계약으로 감싼 어댑터.

    계수는 이 모듈의 `BLOCK`/`WO` 상수에서 오고, 학습 중에는 재적합 없이 고정
    profile(payload)만으로 재구성해 생성한다. NP는 canonical report_formula 고정식이
    담당하므로 이 어댑터는 NP를 거부한다.
    """

    def __init__(self, wo_path=None, block_path=None, mode='spearman', series=None):
        normalized = str(series).strip().upper() if series is not None else None
        if not normalized:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.__init__] "
                "cause=empirical_series_required"
            )
            raise RuntimeError("empirical_series_required")
        if normalized == 'NP':
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.__init__] "
                "cause=np_fixed_formula_required"
            )
            raise RuntimeError("np_fixed_formula_required")
        if normalized not in EMPIRICAL_SERIES:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.__init__] "
                f"cause=unsupported_empirical_series series={normalized}"
            )
            raise RuntimeError(f"unsupported_empirical_series:{normalized}")
        self.series = normalized
        self.mode = mode
        self.wo_path = wo_path
        self.block_path = block_path
        self.block_params = deepcopy(BLOCK[normalized])
        self.wo_params = deepcopy(WO[normalized])
        self.thk_specs = np.array(THK_SPECS, dtype=float)
        self._fitted = False

    def fit(self):
        """계수가 상수로 고정되어 있으므로 Excel 재적합 없이 fit 상태만 확정한다."""

        self._fitted = True
        return self

    def to_generation_profile(self):
        """고정 profile JSON에 직렬화할 계열별 계수 payload를 반환한다."""

        return {
            'schema': EMPIRICAL_GENERATION_PROFILE_SCHEMA,
            'series': self.series,
            'block_params': deepcopy(self.block_params),
            'wo_params': deepcopy(self.wo_params),
            'thk_specs': [float(value) for value in self.thk_specs],
            'tact': {
                'CUT_LTH': TACT_CUT,
                'MARK_LTH': TACT_MARK,
                'THK': TACT_THK,
                'PTLST_QTY': TACT_PT,
                'scope': TACT_FORMULA_SCOPE,
            },
        }

    @classmethod
    def from_generation_profile(cls, profile):
        """고정 profile payload에서 재적합 없이 생성기를 복원한다."""

        required = ('schema', 'series', 'block_params', 'wo_params', 'thk_specs')
        missing = [key for key in required if key not in profile]
        if missing:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.from_generation_profile] "
                f"cause=missing_profile_keys keys={missing}"
            )
            raise RuntimeError(f"missing_empirical_profile_keys:{missing}")
        if profile['schema'] != EMPIRICAL_GENERATION_PROFILE_SCHEMA:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.from_generation_profile] "
                f"cause=schema_mismatch actual={profile['schema']} "
                f"expected={EMPIRICAL_GENERATION_PROFILE_SCHEMA}"
            )
            raise RuntimeError("empirical_profile_schema_mismatch")
        series = str(profile['series']).strip().upper()
        if series not in EMPIRICAL_SERIES:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.from_generation_profile] "
                f"cause=unsupported_empirical_series series={series}"
            )
            raise RuntimeError(f"unsupported_empirical_series:{series}")
        generator = object.__new__(cls)
        generator.series = series
        generator.mode = str(profile.get('mode', 'spearman'))
        generator.wo_path = None
        generator.block_path = None
        generator.block_params = deepcopy(profile['block_params'])
        generator.wo_params = deepcopy(profile['wo_params'])
        generator.thk_specs = np.asarray(profile['thk_specs'], dtype=float)
        generator._fitted = True
        return generator

    def _aggregate_blocks(self, work_orders):
        """생성 W/O를 블록 집계 계약(WO_QTY=count, MAX/Σ)으로 되접는다."""

        grouped = work_orders.groupby(['BLK_ID', 'GYEL'], sort=True)
        blocks = grouped.agg(
            LTH=('LTH', 'max'),
            BTH=('BTH', 'max'),
            THK=('THK', 'max'),
            CUT_LTH=('CUT_LTH', 'sum'),
            MARK_LTH=('MARK_LTH', 'sum'),
            BVL_LTH=('BVL_LTH', 'sum'),
            STL_QTY=('STL_QTY', 'sum'),
            WO_QTY=('LTH', 'size'),
            BV_QTY=('BV_QTY', 'sum'),
            TACT_TIME=('TACT_TIME', 'max'),
            PTLST_QTY=('PTLST_QTY', 'sum'),
        ).reset_index()
        return blocks[list(ADAPTER_BLOCK_COLUMNS)]

    def generate(self, n_blocks, seed=2026, wo_counts=None, block_seeds=None):
        """블록 n_blocks개를 생성한다. 반환: (wo_df, blk_df). wo_df에 BLK_ID를 부여한다.

        wo_counts가 주어지면 블록당 W/O 수를 그 값으로 강제하고, block_seeds가
        주어지면 블록마다 그 seed로 재현 가능하게 생성한다.
        """

        if not self._fitted:
            self.fit()
        block_count = int(n_blocks)
        if block_count <= 0:
            print(
                "[ERROR][shipyard_data_generator.ShipyardGenerator.generate] "
                f"cause=invalid_n_blocks value={n_blocks}"
            )
            raise RuntimeError("invalid_n_blocks")
        resolved_wo_counts = _validate_optional_positive_integer_sequence(
            wo_counts, block_count, 'wo_counts'
        )
        resolved_block_seeds = _validate_optional_positive_integer_sequence(
            block_seeds, block_count, 'block_seeds'
        )
        shared_rng = np.random.default_rng(seed)
        wo_frames = []
        for block_id in range(block_count):
            block_rng = (
                np.random.default_rng(resolved_block_seeds[block_id])
                if resolved_block_seeds is not None
                else shared_rng
            )
            block = generate_blocks(
                self.block_params, 1, block_rng, self.series, self.thk_specs
            )
            if resolved_wo_counts is not None:
                block = block.copy()
                block['STL_QTY'] = int(resolved_wo_counts[block_id])
            work_orders = generate_wos(self.wo_params, block, block_rng, self.series)
            work_orders = work_orders.copy()
            work_orders.insert(0, 'BLK_ID', block_id)
            wo_frames.append(work_orders)
        wo_df = pd.concat(wo_frames, ignore_index=True)
        wo_df['TACT_TIME'] = _calculate_tact_time(
            wo_df['CUT_LTH'].to_numpy(dtype=float),
            wo_df['MARK_LTH'].to_numpy(dtype=float),
            wo_df['THK'].to_numpy(dtype=float),
            wo_df['PTLST_QTY'].to_numpy(dtype=float),
            wo_df['BV_QTY'].to_numpy(dtype=float),
        )
        wo_out = wo_df[list(ADAPTER_WO_IDENTITY) + list(ADAPTER_WO_FEATURES)].reset_index(drop=True)
        blk_out = self._aggregate_blocks(wo_out)
        return wo_out, blk_out


def main():
    parser = argparse.ArgumentParser(description='절단 블록·W/O 합성 데이터 생성')
    parser.add_argument('--series', default='ALL', help="NP / 'NP,FL' / ALL")
    parser.add_argument('--n', type=int, default=800, help='생성 블록 수 총합')
    parser.add_argument('--split', choices=['equal', 'proportional'],
                        default='proportional', help='계열별 블록 수 배분')
    parser.add_argument('--blocks_per_project', type=int,
                        default=DEFAULT_BLOCKS_PER_PROJECT)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out', default='generated', help='출력 파일 접두사')
    parser.add_argument('--out_format', choices=['xlsx', 'csv'], default='xlsx')
    parser.add_argument('--report', action='store_true',
                        help='블록 ↔ W/O 집계 정합 리포트 출력')
    args = parser.parse_args()

    if str(args.series).strip().upper() == 'ALL':
        names = list(SERIES)
    else:
        names = [part.strip().upper() for part in str(args.series).split(',')
                 if part.strip()]
        unknown = [name for name in names if name not in BLOCK]
        if unknown:
            raise SystemExit(f'unknown_series: {unknown}')

    counts = allocate(names, args.n, args.split)
    blocks, wos = generate(names, counts, args.seed, args.blocks_per_project)

    if args.out_format == 'xlsx':
        block_path, wo_path = f'{args.out}_blk.xlsx', f'{args.out}_wo.xlsx'
        blocks.to_excel(block_path, index=False)
        wos.to_excel(wo_path, index=False)
    else:
        block_path, wo_path = f'{args.out}_blk.csv', f'{args.out}_wo.csv'
        blocks.to_csv(block_path, index=False, encoding='utf-8-sig')
        wos.to_csv(wo_path, index=False, encoding='utf-8-sig')

    print(f'[생성] 계열={names} 배분={counts} seed={args.seed}')
    print(f'[생성] 블록 {len(blocks)}행 → {block_path}')
    print(f'[생성] W/O  {len(wos)}행 → {wo_path}')
    print(blocks.groupby('GYEL')[['LTH', 'CUT_LTH', 'STL_QTY', 'THK', 'BTH']]
          .mean().round(1).to_string())

    if args.report:
        print('\n[정합] 블록값 대비 W/O 집계 (강제하지 않음)')
        print(contract_report(blocks, wos).to_string(index=False))


if __name__ == '__main__':
    main()