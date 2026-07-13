# -*- coding: utf-8 -*-
"""
조선소 블록·W/O 합성 데이터 생성기 (2가지 모드)
====================================================

실적 엑셀(WO/블록)에서 파라미터를 적합한 뒤, RL 학습용 합성 블록·작업지시(W/O)
데이터를 생성한다.

모드 (mode 인자)
  · 'spearman' (기본, 길이 직접) : 로그-로그·멱함수·코퓰러로 순위 구조·분포 보존
        → WO Spearman MAE 0.084 / 블록 0.179
  · 'pearson'  (선형식)          : 선형회귀+정규잔차로 선형 상관(Pearson) 보존
        → WO Pearson MAE 0.114 / 블록 0.123 (Spearman MAE 0.104 / 0.152)
    두 모드는 '확정식'을 공유하고, 물량 변수(두께 의존·마킹·절단·베벨·부재) 식만 다르다.

공통 생성 구조
  1. 블록 속성   : 길이 ~ N → 마킹 → 절단 → 강재(n) = 확정 사슬,
                   블록두께 = a·ln(강재) + b + 잔차 → 규격 스냅 (강재로 피팅)
  2. WO 길이     : 함수형(병합 패턴 + 멱감쇠), max = 블록길이 보존
  3. WO 두께     : μ(rL)+σ(rL)·SkewNormal, 0.5mm 빈도가중 스냅 (+모드별 ρ 코퓰러)
  4. WO 마킹·절단: WO 길이로 자유 생성 → 합 제약 후처리 보정
                   (pearson: 길이가중 가법 / spearman: 곱셈 스케일)
  5. WO 베벨·부재·택트: WO 직접 생성 (베벨길이 ← 두께+길이+마킹, 부재 ← 절단+길이+마킹)
     → 블록값: 베벨길이·베벨수량·부재 = Σ WO,  택트타임 = MAX WO

확정식(사용자 지정, 두 모드 공통, 재적합 안 함)
  · 블록길이 ~ N(13380.9, 5376.4²)
  · 블록MARK = 0.03386·길이 − 172.90 + N(0,238.4²)
  · 블록CUT  = 1.70636·MARK + 58.76 + N(0,218.7²)
  · 블록STL  = 0.01203·CUT + 2.014  → n(WO수) = round(STL)
  · WO TACT  = 0.3037·CUT + 0.1325·MARK + 0.4790·THK + 0.3840·PT
               (부재·두께 의존 — 생성에서 PT·THK 계산 후 마지막에 산출)
  · 블록두께 = a·ln(강재) + b + N(0,σ²) → 규격 스냅 (a,b,σ는 실적 적합)

선형식(pearson 모드 — 확정식 제외, 실적에서 적합)
  · 베벨길이 = a·두께 + b + N(0,σ)            (허들 유지)
  · 베벨수량 = round(a·베벨길이 + b + N(0,σ))
  · 부재수량 = round(a·절단 + b·길이 + c + N(0,σ))
  · 마킹·절단 = 길이 비례 선형 배분 + 정규노이즈 (합 보존)
  · 두께 = Pearson ρ 직접 부여(선형 조건부)

사용법
  from shipyard_data_generator import ShipyardGenerator
  gen = ShipyardGenerator(wo_xlsx, blk_xlsx, mode='spearman')   # 또는 'pearson'
  gen.fit()
  wo_df, blk_df = gen.generate(n_blocks=800, seed=2026)
  gen.compare(wo_df, blk_df, method='spearman')   # 'pearson'도 가능

  또는 CLI:  python shipyard_data_generator.py WO.xlsx 블록.xlsx --mode pearson --n 800
"""
import argparse
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import spearmanr, rankdata, norm, skewnorm

WCOLS = ['LTH', 'THK', 'MARK_LTH', 'CUT_LTH', 'BVL_LTH', 'BV_QTY', 'PTLST_QTY', 'TACT_TIME']
BCOLS = ['LTH', 'THK', 'MARK_LTH', 'CUT_LTH', 'BVL_LTH', 'BV_QTY', 'PTLST_QTY', 'STL_QTY', 'TACT_TIME']

# --- 확정식 상수 (사용자 지정) ---
BLK_LEN_MU, BLK_LEN_SD = 13380.9, 5376.4
BLK_MARK_A, BLK_MARK_B, BLK_MARK_SD = 0.03386, -172.90, 238.4
BLK_CUT_A, BLK_CUT_B, BLK_CUT_SD = 1.70636, 58.76, 218.7
BLK_STL_A, BLK_STL_B = 0.01203, 2.014
# WO 택트타임 계수: 0.3037·CUT + 0.1325·MARK + 0.4790·THK + 0.3840·PT
TACT_CUT, TACT_MARK, TACT_THK, TACT_PT = 0.3037, 0.1325, 0.4790, 0.3840


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


class ShipyardGenerator:
    def __init__(self, wo_xlsx, blk_xlsx, mode='spearman'):
        assert mode in ('spearman', 'pearson'), "mode는 'spearman' 또는 'pearson'"
        self.mode = mode
        self.wo = pd.read_excel(wo_xlsx)
        self.blk = pd.read_excel(blk_xlsx)
        g = self.wo.groupby(['PROJ_NO', 'BLK_NO'])
        self.wo['n_wo'] = g['LTH'].transform('size')
        self.woF = self.wo[self.wo['n_wo'] >= 2].copy()
        self._fitted = False

    # ================= 적합 =================
    def fit(self):
        woF = self.woF
        g = woF.groupby(['PROJ_NO', 'BLK_NO'])

        # --- WO 길이: 블록별 (a,b) 적합 후 n에 대한 경향 ---
        NB = []
        for _, b in g:
            vs = np.sort(b['LTH'].values)[::-1]
            n = len(vs)
            if n < 3 or len(np.unique(vs)) < 2:
                continue
            idx = _cluster_rel(vs)
            gpos = np.array([np.mean(grp) / (n - 1) for grp in idx])
            gm = np.array([vs[grp].mean() for grp in idx])
            M = vs.max()
            try:
                fab, _ = optimize.curve_fit(
                    lambda u, a, bb: M * _powf(u, a, bb), gpos, gm,
                    p0=[0.7, 1.0], bounds=([0.01, 0.1], [1.5, 3.0]), maxfev=5000)
                NB.append((n, fab[0], fab[1]))
            except Exception:
                pass
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
                mp[min(int(i / (nn - 1) * 10), 9)].append(
                    (vs[i] - vs[i + 1]) / max(vs[i], 1e-9) <= 0.05)
        self.puL = np.array([np.mean(mp[i]) if mp[i] else 0.3 for i in range(10)])
        self.floorL = max(int(woF['LTH'].quantile(.005)), 1)

        # --- WO 두께: μ(rL), σ(rL), 잔차 분포 ---
        woF = woF.assign(ML=g['LTH'].transform('max'))
        rLv = (woF['LTH'] / woF['ML']).values
        THKv = woF['THK'].values
        edg = np.linspace(0, 1, 11)
        cen = (edg[:-1] + edg[1:]) / 2
        mu_pts = np.array([THKv[(rLv >= edg[i]) & ((rLv < edg[i + 1]) if i < 9 else (rLv <= 1))].mean() for i in range(10)])
        sd_pts = np.array([THKv[(rLv >= edg[i]) & ((rLv < edg[i + 1]) if i < 9 else (rLv <= 1))].std() for i in range(10)])
        self.pmu, _ = optimize.curve_fit(lambda x, inf, amp, k: inf + amp * np.exp(-k * x), cen, mu_pts, p0=[12, 6, 3])
        self.psd = np.polyfit(cen, sd_pts, 2)
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
        # 합 보정 시 사용할 하한(실적 최소값) — 0/음수 방지
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

        # 블록 두께 = f(강재수량 n) — 강재가 사슬 끝(길이→마킹→절단→강재)이라
        # 두께를 강재로 피팅하면 두께가 길이·마킹·절단·부재와 자동 연결된다.
        # 블록 두께 = MAX(WO두께), 강재수량 = 블록당 WO 개수
        bagg = self.wo.groupby(['PROJ_NO', 'BLK_NO']).agg(THK=('THK', 'max'), STL=('LTH', 'size'))
        vc = bagg['THK'].value_counts(normalize=True)
        self.blk_thk_vals = np.array(sorted(vc.index))     # 두께 규격 격자 (스냅용)
        # 두께 ~ a·ln(강재) + b + N(0, σ²)
        self.thk_a, self.thk_b = np.polyfit(np.log(bagg['STL']), bagg['THK'], 1)
        self.thk_sig = float((bagg['THK'] - (self.thk_a * np.log(bagg['STL']) + self.thk_b)).std())
        self.n_max = int(woF.groupby(['PROJ_NO', 'BLK_NO']).size().max())
        self._fitted = True
        return self

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
        for _, b in self.woF.groupby(['PROJ_NO', 'BLK_NO']):
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
            return min(max(round(t * 2) / 2, 6), MT)
        w = cf * np.exp(-((cand - t) / 0.6) ** 2)
        return float(rng.choice(cand, p=w / w.sum())) if w.sum() > 0 else cand[np.argmin(np.abs(cand - t))]

    # ================= 생성 =================
    def _gen_one(self, rng):
        # 1) 블록 속성: 길이 → 마킹 → 절단 → 강재(n)  (확정 사슬 유지)
        ML = float(np.clip(rng.normal(BLK_LEN_MU, BLK_LEN_SD), self.floorL + 500, 30000))
        bMARK = max(BLK_MARK_A * ML + BLK_MARK_B + rng.normal(0, BLK_MARK_SD), 1.0)
        bCUT = max(BLK_CUT_A * bMARK + BLK_CUT_B + rng.normal(0, BLK_CUT_SD), 1.0)
        STL = BLK_STL_A * bCUT + BLK_STL_B
        n = int(np.clip(round(STL + rng.normal(0, 1.2)), 2, self.n_max))
        # 블록 두께 = f(강재수량 n): a·ln(n)+b + 잔차 → 가장 가까운 규격으로 스냅
        #   강재가 사슬 끝이라, 두께가 길이·마킹·절단·부재와 자동 연결됨
        thk_cont = self.thk_a * np.log(max(n, 1)) + self.thk_b + rng.normal(0, self.thk_sig)
        MT = float(self.blk_thk_vals[np.argmin(np.abs(self.blk_thk_vals - thk_cont))])

        # 2) WO 길이 (함수형, max=ML)
        if n < 2:
            L = np.array([ML])
        else:
            sizes = [1]
            for i in range(1, n):
                if rng.random() < self.puL[min(int(i / (n - 1) * 10), 9)]:
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

        # 4) WO 마킹·절단: WO 길이로 자유 생성 → 합 제약 후처리 보정
        #    (배분 대신 자유생성+스케일 → 마킹↔절단 동조 해소, 길이↔마킹 회복)
        if self.mode == 'pearson':
            MARK = self.mkA * L + self.mkB + rng.normal(0, self.mkS, n)
            CUT = self.ctA * L + self.ctB + rng.normal(0, self.ctS, n)
            # 길이가중 가법 + 하한 보장 (음수·0 방지, 합 정확히 보존)
            MARK = self._scale_to_sum(MARK, bMARK, L, self.mk_lo)
            CUT = self._scale_to_sum(CUT, bCUT, L, self.ct_lo)
        else:
            MARK = np.exp(self.mkA * np.log(L) + self.mkB + rng.normal(0, self.mkS, n))
            CUT = np.exp(self.ctA * np.log(L) + self.ctB + rng.normal(0, self.ctS, n))
            # 곱셈 스케일 후 하한 가법 보정 (실적 최소값 미만 방지, 합 보존)
            MARK = self._scale_to_sum(MARK * bMARK / MARK.sum(), bMARK, L, self.mk_lo)
            CUT = self._scale_to_sum(CUT * bCUT / CUT.sum(), bCUT, L, self.ct_lo)

        # 5) WO 베벨·부재·택트 (WO 직접 생성)
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
            PT = np.maximum(np.round(self.cPlin[0] + self.cPlin[1] * CUT + self.cPlin[2] * 0.7 * L + self.cPlin[3] * MARK + rng.normal(0, self.sPlin, n)), 1)
        TACT = TACT_CUT * CUT + TACT_MARK * MARK + TACT_THK * THK + TACT_PT * PT

        wodf = pd.DataFrame({'LTH': L, 'THK': THK, 'MARK_LTH': MARK, 'CUT_LTH': CUT,
                             'BVL_LTH': BVL, 'BV_QTY': BVQ, 'PTLST_QTY': PT, 'TACT_TIME': TACT})
        bl = {'LTH': L.max(), 'THK': THK.max(), 'MARK_LTH': MARK.sum(), 'CUT_LTH': CUT.sum(),
              'BVL_LTH': BVL.sum(), 'BV_QTY': BVQ.sum(), 'PTLST_QTY': PT.sum(),
              'STL_QTY': n, 'TACT_TIME': TACT.max()}
        return wodf, bl

    def generate(self, n_blocks=800, seed=2026):
        """합성 블록 n_blocks개 생성. 반환: (wo_df, blk_df). wo_df엔 BLK_ID 부여."""
        if not self._fitted:
            self.fit()
        rng = np.random.default_rng(seed)
        wos, bls = [], []
        for bid in range(n_blocks):
            w, b = self._gen_one(rng)
            w = w.copy()
            w.insert(0, 'BLK_ID', bid)
            wos.append(w)
            b = dict(b)
            b['BLK_ID'] = bid
            bls.append(b)
        wo_df = pd.concat(wos, ignore_index=True)
        blk_df = pd.DataFrame(bls)[['BLK_ID'] + BCOLS]
        return wo_df, blk_df

    # ================= 검증 =================
    def _actuals(self):
        act_wo = self.wo[WCOLS]
        act_blk = self.wo.groupby(['PROJ_NO', 'BLK_NO']).agg(
            LTH=('LTH', 'max'), THK=('THK', 'max'), MARK_LTH=('MARK_LTH', 'sum'),
            CUT_LTH=('CUT_LTH', 'sum'), BVL_LTH=('BVL_LTH', 'sum'), BV_QTY=('BV_QTY', 'sum'),
            PTLST_QTY=('PTLST_QTY', 'sum'), STL_QTY=('LTH', 'size'), TACT_TIME=('TACT_TIME', 'max')
        ).reset_index()
        return act_wo, act_blk

    def compare(self, wo_df, blk_df, method='spearman'):
        """생성 vs 실적 상관행렬 MAE(상삼각) 출력 및 반환."""
        act_wo, act_blk = self._actuals()
        cwa, cwg = act_wo[WCOLS].corr(method), wo_df[WCOLS].corr(method)
        cba, cbg = act_blk[BCOLS].corr(method), blk_df[BCOLS].corr(method)
        wmae = np.abs((cwg.values - cwa.values)[np.triu_indices(len(WCOLS), 1)]).mean()
        bmae = np.abs((cbg.values - cba.values)[np.triu_indices(len(BCOLS), 1)]).mean()
        print(f"[{method}] WO 상관행렬 MAE={wmae:.3f} / 블록 MAE={bmae:.3f}")
        return {'wo_mae': wmae, 'blk_mae': bmae,
                'wo_actual': cwa, 'wo_gen': cwg, 'blk_actual': cba, 'blk_gen': cbg}


def main():
    ap = argparse.ArgumentParser(description="조선소 블록·W/O 합성 데이터 생성")
    ap.add_argument('--wo_xlsx', help='실적 WO 엑셀 경로')
    ap.add_argument('--blk_xlsx', help='실적 블록 엑셀 경로')
    ap.add_argument('--mode', choices=['spearman', 'pearson'], default='spearman',
                    help="생성 모드: spearman(길이직접,기본) / pearson(선형식)")
    ap.add_argument('--n', type=int, default=800, help='생성 블록 수 (기본 800)')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', default='generated', help='출력 파일 접두사')
    args = ap.parse_args()

    gen = ShipyardGenerator(args.wo_xlsx, args.blk_xlsx, mode=args.mode).fit()
    wo_df, blk_df = gen.generate(n_blocks=args.n, seed=args.seed)
    wo_df.to_csv(f'{args.out}_wo.csv', index=False, encoding='utf-8-sig')
    blk_df.to_csv(f'{args.out}_blk.csv', index=False, encoding='utf-8-sig')
    print(f"[mode={args.mode}] 생성 완료: {args.out}_wo.csv ({len(wo_df)} WO), {args.out}_blk.csv ({len(blk_df)} 블록)")
    gen.compare(wo_df, blk_df, 'spearman')
    gen.compare(wo_df, blk_df, 'pearson')


if __name__ == '__main__':
    main()