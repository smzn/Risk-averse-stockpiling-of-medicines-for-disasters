# -*- coding: utf-8 -*-
"""
プール化再定式化の 2段階CVaR-LP を scipy 疎行列で構築する（Gurobi非依存）。
"""
import numpy as np
import scipy.sparse as sp


def build_pool_ef(d):
    """d: pl_params.ProblemData。プール化 Extensive-Form LP を返す。"""
    W = d.warehouses; N = d.nodes; S = d.scenarios
    K = d.products; Cs = d.components
    nW, nN, nS, nK, nC = len(W), len(N), len(S), len(K), len(Cs)

    wi = {w: i for i, w in enumerate(W)}
    ni = {n: i for i, n in enumerate(N)}
    ki = {k: i for i, k in enumerate(K)}
    ci = {c: i for i, c in enumerate(Cs)}
    si = {s: i for i, s in enumerate(S)}
    comp_idx_of_prod = np.array([ci[d.comp_of[k]] for k in K])   # 製品→成分index

    # 成分ペア（プール代替）
    P = d.comp_pairs                      # list (cd,cr,tier,pen)
    nP = len(P)
    pair_cd = np.array([ci[cd] for (cd, cr, t, pen) in P])
    pair_cr = np.array([ci[cr] for (cd, cr, t, pen) in P])
    pair_pen = np.array([pen for (cd, cr, t, pen) in P], dtype=float)
    pair_tier = np.array([t for (cd, cr, t, pen) in P], dtype=int)

    # ── 変数レイアウト（列オフセット）─────────────────────────
    off = {}
    col = 0
    off["s"] = col; col += nW * nK                 # s[w,k]
    off["eta"] = col; col += 1                      # eta
    off["x"] = []; off["g"] = []; off["u"] = []; off["q"] = []
    for sc in range(nS):
        off["x"].append(col); col += nW * nN * nK   # x[w,n,k]
        off["g"].append(col); col += nN * nP        # g[n,pair]
        off["u"].append(col); col += nN * nC        # u[n,c]
        off["q"].append(col); col += 1              # q
    ncol = col

    def s_col(w, k):    # scalar
        return off["s"] + wi[w] * nK + ki[k]
    # ベクトル化用インデックス関数
    def x_base(sc):
        return off["x"][sc]
    def g_base(sc):
        return off["g"][sc]
    def u_base(sc):
        return off["u"][sc]

    # ── 目的関数 c ─────────────────────────────────────────
    c = np.zeros(ncol)
    cost_arr = np.array([d.cost[k] for k in K])
    # s: cost[k]（全倉庫）
    for w in range(nW):
        c[off["s"] + w * nK: off["s"] + (w + 1) * nK] = cost_arr
    c[off["eta"]] = d.lam
    for sc, s in enumerate(S):
        c[off["q"][sc]] = d.lam * d.probs[s] / (1.0 - d.alpha)

    # ── 制約を COO で蓄積 ───────────────────────────────────
    ub_rows_i = []; ub_rows_j = []; ub_rows_v = []; b_ub = []
    eq_rows_i = []; eq_rows_j = []; eq_rows_v = []; b_eq = []

    def add_ub_block(rows, cols, vals, rhs_list):
        base = len(b_ub)
        ub_rows_i.append(rows + base); ub_rows_j.append(cols); ub_rows_v.append(vals)
        b_ub.extend(rhs_list)
        return base
    # ↑ rows は 0起点の局所行番号。base を足してグローバル化。

    # (1a) 倉庫の総容量（W 行, ≤）: Σ_k s[w,k] ≤ cap_wh
    #      κ=1 のときは Σ_k cap_k = cap_wh なので、この行は冗長（解に影響しない）。
    r0 = np.repeat(np.arange(nW), nK)
    c0 = off["s"] + (np.repeat(np.arange(nW), nK) * nK + np.tile(np.arange(nK), nW))
    add_ub_block(r0, c0, np.ones(nW * nK), [d.cap_wh] * nW)

    # (1b) 薬クラス単位の上限（W*C 行, ≤）: Σ_{k∈c} s[w,k] ≤ κ_c · cap_c
    #      κ_c=inf なら行を作らない。ATC では薬クラス=成分なので、この行が
    #      製品単位の上限と一致し、κ_p を緩めても解は変わらない。
    kap_c = float(getattr(d, "kappa_c", float("inf")))
    if np.isfinite(kap_c):
        rc = np.repeat(np.arange(nW), nK) * nC + np.tile(comp_idx_of_prod, nW)
        cc2 = off["s"] + (np.repeat(np.arange(nW), nK) * nK + np.tile(np.arange(nK), nW))
        rhs = [kap_c * d.cap_comp[Cs[ci_]] for _ in range(nW) for ci_ in range(nC)]
        add_ub_block(rc, cc2, np.ones(nW * nK), rhs)

    # (1) 予算（W 行, ≤）
    r = np.repeat(np.arange(nW), nK)
    cc = (off["s"] + (np.repeat(np.arange(nW), nK) * nK + np.tile(np.arange(nK), nW)))
    vv = np.tile(cost_arr, nW)
    add_ub_block(r, cc, vv, [d.budget_per_wh] * nW)

    # 容量は変数上限（後で ub に設定）。
    # 事前に成分別需要ベクトルを作る: Dc[sc][n,c]
    Dc = np.zeros((nS, nN, nC))
    for (n, cmp_, s), val in d.demand_ck.items():
        Dc[si[s], ni[n], ci[cmp_]] = val

    # シナリオ別ブロック
    for sc, s in enumerate(S):
        xb = x_base(sc); gb = g_base(sc); ub_ = u_base(sc)
        # --- 需要バランス（等式, N*C 行）---
        #   A[n,c] + Σ_cr g[n,c,cr] + u[n,c] − Σ_cd g[n,cd,c] = Dc
        # A[n,c] = Σ_{k∈c}Σ_w x[w,n,k]
        # x 係数（+1）を demand_balance 行 (n, comp(k)) に。
        base = len(b_eq)
        # x block: rows = n*nC + comp(k), cols = xb + (w*nN + n)*nK + k
        w_idx = np.repeat(np.arange(nW), nN * nK)
        n_idx = np.tile(np.repeat(np.arange(nN), nK), nW)
        k_idx = np.tile(np.arange(nK), nW * nN)
        rowsx = base + n_idx * nC + comp_idx_of_prod[k_idx]
        colsx = xb + (w_idx * nN + n_idx) * nK + k_idx
        eq_rows_i.append(rowsx); eq_rows_j.append(colsx); eq_rows_v.append(np.ones(len(colsx)))
        # g block: donor(+1) row(n,cd), receiver(−1) row(n,cr)
        gn = np.repeat(np.arange(nN), nP)
        gp = np.tile(np.arange(nP), nN)
        gcol = gb + gn * nP + gp
        # donor +1
        eq_rows_i.append(base + gn * nC + pair_cd[gp]); eq_rows_j.append(gcol); eq_rows_v.append(np.ones(len(gcol)))
        # receiver −1
        eq_rows_i.append(base + gn * nC + pair_cr[gp]); eq_rows_j.append(gcol); eq_rows_v.append(-np.ones(len(gcol)))
        # u block: +1 row(n,c)
        un = np.repeat(np.arange(nN), nC); uc = np.tile(np.arange(nC), nN)
        ucol = ub_ + un * nC + uc
        eq_rows_i.append(base + un * nC + uc); eq_rows_j.append(ucol); eq_rows_v.append(np.ones(len(ucol)))
        b_eq.extend(Dc[sc].reshape(-1).tolist())

        # --- 自己消費非負（≤, N*C 行）: Σ_cr g[n,c,cr] + u[n,c] ≤ Dc ---
        base = len(b_ub)
        eq2col = gb + gn * nP + gp
        ub_rows_i.append(base + gn * nC + pair_cd[gp]); ub_rows_j.append(eq2col); ub_rows_v.append(np.ones(len(eq2col)))
        ub_rows_i.append(base + un * nC + uc); ub_rows_j.append(ucol); ub_rows_v.append(np.ones(len(ucol)))
        b_ub.extend(Dc[sc].reshape(-1).tolist())

        # --- 倉庫在庫（≤, W*K 行）: Σ_n x[w,n,k] − s[w,k] ≤ 0 ---
        base = len(b_ub)
        rowswk = base + (w_idx * nK + k_idx)
        ub_rows_i.append(rowswk); ub_rows_j.append(colsx); ub_rows_v.append(np.ones(len(colsx)))
        # − s[w,k]
        swk_rows = base + np.arange(nW * nK)
        swk_cols = off["s"] + np.arange(nW * nK)
        ub_rows_i.append(swk_rows); ub_rows_j.append(swk_cols); ub_rows_v.append(-np.ones(nW * nK))
        b_ub.extend([0.0] * (nW * nK))

        # --- Tier2上限（≤, N*C 行）: Σ_cr g[n,c,cr] ≤ p_m·Dc ---
        base = len(b_ub)
        ub_rows_i.append(base + gn * nC + pair_cd[gp]); ub_rows_j.append(eq2col); ub_rows_v.append(np.ones(len(eq2col)))
        b_ub.extend((d.p_m * Dc[sc].reshape(-1)).tolist())

        # --- Tier2 のみの追加上限（任意, ≤, N*C 行）---
        #   Σ_{cr: tier=2} g[n,c,cr] ≤ cap_t2 · Dc[n,c,s]
        #   d.tier2_cap_extra が設定され、かつ p_m より小さいときだけ行を追加する
        #   （受入側の制約。Tier1 は上の総量上限 p_m のみが効く）
        cap_t2 = getattr(d, "tier2_cap_extra", None)
        if cap_t2 is not None and cap_t2 < d.p_m - 1e-12:
            sel = np.where(pair_tier == 2)[0]
            if len(sel):
                base = len(b_ub)
                gn2 = np.repeat(np.arange(nN), len(sel))
                gp2 = np.tile(sel, nN)
                gcol2 = gb + gn2 * nP + gp2
                ub_rows_i.append(base + gn2 * nC + pair_cd[gp2])
                ub_rows_j.append(gcol2)
                ub_rows_v.append(np.ones(len(gcol2)))
                b_ub.extend((cap_t2 * Dc[sc].reshape(-1)).tolist())

        # --- CVaR カット（≤, 1 行）: Σ ω u + Σ π g − eta − q ≤ 0 ---
        base = len(b_ub)
        omega_c = np.array([d.omega_comp[cc_] for cc_ in Cs])
        # u 係数 ω_c
        ub_rows_i.append(np.full(len(ucol), base)); ub_rows_j.append(ucol)
        ub_rows_v.append(np.tile(omega_c, nN))
        # g 係数 π
        ub_rows_i.append(np.full(len(gcol), base)); ub_rows_j.append(gcol)
        ub_rows_v.append(np.tile(pair_pen, nN))
        # − eta, − q
        ub_rows_i.append(np.array([base, base])); ub_rows_j.append(np.array([off["eta"], off["q"][sc]]))
        ub_rows_v.append(np.array([-1.0, -1.0]))
        b_ub.append(0.0)

    # ── 疎行列組み立て ─────────────────────────────────────
    def cat(parts):
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    A_ub = sp.csr_matrix((cat(ub_rows_v), (cat(ub_rows_i).astype(np.int64), cat(ub_rows_j).astype(np.int64))),
                         shape=(len(b_ub), ncol))
    A_eq = sp.csr_matrix((cat(eq_rows_v), (cat(eq_rows_i).astype(np.int64), cat(eq_rows_j).astype(np.int64))),
                         shape=(len(b_eq), ncol))

    # ── 変数上下限 ─────────────────────────────────────────
    lb = np.zeros(ncol); ub = np.full(ncol, np.inf)
    lb[off["eta"]] = -np.inf                      # eta 自由
    # s の上限: κ_p·cap_k。κ_p=inf のときは「成分の総需要」で頭打ちにする
    # （それ以上積んでも使い道がないので解には効かない。数値的な発散だけを防ぐ）
    kappa = float(getattr(d, "kappa_p", 1.0))
    cap_arr = np.array([d.cap[k] for k in K])
    if np.isfinite(kappa):
        s_ub = kappa * cap_arr
    else:
        comp_need = {}
        for k in K:
            comp_need[d.comp_of[k]] = comp_need.get(d.comp_of[k], 0.0) + d.cap[k] * nW
        s_ub = np.array([comp_need[d.comp_of[k]] for k in K])
    for w in range(nW):
        ub[off["s"] + w * nK: off["s"] + (w + 1) * nK] = s_ub

    return dict(c=c, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=A_eq, b_eq=np.array(b_eq),
                lb=lb, ub=ub, ncol=ncol, off=off,
                dims=dict(nW=nW, nN=nN, nS=nS, nK=nK, nC=nC, nP=nP),
                idx=dict(wi=wi, ni=ni, ki=ki, ci=ci, si=si),
                comp_pairs=P, products=K, components=Cs, nodes=N, warehouses=W,
                scenarios=S)


if __name__ == "__main__":
    import sys
    from . import pl_params, config as C
    seibun = C.SUBSET_SMALL if (len(sys.argv) > 1 and sys.argv[1] == "subset") else None
    d = pl_params.build(seibun)
    lp = build_pool_ef(d)
    print("cols=%d  ub_rows=%d  eq_rows=%d  A_ub_nnz=%d  A_eq_nnz=%d"
          % (lp["ncol"], lp["A_ub"].shape[0], lp["A_eq"].shape[0], lp["A_ub"].nnz, lp["A_eq"].nnz))
