# -*- coding: utf-8 -*-
"""プール化LPの最適解 z から指標・テーブルを取り出しフォルダ保存する。"""
import os
import json
import numpy as np
import pandas as pd



def extract_tier0(z, lp, d, outdir, save_products=True):
    """
    プール定式化で暗黙に吸収されている Tier0（同一成分の製品間代替）を解から復元する。

        surplus[n,k,s] = max(0, X[n,k,s] − D[n,k,s])   … 余った製品（供給側）
        deficit[n,k,s] = max(0, D[n,k,s] − X[n,k,s])   … 足りない製品（需要側）
        Tier0[n,c,s]   = Σ_{k∈c} surplus[n,k,s]
        X[n,k,s] = Σ_w x[w,n,k,s]

    製品→製品の内訳（save_products=True）:
      プール内の donor→receiver の対応づけは一意に定まらない（縮退した輸送問題）ため、
      各拠点・成分内で **余剰に比例した按分** という標準的な正準解を採る:
          flow[k→k'] = Σ_{n,s} p_s · surplus[n,k,s] · deficit[n,k',s] / Σ_{k''} deficit[n,k'',s]
      これは「どの余剰製品も、不足している製品へ同じ割合で配分される」という仮定に対応する。

    出力:
      tier0_by_component.csv : scenario, component, tier0_amount
      tier0_by_product.csv   : component, donor_product, receiver_product, amount（期待値）
    戻り値: Tier0 の期待値合計
    """
    off = lp["off"]; dims = lp["dims"]
    nW, nN, nS, nK, nC = (dims[k] for k in ["nW", "nN", "nS", "nK", "nC"])
    N, K, Cs, S = lp["nodes"], lp["products"], lp["components"], lp["scenarios"]
    ni = {n: i for i, n in enumerate(N)}
    ki = {k: i for i, k in enumerate(K)}
    ci = {c: i for i, c in enumerate(Cs)}
    si = {s: i for i, s in enumerate(S)}
    comp_idx = np.array([ci[d.comp_of[k]] for k in K])
    probs = [d.probs[s] for s in S]

    D = np.zeros((nS, nN, nK))
    for (n, k, s), v in d.demand_pk.items():
        if n in ni and k in ki and s in si:
            D[si[s], ni[n], ki[k]] = v

    rows = []
    tier0_total = 0.0
    # 製品ペア別フロー（成分内のみ）。キー: (component, donor, receiver)
    pair_flow = {}

    for sc, ss in enumerate(S):
        x = z[off["x"][sc]: off["x"][sc] + nW * nN * nK].reshape(nW, nN, nK)
        X = x.sum(axis=0)
        surplus = np.maximum(0.0, X - D[sc])
        deficit = np.maximum(0.0, D[sc] - X)

        for cci, cc in enumerate(Cs):
            cols = np.where(comp_idx == cci)[0]
            if len(cols) <= 1:
                continue                      # 製品が1つなら Tier0 は起こりえない
            sur_c = surplus[:, cols]          # [nN, n_k]
            amt = float(sur_c.sum())
            if amt <= 1e-6:
                continue
            rows.append(dict(scenario=ss, component=cc, tier0_amount=amt))
            tier0_total += amt * probs[sc]

            if not save_products:
                continue
            def_c = deficit[:, cols]
            tot_def = def_c.sum(axis=1)                     # 拠点ごとの不足合計
            act = np.where(tot_def > 1e-9)[0]               # 不足がある拠点のみ
            if len(act) == 0:
                continue
            # 余剰×不足シェア の外積を拠点方向に集約（比例按分）
            share = def_c[act] / tot_def[act][:, None]      # [n_act, n_k]
            F = sur_c[act].T.dot(share)                     # [n_k(donor), n_k(receiver)]
            np.fill_diagonal(F, 0.0)                        # 自分自身への代替は数えない
            for a, kd in enumerate(cols):
                nz = np.where(F[a] > 1e-6)[0]
                for b in nz:
                    key = (cc, K[kd], K[cols[b]])
                    pair_flow[key] = pair_flow.get(key, 0.0) + float(F[a, b]) * probs[sc]

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "tier0_by_component.csv"), index=False)
    if save_products:
        pf = pd.DataFrame(
            [dict(component=c, donor_product=a, receiver_product=b, amount=v)
             for (c, a, b), v in sorted(pair_flow.items(), key=lambda x: -x[1])])
        pf.to_csv(os.path.join(outdir, "tier0_by_product.csv"), index=False)
    return tier0_total


def extract_and_save(z, lp, d, outdir, method, solve_info=None):
    """
    z: 最適解ベクトル（長さ ncol）
    lp: build_pool_ef の戻り値
    d: ProblemData
    outdir: 保存先フォルダ（作成される）
    method: 手法名（文字列）
    solve_info: dict（obj値・時間・メモリなど任意）
    """
    os.makedirs(outdir, exist_ok=True)
    off = lp["off"]; dims = lp["dims"]
    nW, nN, nS, nK, nC, nP = (dims[k] for k in ["nW", "nN", "nS", "nK", "nC", "nP"])
    W, N, K, Cs, S = lp["warehouses"], lp["nodes"], lp["products"], lp["components"], lp["scenarios"]
    P = lp["comp_pairs"]
    probs = [d.probs[s] for s in S]

    # ── 第1段階 s[w,k] ──
    s = z[off["s"]: off["s"] + nW * nK].reshape(nW, nK)
    storage_rows = []
    for wi_, w in enumerate(W):
        for ki_, k in enumerate(K):
            if s[wi_, ki_] > 1e-9:
                storage_rows.append(dict(warehouse=w, product=k, component=d.comp_of[k],
                                         storage=float(s[wi_, ki_])))
    pd.DataFrame(storage_rows).to_csv(os.path.join(outdir, "storage_plan.csv"), index=False)

    # ── 目的値・CVaR の再計算（検算）──
    cost_arr = np.array([d.cost[k] for k in K])
    storage_cost = float((s * cost_arr[None, :]).sum())
    eta = float(z[off["eta"]])

    # 各シナリオの u[n,c], g[n,pair], q, 第2段階コスト, tier別代替
    unmet_rows = []; subst_rows = []
    E_unmet = 0.0; E_unmet_hi = 0.0
    tier1 = 0.0; tier2 = 0.0
    scen_cost = {}
    hi_comps = set(c for c in Cs if d.omega_comp[c] > d.omega_base + 1e-6)
    pair_tier = np.array([t for (cd, cr, t, pen) in P])
    pair_pen = np.array([pen for (cd, cr, t, pen) in P])
    for sc, ss in enumerate(S):
        u = z[off["u"][sc]: off["u"][sc] + nN * nC].reshape(nN, nC)
        g = z[off["g"][sc]: off["g"][sc] + nN * nP].reshape(nN, nP)
        omega_c = np.array([d.omega_comp[c] for c in Cs])
        f_unmet = float((u * omega_c[None, :]).sum())
        f_sub = float((g * pair_pen[None, :]).sum())
        scen_cost[ss] = f_unmet + f_sub
        # 期待未充足（患者・日）
        Eu = float(u.sum()) * probs[sc]
        E_unmet += Eu
        for cci, cc in enumerate(Cs):
            col = u[:, cci]
            if col.sum() > 1e-6:
                for nni, nn in enumerate(N):
                    if col[nni] > 1e-9:
                        unmet_rows.append(dict(scenario=ss, node=nn, component=cc, unmet=float(col[nni])))
        E_unmet_hi += float(u[:, [i for i, c in enumerate(Cs) if c in hi_comps]].sum()) * probs[sc]
        # tier別代替（期待値）
        t1 = float(g[:, pair_tier == 1].sum()) * probs[sc]
        t2 = float(g[:, pair_tier == 2].sum()) * probs[sc]
        tier1 += t1; tier2 += t2
        # 代替フロー（成分ペア）保存
        gsum = g.sum(axis=0)   # ペア別合計（全ノード）
        # 【向きに注意】comp_pairs の (cd, cr) は pool_lp の需要バランスで
        #   cd に +1（＝代替を受け取る側）、cr に −1（＝在庫を供出する側）が立つ。
        #   したがって donor=cr, receiver=cd である（名前と逆なので取り違えないこと）。
        for pi_, (cd, cr, t, pen) in enumerate(P):
            if gsum[pi_] > 1e-6:
                subst_rows.append(dict(scenario=ss, donor=cr, receiver=cd, tier=t,
                                       amount_allnodes=float(gsum[pi_])))
    pd.DataFrame(unmet_rows).to_csv(os.path.join(outdir, "unmet_demand.csv"), index=False)
    pd.DataFrame(subst_rows).to_csv(os.path.join(outdir, "substitutions.csv"), index=False)

    cvar = eta + (1.0 / (1.0 - d.alpha)) * sum(probs[sc] * float(z[off["q"][sc]]) for sc in range(nS))
    obj = storage_cost + d.lam * cvar

    # Tier0（同一成分の製品間代替）はプール内で吸収され明示変数を持たないため解から復元する
    tier0 = extract_tier0(z, lp, d, outdir)

    # 解ベクトルを圧縮保存しておく（再求解せずに任意の後解析ができるようにするため）
    try:
        np.savez_compressed(os.path.join(outdir, "solution_z.npz"), z=z)
    except Exception as e:
        print("  [warn] solution_z.npz の保存に失敗: %s" % e)

    summary = dict(
        method=method,
        objective=obj, storage_cost=storage_cost, cvar=cvar, eta=eta,
        E_unmet=E_unmet, E_unmet_high=E_unmet_hi,
        tier0_subst=tier0, tier1_subst=tier1, tier2_subst=tier2,
        scenario_cost=scen_cost,
        meta=d.meta,
    )
    if solve_info:
        summary["solve_info"] = solve_info
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 人間可読サマリ
    with open(os.path.join(outdir, "run_summary.txt"), "w") as f:
        f.write("method       : %s\n" % method)
        f.write("objective    : %.0f\n" % obj)
        f.write("storage_cost : %.0f\n" % storage_cost)
        f.write("CVaR         : %.0f\n" % cvar)
        f.write("E[unmet]     : %.0f  (high-risk %.0f)\n" % (E_unmet, E_unmet_hi))
        f.write("Tier0 subst  : %.0f  (reconstructed)\n" % tier0)
        f.write("Tier1 subst  : %.0f\n" % tier1)
        f.write("Tier2 subst  : %.0f\n" % tier2)
        if solve_info:
            for k, v in solve_info.items():
                f.write("%-12s : %s\n" % (k, v))
    print("[save] %s  obj=%.3e cvar=%.3e E=%.0f T0=%.0f T1=%.0f T2=%.0f -> %s"
          % (method, obj, cvar, E_unmet, tier0, tier1, tier2, outdir))
    return summary
