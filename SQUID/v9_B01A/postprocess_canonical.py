# -*- coding: utf-8 -*-
"""保存済みの解を正準化して、報告用の CSV を作り直す。
"""
import os, sys, json, time, shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import pl_params, canonicalize as cz, config as C   # noqa: E402


def offsets(nW, nN, nS, nK, nC, nP):
    """pool_lp.build_pool_ef と同じ列レイアウトを、行列を作らずに再現する。"""
    off = {"s": 0}
    col = nW * nK
    off["eta"] = col; col += 1
    off["x"] = []; off["g"] = []; off["u"] = []; off["q"] = []
    for _ in range(nS):
        off["x"].append(col); col += nW * nN * nK
        off["g"].append(col); col += nN * nP
        off["u"].append(col); col += nN * nC
        off["q"].append(col); col += 1
    return off, col


def run(tag):
    rdir = os.path.join(C.RESULTS, tag)
    meta = json.load(open(os.path.join(rdir, "summary.json")))["meta"]
    print("=" * 70)
    print(" %s" % tag)
    # 旧版の solve.py は tier_policy を書いていないので、ペア数から判定する
    policy = meta.get("tier_policy")
    if policy is None:
        policy = ("tier0_tier1_only" if meta.get("n_comp_pairs", 0) < 992
                  else "tier0_tier1_tier2")
    print("   δ1=%.2f δ2=%.2f β=%.2f p_m=%.2f α=%.2f  tier=%s"
          % (meta["delta1"], meta["delta2"], meta["budget_rate"],
             meta["p_m"], meta["alpha"], policy))

    def _k(v, dflt):
        v = dflt if v is None else v
        return float("inf") if str(v).lower() in ("inf", "infinity") else float(v)

    # 旧版(phase1)の meta は単一 kappa。その場合は (κp, κc) = (kappa, inf) に読み替える。
    if "kappa_p" in meta:
        kp, kc = _k(meta.get("kappa_p"), 1.0), _k(meta.get("kappa_c"), float("inf"))
    else:
        kp, kc = _k(meta.get("kappa"), 1.0), float("inf")
    print("   κp=%s κc=%s" % (kp, kc))
    d = pl_params.build(None, gamma=meta["gamma"], r_high=meta["r_high"],
                        delta1=meta["delta1"], delta2=meta["delta2"],
                        budget_rate=meta["budget_rate"], p_m=meta["p_m"],
                        alpha=meta["alpha"], kappa_p=kp, kappa_c=kc)
    if policy == "tier0_tier1_only":
        d.comp_pairs = [x for x in d.comp_pairs if x[2] == 1]

    W, N, S, K, Cs = d.warehouses, d.nodes, d.scenarios, d.products, d.components
    nW, nN, nS, nK, nC = len(W), len(N), len(S), len(K), len(Cs)
    P = d.comp_pairs; nP = len(P)
    ni = {n: i for i, n in enumerate(N)}; ki = {k: i for i, k in enumerate(K)}
    ci = {c: i for i, c in enumerate(Cs)}; si = {s: i for i, s in enumerate(S)}
    comp_of_prod = np.array([ci[d.comp_of[k]] for k in K])
    pair_cd = np.array([ci[a] for (a, b, t, p) in P])     # 受け取る側
    pair_cr = np.array([ci[b] for (a, b, t, p) in P])     # 供出する側

    off, ncol = offsets(nW, nN, nS, nK, nC, nP)
    z = np.load(os.path.join(rdir, "solution_z.npz"))["z"]
    assert len(z) == ncol, "列数が合わない: %d vs %d" % (len(z), ncol)

    Dc = np.zeros((nS, nN, nC))
    for (n, c, s), v in d.demand_ck.items():
        Dc[si[s], ni[n], ci[c]] = v
    Dp = np.zeros((nS, nN, nK))
    for (n, k, s), v in d.demand_pk.items():
        if k in ki and n in ni:
            Dp[si[s], ni[n], ki[k]] = v
    S_k = z[off["s"]: off["s"] + nW * nK].reshape(nW, nK).sum(axis=0)
    probs = np.array([d.probs[s] for s in S])
    omega = np.array([d.omega_comp[c] for c in Cs])

    unmet_rows, t0_rows, t0p_rows = [], [], []
    rep = dict(tier0_before=0.0, tier0_after=0.0,
               E_unmet_before=0.0, E_unmet_after=0.0, checks=[])

    for sc, sname in enumerate(S):
        x = z[off["x"][sc]: off["x"][sc] + nW * nN * nK].reshape(nW, nN, nK)
        X0 = x.sum(axis=0)                                    # 元の製品別配送量
        g = z[off["g"][sc]: off["g"][sc] + nN * nP].reshape(nN, nP)
        u = z[off["u"][sc]: off["u"][sc] + nN * nC].reshape(nN, nC).copy()
        u = np.maximum(u, 0.0)

        # 代替の受入 R[n,c] と供出 Dout[n,c]
        R = np.zeros((nN, nC)); np.add.at(R.T, pair_cd, g.T)
        Dout = np.zeros((nN, nC)); np.add.at(Dout.T, pair_cr, g.T)

        # ── (1) 未充足の公平化（成分ごとの合計は保存）──────────
        u_new = cz.fair_unmet(u, R, Dc[sc])
        d_tot = np.abs(u_new.sum(axis=0) - u.sum(axis=0)).max()
        rep["checks"].append("%s 未充足の成分別合計のずれ: %.3e" % (sname, d_tot))

        # 自成分で満たす分（own）と他成分への供出分（Dout）に分ける。
        # Tier0 は own の内訳の問題であり、供出用の在庫確保は Tier0 ではない。
        own = Dc[sc] - R - u_new
        assert own.min() > -1e-6, "own が負: %.3e" % own.min()
        own = np.maximum(own, 0.0)

        # ── (2) Tier0 を最小付け替え量として計算 ──────────────
        t0_before = float(np.maximum(0.0, X0 - Dp[sc]).sum())
        t0_after = 0.0
        for c in range(nC):
            ks = np.where(comp_of_prod == c)[0]
            if not len(ks):
                continue
            v, Xc = cz.min_tier0_component(own[:, c], Dp[sc][:, ks], S_k[ks],
                                           donate=float(Dout[:, c].sum()))
            t0_after += v
            t0_rows.append(dict(scenario=sname, component=Cs[c], tier0_amount=v))
            # 製品×製品の内訳（余剰→不足の按分。総量は上の v で一意）
            sur = np.maximum(0.0, Xc - Dp[sc][:, ks])   # 付け替えの供給側
            dfc = np.maximum(0.0, Dp[sc][:, ks] - Xc)
            tot_d = dfc.sum(axis=1)
            act = np.where((tot_d > 1e-9) & (sur.sum(axis=1) > 1e-9))[0]
            if len(act):
                flow = (sur[act].T @ (dfc[act] / tot_d[act][:, None]))
                for i, a in enumerate(ks):
                    for j, b in enumerate(ks):
                        if flow[i, j] > 1e-9:
                            t0p_rows.append(dict(scenario=sname, component=Cs[c],
                                                 donor_product=K[a], receiver_product=K[b],
                                                 amount=flow[i, j]))
        rep["tier0_before"] += probs[sc] * t0_before
        rep["tier0_after"] += probs[sc] * t0_after
        rep["E_unmet_before"] += probs[sc] * float(u.sum())
        rep["E_unmet_after"] += probs[sc] * float(u_new.sum())

        # 目的値（第2段階コスト）が変わっていないことを確認
        pen = np.array([p for (a, b, t, p) in P])
        f_before = float((u * omega).sum() + (g * pen).sum())
        f_after = float((u_new * omega).sum() + (g * pen).sum())
        rep["checks"].append("%s 第2段階コスト: %.6e → %.6e (差 %.3e)"
                             % (sname, f_before, f_after, abs(f_after - f_before)))

        nz = np.where(u_new > 1e-9)
        for a, b in zip(*nz):
            unmet_rows.append(dict(scenario=sname, node=N[a], component=Cs[b],
                                   unmet=u_new[a, b]))
        print("   %s  Tier0 %11.0f → %10.0f   未充足 %11.0f（不変）"
              % (sname, t0_before, t0_after, u_new.sum()))

    odir = os.path.join(C.RESULTS, tag + "_canon")
    os.makedirs(odir, exist_ok=True)
    pd.DataFrame(unmet_rows).to_csv(os.path.join(odir, "unmet_demand.csv"), index=False)
    pd.DataFrame(t0_rows).to_csv(os.path.join(odir, "tier0_by_component.csv"), index=False)
    if t0p_rows:
        # シナリオ別・重みなしで保存する。どの重み（確率加重か CVaR 加重か）で
        # 集計するかは読み手側の判断なので、ここでは掛けない。
        # tier0_by_component.csv / substitutions.csv と同じ形式に揃えてある。
        pf = (pd.DataFrame(t0p_rows)
                .groupby(["scenario", "component", "donor_product", "receiver_product"],
                         as_index=False).amount.sum()
                .sort_values(["scenario", "amount"], ascending=[True, False]))
        pf.to_csv(os.path.join(odir, "tier0_by_product.csv"), index=False)
    for f in ["substitutions.csv", "storage_plan.csv", "summary.json"]:
        src = os.path.join(rdir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(odir, f))
    rep["tag"] = tag
    json.dump(rep, open(os.path.join(odir, "canonical_summary.json"), "w"),
              ensure_ascii=False, indent=2)

    print("   ── Tier0 合計（確率加重）: %.0f → %.0f  （%.1f%% が任意性だった）"
          % (rep["tier0_before"], rep["tier0_after"],
             100 * (1 - rep["tier0_after"] / rep["tier0_before"])))
    for c in rep["checks"]:
        print("   検証: %s" % c)
    print("   → %s" % odir)
    return rep


if __name__ == "__main__":
    tags = sys.argv[1:] or ["v8_pool_baseline_t0", "v8_pool_no_tier2_t0"]
    t0 = time.time()
    out = [run(t) for t in tags]
    print("\n完了 (%.1f 分)" % ((time.time() - t0) / 60))
