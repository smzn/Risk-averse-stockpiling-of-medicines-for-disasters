# -*- coding: utf-8 -*-
"""
PLモデル（製品/class_id粒度）の全パラメータ・需要を、Gurobi非依存で構築する。
実データ/v4/config/real_paramsv8.py と real_paramsv7.py のロジックを忠実に移植。

返す ProblemData には、モデル構築（プール化・フルペア両対応）に必要な
すべての集合・係数・需要が numpy/dict で入る。
"""
import os
import itertools
import numpy as np
import pandas as pd

from . import config as C


# ────────────────────────────────────────────────────────────
# v7 ヘルパの移植
# ────────────────────────────────────────────────────────────
def _load_dose_mg(df, override):
    """1錠あたりの薬量(mg)。単剤コードに紛れた配合剤は指標成分量で補完。"""
    def to_mg(row):
        dose = row["1錠あたりの薬量"]
        unit = str(row["1錠あたりの薬量単位"]).lower()
        if pd.isna(dose):
            return None
        if "μg" in unit or "ug" in unit:
            return float(dose) / 1000.0
        return float(dose)
    dose_mg = df.apply(to_mg, axis=1)
    ov = dict(zip(override["class_id"], override["指標成分量mg"]))
    for i, cid in df["class_id"].items():
        if cid in ov:
            dose_mg.at[i] = float(ov[cid])
    return dose_mg


def _compute_cbar(data_dir):
    """c̄ = 全32コードの需要加重平均薬価（円/患者・日）。固定スカラー。"""
    df = pd.read_csv(os.path.join(data_dir, "drug_usage_df.csv"))
    rdd = pd.read_csv(os.path.join(data_dir, "rdd_table.csv")).set_index("ATCcode")
    override = pd.read_csv(os.path.join(data_dir, "dose_mg_override.csv"))
    df["dose_mg"] = _load_dose_mg(df, override)
    bd = (pd.read_csv(os.path.join(data_dir, "baseline_demand.csv"))
            .groupby("class_id")["baseline_demand_mg"].sum())
    df["tabs"] = df["class_id"].map(bd).fillna(0.0)
    total_yen, total_pd = 0.0, 0.0
    for atc, g in df.groupby("ATCcode"):
        v = float(rdd.loc[atc, "RDD_value"]); u = rdd.loc[atc, "RDD_unit"]
        if u == "mg":
            pdays = (g["tabs"] * g["dose_mg"]).sum() / v
        else:
            pdays = g["tabs"].sum() / v
        total_yen += (g["薬価"] * g["tabs"]).sum()
        total_pd += pdays
    return total_yen / total_pd


class ProblemData(object):
    """モデル構築に必要な全データの入れ物。"""
    pass


def build(seibun=None, data_dir=None,
          gamma=C.GAMMA, r_high=C.R_HIGH, delta1=C.DELTA1, delta2=C.DELTA2,
          budget_rate=C.BUDGET_RATE, p_m=C.P_M, alpha=C.ALPHA, rho=C.RHO, lam=C.LAM,
          kappa_p=getattr(C, "KAPPA_P", 1.0),
          kappa_c=getattr(C, "KAPPA_C", float("inf"))):
    """
    seibun: 対象 ATCcode のリスト（None なら全32）。
    戻り値: ProblemData（products, components, nodes, scenarios, warehouses,
            demand, cost, omega, capacity, budget, tier pairs & penalties, ...）
    """
    data_dir = data_dir or C.DATA
    seibun = list(seibun) if seibun else list(C.ALL_32)

    # ── 薬剤マスタ ───────────────────────────────────────────
    df = pd.read_csv(os.path.join(data_dir, "drug_usage_df.csv"))
    df = df[df["ATCcode"].isin(seibun)].copy()
    rdd = pd.read_csv(os.path.join(data_dir, "rdd_table.csv")).set_index("ATCcode")
    override = pd.read_csv(os.path.join(data_dir, "dose_mg_override.csv"))
    df["dose_mg"] = _load_dose_mg(df, override)

    products = sorted(df["class_id"].unique().tolist())          # 製品 k
    cid_atc = df.set_index("class_id")["ATCcode"].to_dict()       # 成分(=ATCcode=component c)
    cid_dose = df.set_index("class_id")["dose_mg"].to_dict()
    cid_price = df.set_index("class_id")["薬価"].to_dict()
    cid_comp = df.set_index("class_id")["成分名"].to_dict()

    # 患者・日変換係数 pd_per_tab[k]
    pd_per_tab = {}
    for cid in products:
        atc = cid_atc[cid]; v = float(rdd.loc[atc, "RDD_value"]); u = rdd.loc[atc, "RDD_unit"]
        if u == "mg":
            dm = cid_dose[cid]
            pd_per_tab[cid] = (dm / v) if pd.notna(dm) else None
        else:
            pd_per_tab[cid] = 1.0 / v

    # 製品別 備蓄コスト c_k（円/患者・日）
    cost = {}
    for cid in products:
        ppt = pd_per_tab.get(cid)
        cost[cid] = (float(cid_price[cid]) / ppt) if (ppt and ppt > 0) else float(cid_price[cid])

    # 未充足単価 ω_k = γ·c̄（高リスクは r·γ·c̄）
    cbar = _compute_cbar(data_dir)
    omega_base = gamma * cbar
    omega = {cid: (r_high * omega_base if cid_comp.get(cid) in C.HIGH_RISK_COMPS else omega_base)
             for cid in products}

    # 倉庫
    wh_df = pd.read_csv(os.path.join(data_dir, "warehouses.csv"))
    warehouses = wh_df["warehouse_id"].tolist()
    n_wh = len(warehouses)

    # 容量・予算
    stock_days = 7
    annual_pd = {}
    for cid in products:
        shizu = df.loc[df["class_id"] == cid, "静岡県"]
        tabs = float(shizu.iloc[0]) if len(shizu) and pd.notna(shizu.iloc[0]) else 0.0
        annual_pd[cid] = tabs * (pd_per_tab.get(cid) or 0.0)
    target_stock_total = {cid: annual_pd[cid] * (stock_days / 365.0) for cid in products}
    # ── 容量制約（詳細は config.KAPPA_P / KAPPA_C のコメント）──────
    #   Σ_k s[w,k]     ≤ cap_wh        倉庫の総容量
    #   Σ_{k∈c} s[w,k] ≤ κ_c · cap_c   薬クラス単位の調達可能上限
    #   s[w,k]         ≤ κ_p · cap_k   製品単位の調達可能上限
    # 総容量は κ によらず一定なので、κ が変えるのは「中身の自由度」だけになる。
    cap = {cid: target_stock_total[cid] / n_wh for cid in products}
    cap_wh = sum(target_stock_total.values()) / n_wh
    # cap_comp（成分別容量）は comp_of の定義後に作る
    total_value = sum(cost[cid] * target_stock_total[cid] for cid in products)
    budget_per_wh = total_value * budget_rate / n_wh

    # ── Tier別ペナルティ（π = δ·γ·c̄）─────────────────────────
    pi0 = 0.0
    pi1 = delta1 * omega_base
    pi2 = delta2 * omega_base

    # 成分(component)レベルの L3/L4
    comp_l3 = {c: c[:4] for c in set(cid_atc.values())}
    comp_l4 = {c: c[:5] for c in set(cid_atc.values())}
    components = sorted(set(cid_atc.values()))
    comp_of = cid_atc                                        # product -> component

    # 製品ペア（フルペア用）と成分ペア（プール用）の tier/penalty
    # tier: 0=同一成分, 1=同L4異成分, 2=同L3異L4
    def pair_tier(a_comp, b_comp):
        if a_comp == b_comp:
            return 0
        if comp_l4[a_comp] == comp_l4[b_comp]:
            return 1
        if comp_l3[a_comp] == comp_l3[b_comp]:
            return 2
        return -1
    pi_by_tier = {0: pi0, 1: pi1, 2: pi2}

    # 成分ペア（Tier1/2 のみ。Tier0 はプール内で吸収）
    comp_pairs = []          # list of (cd, cr, tier, penalty)
    for cd, cr in itertools.permutations(components, 2):
        t = pair_tier(cd, cr)
        if t in (1, 2):
            comp_pairs.append((cd, cr, t, pi_by_tier[t]))

    # 製品ペア（フルペア。Tier0/1/2 すべて）
    prod_pairs = []          # list of (a, b, tier, penalty)
    for a, b in itertools.permutations(products, 2):
        t = pair_tier(comp_of[a], comp_of[b])
        if t in (0, 1, 2):
            prod_pairs.append((a, b, t, pi_by_tier[t]))

    # ── 需要（demand.csv：錠数 → 患者・日）───────────────────
    dem = pd.read_csv(os.path.join(data_dir, "demand.csv"))
    dem = dem[dem["class_id"].isin(products)].copy()
    dem["demand_pd"] = dem["demand_mg"] * dem["class_id"].map(pd_per_tab).astype(float)

    scen_df = pd.read_csv(os.path.join(data_dir, "scenarios.csv"))
    scenarios = scen_df["scenario_id"].tolist()
    probs = dict(zip(scen_df["scenario_id"], scen_df["prob"]))

    nodes_df = pd.read_csv(os.path.join(data_dir, "nodes.csv"))
    nodes = nodes_df["node_id"].tolist()

    # 製品別需要 D[(n,k,s)]、成分別需要 Dc[(n,c,s)]（プール用）
    demand_pk = {}
    for r in dem.itertuples(index=False):
        demand_pk[(r.node_id, r.class_id, r.scenario_id)] = float(r.demand_pd)
    dem["comp"] = dem["class_id"].map(comp_of)
    dc = dem.groupby(["node_id", "comp", "scenario_id"])["demand_pd"].sum()
    demand_ck = {(n, c, s): float(v) for (n, c, s), v in dc.items()}

    d = ProblemData()
    d.products = products
    d.components = components
    d.comp_of = comp_of
    d.nodes = nodes
    d.warehouses = warehouses
    d.scenarios = scenarios
    d.probs = probs
    d.cost = cost                    # 製品別備蓄コスト
    d.omega = omega                  # 製品別未充足単価
    d.omega_comp = {c: (r_high * omega_base
                        if any(cid_comp.get(k) in C.HIGH_RISK_COMPS
                               for k in products if comp_of[k] == c)
                        else omega_base)
                    for c in components}
    d.cap = cap                      # 製品別・倉庫あたり容量
    d.cap_wh = cap_wh                # 倉庫あたり総容量
    cap_comp = {}
    for cid in products:
        cap_comp[comp_of[cid]] = cap_comp.get(comp_of[cid], 0.0) + cap[cid]
    d.cap_comp = cap_comp            # 成分別・倉庫あたり容量
    d.kappa_p = kappa_p              # 製品単位の調達可能倍率
    d.kappa_c = kappa_c              # 薬クラス単位の調達可能倍率
    d.budget_per_wh = budget_per_wh
    d.p_m = p_m
    d.alpha = alpha
    d.lam = lam
    d.rho = rho
    d.cbar = cbar
    d.omega_base = omega_base
    d.pi = pi_by_tier
    d.comp_pairs = comp_pairs        # プール化：成分ペア
    d.prod_pairs = prod_pairs        # フルペア：製品ペア
    d.demand_pk = demand_pk          # 製品別需要
    d.demand_ck = demand_ck          # 成分別需要（プール用）
    d.meta = dict(gamma=gamma, r_high=r_high, delta1=delta1, delta2=delta2,
                  budget_rate=budget_rate, p_m=p_m, alpha=alpha, cbar=cbar,
                  pi0=pi0, pi1=pi1, pi2=pi2,
                  n_products=len(products), n_components=len(components),
                  n_nodes=len(nodes), n_comp_pairs=len(comp_pairs),
                  n_prod_pairs=len(prod_pairs))
    return d


if __name__ == "__main__":
    import sys
    seibun = None
    if len(sys.argv) > 1 and sys.argv[1] == "subset":
        seibun = C.SUBSET_SMALL
    d = build(seibun)
    print("=== ProblemData meta ===")
    for k, v in d.meta.items():
        print("  %-16s %s" % (k, v))
    print("nodes=%d scenarios=%d warehouses=%d" % (len(d.nodes), len(d.scenarios), len(d.warehouses)))
