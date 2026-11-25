# -*- coding: utf-8 -*-
"""v9 の汎用ドライバ。製品レベル（PL）モデルの 1構成だけを解く。

  pl … 薬クラス = 製品(class_id)。成分内の製品選択ができる

容量制約:
  Σ_k s[w,k]     ≤ Cap_w         倉庫の総容量
  Σ_{k∈c} s[w,k] ≤ κ_c · cap_c    成分単位の調達可能上限
  s[w,k]         ≤ κ_p · cap_k    製品単位の調達可能上限
  κ_p=1   … 各製品を自分の7日分までしか置けない（Tier0 は構造的に発生しない）
  κ_p=inf … 成分内で安価な製品に寄せられる

代替の許可:
  --tier2 on   … Tier0 + Tier1 + Tier2
  --tier2 off  … Tier0 + Tier1
  --tier2-cap  … Tier2 の受入上限を p_m と別に設定
"""
import os
import sys
import csv
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from common import pl_params, pool_lp, save, config as C   # noqa: E402
from common.solvers import solve_highs                                 # noqa: E402

FIELDS = ["tag", "granularity", "axis", "tier2", "kappa_p", "kappa_c",
          "budget_rate", "r_high", "alpha", "p_m", "delta1", "delta2", "tier2_cap",
          "status", "obj", "storage_cost", "cvar", "E_unmet", "E_unmet_high",
          "total_stock", "mean_unit_price",
          "tier0_subst", "tier1_subst", "tier2_subst",
          "S1_cost", "S2_cost", "S3_cost", "walltime"]


def append_row(csv_path, row):
    """複数ジョブが同時に書いても壊れないよう排他ロックして追記する。"""
    import fcntl
    newfile = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if newfile:
            w.writeheader()
        w.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--granularity", choices=["pl"], default="pl")
    ap.add_argument("--axis", default="")
    ap.add_argument("--tier2", choices=["on", "off"], default="on")
    ap.add_argument("--kappa-p", default="1.0",
                    help="製品(SKU)単位の調達可能倍率。'inf' で製品別上限なし")
    ap.add_argument("--kappa-c", default="inf",
                    help="薬クラス単位の調達可能倍率。'inf' でクラス別上限なし")
    ap.add_argument("--gamma", type=float, default=C.GAMMA)
    ap.add_argument("--r-high", type=float, default=C.R_HIGH)
    ap.add_argument("--delta1", type=float, default=C.DELTA1)
    ap.add_argument("--delta2", type=float, default=C.DELTA2)
    ap.add_argument("--budget-rate", type=float, default=C.BUDGET_RATE)
    ap.add_argument("--p-m", type=float, default=C.P_M)
    ap.add_argument("--alpha", type=float, default=C.ALPHA)
    ap.add_argument("--tier2-cap", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--method", default="ipm", choices=["ipm", "simplex", "choose"])
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--no-crossover", action="store_true",
                    help="crossover を行わない。第2段階の配分が解法依存になるので常用しないこと")
    ap.add_argument("--presolve", default="on", choices=["on", "off"])
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    def _k(v):
        return float("inf") if str(v).lower() in ("inf", "infinity") else float(v)
    kappa_p, kappa_c = _k(args.kappa_p), _k(args.kappa_c)

    outdir = os.path.join(C.RESULTS, args.tag)
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(C.RESULTS, "v9_results.csv")

    t_start = time.time()
    print("=" * 70, flush=True)
    print(" 構成: %s   [%s]" % (args.tag, args.granularity.upper()), flush=True)
    print("  κp=%s κc=%s tier2=%s β=%.2f r=%.1f α=%.2f p_m=%.2f δ1=%.3f δ2=%.3f cap_t2=%s"
          % (args.kappa_p, args.kappa_c,
             args.tier2, args.budget_rate, args.r_high, args.alpha,
             args.p_m, args.delta1, args.delta2,
             "-" if args.tier2_cap is None else "%.2f" % args.tier2_cap), flush=True)
    print("=" * 70, flush=True)

    seibun = C.SUBSET_SMALL if args.subset else None
    d = pl_params.build(seibun, gamma=args.gamma, r_high=args.r_high,
                        delta1=args.delta1, delta2=args.delta2,
                        budget_rate=args.budget_rate, p_m=args.p_m,
                        alpha=args.alpha, kappa_p=kappa_p, kappa_c=kappa_c)
    print("  薬クラス数: %d（成分 %d）" % (len(d.products), len(d.components)), flush=True)

    n_t1 = sum(1 for x in d.comp_pairs if x[2] == 1)
    n_t2 = sum(1 for x in d.comp_pairs if x[2] == 2)
    print("  成分ペア: Tier1=%d Tier2=%d" % (n_t1, n_t2), flush=True)

    d.meta = dict(getattr(d, "meta", {}))
    if args.tier2 == "off":
        before = len(d.comp_pairs)
        d.comp_pairs = [x for x in d.comp_pairs if x[2] == 1]
        d.meta.update(tier_policy="tier0_tier1_only",
                      n_comp_pairs=len(d.comp_pairs),
                      n_tier2_removed=before - len(d.comp_pairs))
        print("  → Tier2 を除去: %d → %d ペア" % (before, len(d.comp_pairs)), flush=True)
    else:
        d.meta.update(tier_policy="tier0_tier1_tier2")
    d.meta.update(granularity=args.granularity,
                  kappa_p=args.kappa_p, kappa_c=args.kappa_c,
                  cap_wh=d.cap_wh, n_classes=len(d.products))

    if args.tier2_cap is not None:
        d.tier2_cap_extra = args.tier2_cap
        d.meta["tier2_cap_extra"] = args.tier2_cap
        print("  → Tier2 受入上限 %.2f（p_m=%.2f）" % (args.tier2_cap, args.p_m), flush=True)

    print("  LP 構築 ...", flush=True)
    lp = pool_lp.build_pool_ef(d)
    print("    cols=%d ub_rows=%d eq_rows=%d"
          % (lp["ncol"], lp["A_ub"].shape[0], lp["A_eq"].shape[0]), flush=True)

    print("  HiGHS 求解 (method=%s, threads=%d, crossover=%s, presolve=%s) ..."
          % (args.method, args.threads, "off" if args.no_crossover else "on",
             args.presolve), flush=True)
    z, info = solve_highs(lp, threads=args.threads, method=args.method,
                          crossover=not args.no_crossover, presolve=args.presolve)
    print("    status=%s obj=%.6e time=%.1fs"
          % (info["status"], info["objective"], info["walltime"]), flush=True)

    if info["status"] != "Optimal":
        print("  [!] 最適解ではないので CSV には status のみ記録する", flush=True)
        append_row(csv_path, dict(tag=args.tag, granularity=args.granularity,
                                  axis=args.axis, tier2=args.tier2,
                                  kappa_p=args.kappa_p, kappa_c=args.kappa_c,
                                  budget_rate=args.budget_rate, r_high=args.r_high,
                                  alpha=args.alpha, p_m=args.p_m,
                                  delta1=args.delta1, delta2=args.delta2,
                                  tier2_cap=("" if args.tier2_cap is None else args.tier2_cap),
                                  status=info["status"],
                                  walltime=round(time.time() - t_start)))
        raise SystemExit(2)

    summary = save.extract_and_save(z, lp, d, outdir, method=args.tag, solve_info=info)

    # 備蓄の総量と平均単価（κ の効果を見るための指標）
    off, dims = lp["off"], lp["dims"]
    nW, nK = dims["nW"], dims["nK"]
    s_arr = z[off["s"]: off["s"] + nW * nK].reshape(nW, nK)
    stock = float(s_arr.sum())
    spend = float(sum(d.cost[k] * s_arr[:, i].sum() for i, k in enumerate(lp["products"])))
    price = spend / stock if stock > 1e-9 else 0.0
    print("  総備蓄 %.0f 患者・日   平均単価 %.2f 円/患者日" % (stock, price), flush=True)

    sc = summary.get("scenario_cost", {})
    append_row(csv_path, dict(
        tag=args.tag, granularity=args.granularity, axis=args.axis,
        tier2=args.tier2, kappa_p=args.kappa_p, kappa_c=args.kappa_c,
        budget_rate=args.budget_rate, r_high=args.r_high, alpha=args.alpha,
        p_m=args.p_m, delta1=args.delta1, delta2=args.delta2,
        tier2_cap=("" if args.tier2_cap is None else args.tier2_cap),
        status=info["status"],
        obj=round(summary["objective"]), storage_cost=round(summary["storage_cost"]),
        cvar=round(summary["cvar"]), E_unmet=round(summary["E_unmet"]),
        E_unmet_high=round(summary["E_unmet_high"]),
        total_stock=round(stock), mean_unit_price=round(price, 2),
        tier0_subst=round(summary.get("tier0_subst", 0.0)),
        tier1_subst=round(summary["tier1_subst"]),
        tier2_subst=round(summary["tier2_subst"]),
        S1_cost=round(sc.get("S1", 0)), S2_cost=round(sc.get("S2", 0)),
        S3_cost=round(sc.get("S3", 0)),
        walltime=round(time.time() - t_start)))

    if args.tier2 == "off" and summary["tier2_subst"] > 1e-6:
        print("  [!] Tier2 禁止のはずが 0 でない", flush=True)
    print("\n=== 完了 (%.1f 分) → %s ===" % ((time.time() - t_start) / 60, outdir), flush=True)
