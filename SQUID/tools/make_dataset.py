# -*- coding: utf-8 -*-
"""薬効群（ATC L3）を指定して、モデル用データ一式を作る。

  python make_dataset.py N03A --dry-run
  python make_dataset.py N03A            # → ../v9_N03A/data
  python make_dataset.py B01A            # → ../v9_B01A/data
"""
import os
import re
import shutil
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # SQUID/
DATA = os.path.join(os.path.dirname(ROOT), "data") # 実データ/data
XLSX = os.path.join(DATA, "医薬品",
                    "第９回NDBオープンデータ_都道府県別薬効分類別数量_20260821ATCコード追加.xlsx")
ATCMAP = os.path.join(DATA, "医薬品", "atc_ddd_prep", "out", "component_atc_map.csv")
REF = os.path.join(ROOT, "data")                   # 糖尿病版（拠点・倉庫・人口シェアの供給元）

L5_RE = re.compile(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$")
PERIOD_DAYS = 7.0

CURATED = os.path.join(HERE, "curated_atc.csv")
OVERRIDE = os.path.join(HERE, "curated_dose_override.csv")   # 配合剤の指標成分量

# 薬効群ごとの表示名。成分の定義そのものは curated_atc.csv が持つ。
CLASSES = {
    "N03A": "抗てんかん薬",
    "B01A": "抗血栓薬",
}


def load_ndb(l3):
    """NDB を読み、curated_atc.csv に載っている成分だけを残す。"""
    d = pd.read_excel(XLSX, sheet_name="Sheet1")
    d = d[d["静岡県"].fillna(0) > 0].copy()
    cur = pd.read_csv(CURATED)
    cur = cur[cur["薬効群"] == l3]
    d = d.merge(cur[["成分名", "ATCcode", "RDD_mg", "RDD出典", "確認要"]],
                on="成分名", how="inner")
    return d, cur


def validate(d, cur, l3):
    """使えない製品を落とし、除外理由を返す。"""
    rep = []

    def drop(mask, reason):
        n = int(mask.sum())
        if n:
            g = d[mask]
            rep.append(dict(理由=reason, 製品数=n, 成分数=g["ATCcode"].nunique(),
                            静岡県数量=float(g["静岡県"].sum()),
                            成分="; ".join(sorted(set(g["成分名"].astype(str)))[:4])))
        return ~mask

    keep = pd.Series(True, index=d.index)
    # 錠剤以外（%＝散剤・液剤・シロップ）は「1錠あたりの薬量」の意味が違うので扱えない
    keep &= drop(d["1錠あたりの薬量単位"].astype(str).eq("%"), "剤形が錠剤でない（薬量単位=%）")
    keep &= drop(d["1錠あたりの薬量"].isna(), "1錠あたりの薬量が欠損（指標成分の指定が要る）")
    keep &= drop(d["RDD_mg"].isna(), "RDD 未設定")
    return d[keep].copy(), pd.DataFrame(rep)


def to_mg(v, unit):
    u = str(unit).strip()
    return float(v) * {"g": 1000.0, "mg": 1.0, "μg": 0.001, "ug": 0.001}.get(u, np.nan)


def build(l3, outdir, dry=False):
    print("=" * 72)
    print(" %s  %s → %s" % (l3, CLASSES[l3], outdir))
    print("=" * 72)

    raw, cur = load_ndb(l3)
    miss = sorted(set(cur["成分名"]) - set(raw["成分名"]))
    if miss:
        print("  [!] curated_atc.csv にあるが NDB(静岡県>0)に無い成分: %s" % ", ".join(miss))
    print("  対象（curated_atc.csv 基準）: 製品 %d / 成分 %d / 数量 %.3e"
          % (len(raw), raw["ATCcode"].nunique(), raw["静岡県"].sum()))
    nq = cur[cur["確認要"] == 1]
    if len(nq):
        print("  [!] 要確認の成分 %d 件: %s" % (len(nq), ", ".join(nq["成分名"])))

    # NaN 混在で str 型になることがあるので数値に揃える
    raw["1錠あたりの薬量"] = pd.to_numeric(raw["1錠あたりの薬量"], errors="coerce")
    raw["1錠あたりの薬量単位"] = raw["1錠あたりの薬量単位"].astype(object)

    ov = pd.read_csv(OVERRIDE)
    ov = ov[ov["薬効群"] == l3]
    if len(ov):
        mp = dict(zip(ov["医薬品名"], ov["指標成分量mg"]))
        hit = raw["医薬品名"].isin(mp)
        raw.loc[hit, "1錠あたりの薬量"] = raw.loc[hit, "医薬品名"].map(mp)
        raw.loc[hit, "1錠あたりの薬量単位"] = "mg"
        print("  配合剤の指標成分量を補完: %d 製品（%s）"
              % (int(hit.sum()), ", ".join(sorted(set(raw.loc[hit, "医薬品名"])))))

    d, rep = validate(raw, cur, l3)
    if len(rep):
        print("\n  === 除外 ===")
        print(rep.to_string(index=False))
    print("\n  採用: 製品 %d / 成分 %d / 数量 %.3e（元の %.1f%%）"
          % (len(d), d["ATCcode"].nunique(), d["静岡県"].sum(),
             100 * d["静岡県"].sum() / raw["静岡県"].sum()))

    # L4 構造（Tier1/Tier2 が成立するか）
    d["L4"] = d["ATCcode"].str[:5]
    s = d.groupby("L4").agg(成分数=("ATCcode", "nunique"), 製品数=("成分名", "size"),
                            静岡県=("静岡県", "sum"))
    s["数量%"] = (100 * s.静岡県 / s.静岡県.sum()).round(1)
    print("\n  === L4構造（Tier2の厚み）===")
    print(s.sort_values("静岡県", ascending=False).to_string())
    if d["L4"].nunique() < 2:
        print("  [!] L4 が1つしかない。Tier2 が構成できないので対象に適さない")

    # 高リスク成分（災害フラグ=1 かつ 単剤）
    hi = d[(d["災害フラグ"] == 1) & (~d["成分名"].astype(str).str.contains("・"))]
    hi_comps = sorted(hi["ATCcode"].unique())
    print("\n  === 高リスク成分（JADM災害フラグ=1 かつ 単剤）===")
    for c in hi_comps:
        g = hi[hi.ATCcode == c]
        print("    %s  %-22s 製品%2d  数量 %.3e" %
              (c, g["成分名"].iloc[0], len(g), g["静岡県"].sum()))
    if not hi_comps:
        print("    [!] 該当なし。ω の設定を別途決める必要がある")

    if dry:
        return

    os.makedirs(outdir, exist_ok=True)
    d = d.sort_values(["ATCcode", "医薬品名"]).reset_index(drop=True)
    d["class_id"] = ["C_%03d" % i for i in range(len(d))]
    # 1錠あたりの薬量を mg に揃える（μg 表記の製品がある）
    conv = d["1錠あたりの薬量単位"].astype(str).map({"mg": 1.0, "μg": 0.001, "ug": 0.001})
    if conv.isna().any():
        raise SystemExit("未知の薬量単位: %s"
                         % sorted(set(d.loc[conv.isna(), "1錠あたりの薬量単位"])))
    d["1錠あたりの薬量"] = d["1錠あたりの薬量"] * conv
    d["1錠あたりの薬量単位"] = "mg"

    # ── 薬剤に依存しないファイルは複製 ──
    for f in ["nodes.csv", "warehouses.csv", "scenarios.csv"]:
        shutil.copy(os.path.join(REF, f), os.path.join(outdir, f))

    # ── 人口シェアを既存 baseline_demand.csv から厳密に逆算 ──
    ref_base = pd.read_csv(os.path.join(REF, "baseline_demand.csv"))
    ref_use = pd.read_csv(os.path.join(REF, "drug_usage_df.csv"))
    ref_ann = ref_use.groupby("class_id")["静岡県"].sum()
    one = ref_base[ref_base.class_id == ref_ann.index[0]]
    denom = ref_ann.iloc[0] / 365.0 * PERIOD_DAYS
    pop = one[["node_id"]].copy()
    pop["pop_share"] = one["baseline_demand_mg"].values / denom
    assert abs(pop["pop_share"].sum() - 1.0) < 1e-9, pop["pop_share"].sum()
    print("\n  人口シェアを復元: 拠点 %d / 合計 %.10f" % (len(pop), pop.pop_share.sum()))

    # ── 出力 ──
    d.to_csv(os.path.join(outdir, "drug_usage_df.csv"), index=False)
    d[["医薬品名", "class_id"]].to_csv(os.path.join(outdir, "drug_class_map.csv"), index=False)
    use = (d.groupby("class_id", as_index=False)["静岡県"].sum()
             .rename(columns={"静岡県": "annual_use"}))          # ★錠数建て
    use.to_csv(os.path.join(outdir, "drug_use_by_class.csv"), index=False)

    # RDD は curated_atc.csv の値（mg/日）
    rdd = (d.groupby("ATCcode")
             .agg(成分名=("成分名", "first"), RDD_value=("RDD_mg", "first"),
                  用法=("RDD出典", "first"), 要確認=("確認要", "first"))
             .reset_index())
    rdd["RDD_unit"] = "mg"
    rdd["基準製品"] = ""
    rdd["出典種別"] = "curated"
    rdd[["ATCcode", "成分名", "RDD_value", "RDD_unit", "基準製品", "用法", "出典種別", "要確認"]] \
        .to_csv(os.path.join(outdir, "rdd_table.csv"), index=False)

    # 配合剤の指標成分（モデル側が参照する）
    o = d.merge(ov.drop(columns=["薬効群", "ATCcode"]), on="医薬品名", how="inner")
    o[["class_id", "医薬品名", "ATCcode", "指標成分", "指標成分量mg", "用法", "出典"]] \
        .to_csv(os.path.join(outdir, "dose_mg_override.csv"), index=False)

    # baseline / demand
    base = (pop.assign(key=1).merge(use.assign(key=1), on="key").drop(columns="key"))
    base["baseline_demand_mg"] = base.annual_use / 365.0 * PERIOD_DAYS * base.pop_share
    base[["node_id", "class_id", "baseline_demand_mg"]] \
        .to_csv(os.path.join(outdir, "baseline_demand.csv"), index=False)

    nodes = pd.read_csv(os.path.join(outdir, "nodes.csv"))
    sc = pd.read_csv(os.path.join(outdir, "scenarios.csv"))
    ntype = nodes.set_index("node_id")["node_type"].to_dict()

    # シナリオ倍率は既存（糖尿病版）の demand/baseline の比から逆算する。
    # SQUID にコピーされた scenarios.csv には multiplier 列が無いため。
    # こうすることで拠点条件・シナリオ条件が糖尿病版と厳密に一致する。
    rb = ref_base.set_index(["node_id", "class_id"])["baseline_demand_mg"]
    rd = pd.read_csv(os.path.join(REF, "demand.csv"))
    rd = rd[rd.class_id == ref_ann.index[0]]
    rd["base"] = rd.set_index(["node_id", "class_id"]).index.map(rb)
    rd["ratio"] = rd.demand_mg / rd.base
    rd["is_sh"] = rd.node_id.map(lambda n: ntype.get(n, "HOSPITAL") == "SHELTER")
    mult = {}
    for sid, g in rd.groupby("scenario_id"):
        m = float(g.loc[~g.is_sh, "ratio"].median())
        sm = float(g.loc[g.is_sh, "ratio"].median()) / m
        mult[sid] = (m, sm)
        assert g.loc[~g.is_sh, "ratio"].std() < 1e-9, "倍率が拠点で一定でない"
    print("  シナリオ倍率を逆算: " +
          "  ".join("%s(×%.2f, 避難所×%.2f)" % (k, v[0], v[1]) for k, v in sorted(mult.items())))
    base["is_shelter"] = base.node_id.map(lambda n: ntype.get(n, "HOSPITAL") == "SHELTER")
    out = []
    for sid, (m, sm) in sorted(mult.items()):
        t = base[["node_id", "class_id"]].copy()
        t["scenario_id"] = sid
        t["demand_mg"] = (base.baseline_demand_mg * m *
                          np.where(base.is_shelter, sm, 1.0))
        out.append(t)
    dem = pd.concat(out, ignore_index=True)
    dem.to_csv(os.path.join(outdir, "demand.csv"), index=False)

    print("  demand.csv %d 行 / baseline %d 行 / 製品 %d / 成分 %d"
          % (len(dem), len(base), len(d), d.ATCcode.nunique()))
    print("  → %s" % outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("l3", choices=sorted(CLASSES),
                    help="A10B(糖尿病)は計算済みのため対象外（ATCの振り直しをしない）")
    ap.add_argument("--out", required=False)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    build(a.l3, a.out or os.path.join(ROOT, "v9_%s" % a.l3, "data"), a.dry_run)
