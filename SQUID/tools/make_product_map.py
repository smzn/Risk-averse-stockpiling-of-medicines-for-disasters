# -*- coding: utf-8 -*-
"""class_id（C_xxx）と製品情報の対応表を作る。
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SQ = os.path.dirname(HERE)
CLASSES = [("A10B", "v9", "糖尿病用剤"),
           ("N03A", "v9_N03A", "抗てんかん薬"),
           ("B01A", "v9_B01A", "抗血栓薬")]


def short_name(name, seibun, maxlen=20):
    """図のラベル用に医薬品名を詰める。

    後発品の名称は「成分名＋剤形＋規格＋「メーカー」」という形が多く、
    識別に必要なのは末尾のメーカー名と規格である。単純に後ろを切ると
    どれも「メトホルミン塩酸塩錠５００ｍｇＭＴ…」になって区別がつかない。
    そこで先頭の成分名を落とし、それでも長い場合は中間を省いて末尾を残す。
    """
    s = str(name).strip()
    b = str(seibun).strip()
    if b and b != 'nan' and s.startswith(b):
        s = s[len(b):]                      # 先頭の成分名を落とす
    if len(s) <= maxlen:
        return s
    head = (maxlen - 1) // 2
    tail = maxlen - 1 - head
    return s[:head] + '…' + s[-tail:]       # 末尾（メーカー名）を必ず残す


def fix_generic_flag(m):
    """統一名収載品目の先発／後発を直す。

    NDB の「後発品区分」は、複数メーカーが 1 つの薬価基準収載医薬品コードと
    薬価を共有する統一名収載品目で 0（＝先発扱い）になることがある。同じ
    メトホルミン MT シリーズでも 250mgMT(3962002F2019) は 0、500mgMT
    (3962002F3015) は 1 と NDB 内で割れている。

    統一名収載品目は先発品も後発品と同一薬価で収載され、薬価基準の側が
    先発／後発の区別を付けていない。そこで同一コードを複数銘柄が共有して
    いれば統一名収載とみなし、薬価基準マスタで「先発品」と明記されたものを
    除いて後発に寄せる（ワーファリン錠のように真の先発が統一名収載に含まれる
    場合も後発表示になるが、薬価は後発品と同額なので図の読み方は変わらない）。
    """
    shared = (m.groupby("薬価基準収載医薬品コード")["医薬品名"]
               .transform("nunique") > 1)
    bad = shared & ~m["_先発品明記"] & (m["後発品"] == "先発")
    if bad.any():
        print("   統一名収載のため先発→後発に修正: %d 製品" % int(bad.sum()))
        for code, s in m[bad].groupby("薬価基準収載医薬品コード"):
            print("     %s  %-22s ×%d" % (code, s["成分名"].iloc[0], len(s)))
    m.loc[bad, "後発品"] = "後発"
    return m.drop(columns=["_先発品明記"])


def build():
    out = []
    for code, folder, jp in CLASSES:
        d = pd.read_csv(os.path.join(SQ, folder, "data", "drug_usage_df.csv"))
        t = pd.DataFrame({
            "薬効群": "%s %s" % (code, jp),
            "class_id": d["class_id"],
            "ATCcode": d["ATCcode"],
            "成分名": d["成分名"],
            "医薬品名": d["医薬品名"],
            "薬価基準収載医薬品コード": d["薬価基準収載医薬品コード"],
            "規格": d.get("規格"),
            "メーカー名": d.get("メーカー名"),
            "薬価": d["薬価"],
            "後発品": d["後発品\n区分"].map({0: "先発", 1: "後発"}).fillna("不明"),
            "1錠あたりの薬量mg": d["1錠あたりの薬量"],
            "静岡県年間使用量": d["静岡県"],
            "災害フラグ": d["災害フラグ"],
        })
        t["_先発品明記"] = d["先発医薬品"].eq("先発品").to_numpy()
        t["短縮名"] = [short_name(a, b) for a, b in zip(t["医薬品名"], t["成分名"])]
        out.append(t)
    m = pd.concat(out, ignore_index=True)
    m = fix_generic_flag(m)
    p = os.path.join(SQ, "product_map.csv")
    m.to_csv(p, index=False)
    print("→ %s（%d 製品）" % (p, len(m)))
    for code, _, jp in CLASSES:
        sub = m[m.薬効群.str.startswith(code)]
        print("   %-5s %s: %3d 製品 / %2d 成分 / 先発 %d・後発 %d"
              % (code, jp, len(sub), sub.ATCcode.nunique(),
                 (sub.後発品 == "先発").sum(), (sub.後発品 == "後発").sum()))
    return m


if __name__ == "__main__":
    m = build()
    print()
    print("=== 例：A10BA02 メトホルミン ===")
    print(m[m.ATCcode == "A10BA02"][
        ["class_id", "医薬品名", "薬価基準収載医薬品コード", "薬価", "後発品"]]
        .head(6).to_string(index=False))
