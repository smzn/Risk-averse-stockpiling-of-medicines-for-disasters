# -*- coding: utf-8 -*-
"""v9_B01A（抗血栓薬）のジョブスクリプトを生成する（PL モデル）。

  python gen_jobs.py phase1     … 済（PL × Tier2 on/off × κ=1,inf = 4本）
  python gen_jobs.py phase1_5   … 粒度の効果だけを取り出す（κp=inf, κc=1 の4本）
  python gen_jobs.py phase2     … 本番（感度分析の全軸）
  python gen_jobs.py retry      … results/*.csv で Optimal でなかった構成を作り直す
  python gen_jobs.py phase1 --dry-run

1構成＝1ジョブ。SQUID の1ジョブ上限は120時間だが、途中で落ちたときの
やり直しを小さくするため構成単位に分ける（v8 と同じ方針）。
"""
import os
import csv
import glob
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

DRUG = "B01A"          # 薬効群。タグに付けて他薬効群の結果と混ざらないようにする
GROUP = "GXXXXX"
MAIL = "your_mail@example.com"
VENV = "/sqfs/work/GXXXXX/$USER/plenv/bin/activate"
MEM = "200GB"
CPUS = 76

BASE = dict(budget_rate=0.5, r_high=5.0, alpha=0.5, p_m=0.4,
            delta1=0.5, delta2=0.75, tier2_cap=None,
            kappa_p="inf", kappa_c="1")

TEMPLATE = """#!/bin/bash
# ===== v9 {gran} : {tag} =====
# 軸: {axis} / Tier2: {tier2} / κp={kp} κc={kc}
# β={b} r={r} α={a} p_m={pm} δ1={d1} δ2={d2} Tier2上限={cap}
#PBS -q SQUID
#PBS --group={group}
#PBS -l cpunum_job={cpus},elapstim_req={elaps},memsz_job={mem}
#PBS -m be
#PBS -M {mail}
#PBS -N {jobname}

set -e
module load BasePy
module --force switch python3/3.6 python3/3.8
source {venv}

cd $PBS_O_WORKDIR
export PL_DATA=$PBS_O_WORKDIR/data
export PL_RESULTS=$PBS_O_WORKDIR/results
export OMP_NUM_THREADS={cpus}
mkdir -p results/logs

python {gran}/solve_one.py \\{extra}
    --tag {tag} \\
    --axis {axis} \\
    --tier2 {tier2} \\
    --kappa-p {kp} \\
    --kappa-c {kc} \\
    --budget-rate {b} \\
    --r-high {r} \\
    --alpha {a} \\
    --p-m {pm} \\
    --delta1 {d1} \\
    --delta2 {d2}{capopt} \\
    --threads {cpus} \\
    > results/logs/{tag}.log 2>&1
"""


def f(x):
    return ("%g" % x).replace(".", "p")


def norm(kp, kc):
    """κ の実効値に正規化する。

    κ_p=1 のとき Σ_{k∈c} s[w,k] ≤ Σ_{k∈c} cap_k = cap_c が自動的に成り立つので、
    κ_c≥1 は拘束しない。したがって (1, *) は (1, 1) と同じ問題になる。
    """
    kp, kc = str(kp), str(kc)
    if kp == "1":
        kc = "1"
    return kp, kc


def done_configs():
    """results/*.csv から計算済みの構成を読む（重複投入を防ぐ）。"""
    done = {}
    for path in glob.glob(os.path.join(HERE, "results", "*.csv")):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != "Optimal":
                    continue
                if "kappa_p" in row:
                    kp, kc = row["kappa_p"], row["kappa_c"]
                else:                       # 旧列（phase1）: κ は製品単位、クラス上限なし
                    kp, kc = row.get("kappa", "1"), "inf"
                kp, kc = norm(kp, kc)
                key = (row["granularity"], row["tier2"], kp, kc,
                       f(float(row["budget_rate"])), f(float(row["r_high"])),
                       f(float(row["alpha"])), f(float(row["p_m"])),
                       f(float(row["delta1"])),
                       f(float(row["delta2"])) if row["tier2"] == "on" else "-",
                       row.get("tier2_cap") or "-")
                done[key] = row["tag"]
    return done


def key_of(gran, tier2, p):
    kp, kc = norm(p["kappa_p"], p["kappa_c"])
    return (gran.lower(), tier2, kp, kc,
            f(p["budget_rate"]), f(p["r_high"]), f(p["alpha"]), f(p["p_m"]),
            f(p["delta1"]), f(p["delta2"]) if tier2 == "on" else "-",
            ("%g" % p["tier2_cap"]) if p["tier2_cap"] is not None else "-")



def tag_of(gran, axis, tier2, p):
    s = "%s_%s_%s_t2%s_kp%s_kc%s_b%s_r%s_a%s_pm%s_d1%s" % (
        DRUG, gran, axis, tier2,
        str(p["kappa_p"]).replace(".", "p"), str(p["kappa_c"]).replace(".", "p"),
        f(p["budget_rate"]), f(p["r_high"]), f(p["alpha"]), f(p["p_m"]), f(p["delta1"]))
    if tier2 == "on":
        s += "_d2%s" % f(p["delta2"])
    if p["tier2_cap"] is not None:
        s += "_cap%s" % f(p["tier2_cap"])
    return s


def phase1():
    """済: PL × Tier2 on/off × (κp,κc)=(1,inf)/(inf,inf)。"""
    out = []
    for gran in ["PL"]:
        for kp, kc in [("1", "inf"), ("inf", "inf")]:
            for t in ["on", "off"]:
                p = dict(BASE); p["kappa_p"], p["kappa_c"] = kp, kc
                out.append((gran, "base", t, p))
    return out


def phase1_5():
    """粒度の効果だけを取り出す: κp=inf, κc=1。

    薬クラスの総量は v1_3 のまま据え置き、クラス内の製品構成だけ自由にする。
    PL だけが「クラス内で安い後発品に寄せる」自由を得る。
    """
    out = []
    for gran in ["PL"]:
        for t in ["on", "off"]:
            p = dict(BASE); p["kappa_p"], p["kappa_c"] = "inf", "1"
            out.append((gran, "gran", t, p))
    return out


def phase2():
    """本番: 6軸 + κ軸。"""
    out, seen = [], set()

    def add(gran, axis, tier2, **kw):
        p = dict(BASE); p.update(kw)
        d2 = None if tier2 == "off" else p["delta2"]
        key = (gran, tier2, p["kappa_p"], p["kappa_c"], p["budget_rate"], p["r_high"],
               p["alpha"], p["p_m"], p["delta1"], d2, p["tier2_cap"])
        if key in seen:
            return
        seen.add(key)
        out.append((gran, axis, tier2, p))

    for t in ["on", "off"]:
        add("PL", "base", t)
        for k in ["1", "2", "5", "inf"]:           # 製品単位の調達可能倍率
            add("PL", "kappa", t, kappa_p=k, kappa_c="1")
        for k in ["1", "2", "5", "inf"]:           # クラス単位の調達可能倍率
            add("PL", "kappac", t, kappa_p="inf", kappa_c=k)
        for b in [1.0, 0.7, 0.3, 0.15, 0.05]:
            add("PL", "budget", t, budget_rate=b)
        for a in [0.3, 0.7, 0.9]:
            add("PL", "alpha", t, alpha=a)
        for pm in [0.2, 0.6, 0.8]:
            add("PL", "pm", t, p_m=pm)
        for r in [1.0, 3.0, 7.5, 10.0]:
            add("PL", "rhigh", t, r_high=r)
    for d1 in [0.25, 0.5, 0.75]:
        for d2 in [0.75, 1.0, 1.5, 2.0]:
            add("PL", "delta", "on", delta1=d1, delta2=d2)
        add("PL", "delta", "off", delta1=d1)
    for cap in [0.3, 0.2, 0.1]:
        add("PL", "t2cap", "on", tier2_cap=cap)
    return out


def _retry_opts(phase, p):
    """失敗の種類に応じた再実行オプションを返す。"""
    if phase != "retry":
        return ""
    if p.get("_status") == "Infeasible":
        return "\n    --presolve off \\"      # presolve の誤判定
    return "\n    --no-crossover \\"          # crossover 失敗


def retry():
    """Optimal にならなかった構成を、crossover を切って作り直す。

    v9 で失敗した6本はいずれも
        Status interior point solve: optimal
        Status crossover:            failed   （ERROR: Ipx: Crossover failed）
    で、**IPM は最適解に到達している**（相対 gap 1e-11〜1e-13）。
    IPM が総時間の97%を占め、crossover は約1,000秒で IPX 内部の上限に達して
    打ち切られており、壁時間を伸ばしても解決しない。

    対処は失敗の種類で変える。

      status="Solve error"（IPM は optimal、crossover が失敗）
        → --no-crossover。IPM が総時間の97%を占め、crossover は約1,000秒で
          IPX 内部の上限に達して打ち切られるため、壁時間を伸ばしても解決しない。
          返るのは内点解なので、未充足と代替の内訳は頂点解の構成と直接比較できない。

      status="Infeasible"（IPM が primal infeas を宣言）
        → --presolve off。α=0.9 では 1/(1-α)=10 が係数レンジを広げ、
          presolve が実行不可能と誤判定する。残差が収束しないまま
          18 反復程度で打ち切られるのが特徴。モデル自体は実行可能である
          （同じ構成で α を下げれば解ける）。
    """
    out = []
    for path in glob.glob(os.path.join(HERE, "results", "*.csv")):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "Optimal":
                    continue
                p = dict(BASE)
                p.update(budget_rate=float(row["budget_rate"]),
                         r_high=float(row["r_high"]), alpha=float(row["alpha"]),
                         p_m=float(row["p_m"]), delta1=float(row["delta1"]),
                         delta2=float(row["delta2"]),
                         kappa_p=row.get("kappa_p", "inf"),
                         kappa_c=row.get("kappa_c", "1"),
                         tier2_cap=(float(row["tier2_cap"]) if row.get("tier2_cap") else None))
                p["_status"] = row.get("status", "")
                out.append((row["granularity"].upper(), row["axis"], row["tier2"], p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["phase1", "phase1_5", "phase2", "retry"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-done", action="store_true",
                    help="results/*.csv にある計算済み構成を除外する")
    ap.add_argument("--elaps", default="12:00:00",
                    help="壁時間。LP規模が糖尿病版の1/3〜1/4なので12時間で足りる見込み")
    args = ap.parse_args()

    cfgs = {"phase1": phase1, "phase1_5": phase1_5,
            "phase2": phase2, "retry": retry}[args.phase]()

    if args.skip_done:
        done = done_configs()
        keep, skipped = [], []
        for gran, axis, tier2, p in cfgs:
            k = key_of(gran, tier2, p)
            (skipped if k in done else keep).append(
                (gran, axis, tier2, p, done.get(k)))
        if skipped:
            print("=== 計算済みとして除外 %d 構成 ===" % len(skipped))
            for gran, axis, tier2, p, tag in skipped:
                print("  %-46s ← %s" % (tag_of(gran, axis, tier2, p), tag))
            print()
        cfgs = [(g, a, t, p) for g, a, t, p, _ in keep]
    cnt = Counter("%s/%s" % (g, a) for g, a, _, _ in cfgs)
    print("=== %s : %d 構成 ===" % (args.phase, len(cfgs)))
    for k in sorted(cnt):
        print("  %-12s %2d" % (k, cnt[k]))

    if args.dry_run:
        for g, a, t, p in cfgs:
            print("   ", tag_of(g, a, t, p))
        return

    made = {"PL": 0}
    names = []
    for gran, axis, tier2, p in cfgs:
        tag = tag_of(gran, axis, tier2, p)
        if args.phase == "retry":
            tag += "_nops" if p.get("_status") == "Infeasible" else "_nox"
        capopt = ("" if p["tier2_cap"] is None
                  else " \\\n    --tier2-cap %g" % p["tier2_cap"])
        body = TEMPLATE.format(
            tag=tag, gran=gran, axis=axis, tier2=tier2,
            kp=p["kappa_p"], kc=p["kappa_c"],
            b=p["budget_rate"], r=p["r_high"], a=p["alpha"], pm=p["p_m"],
            d1=p["delta1"], d2=p["delta2"],
            cap=("なし" if p["tier2_cap"] is None else p["tier2_cap"]),
            capopt=capopt, group=GROUP, mail=MAIL, venv=VENV,
            cpus=CPUS, elaps=args.elaps, mem=MEM, jobname=tag[:15],
            extra=_retry_opts(args.phase, p))
        path = os.path.join(HERE, gran, "jobs", "%s.sh" % tag)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
        names.append("%s/jobs/%s.sh" % (gran, tag))
        made[gran] += 1

    # 投入スクリプトは **そのフェーズで生成した .sh だけ** を明示列挙する。
    # jobs/*.sh をグロブすると、前フェーズの済んだジョブまで再投入してしまう。
    sub = os.path.join(HERE, "submit_%s.sh" % args.phase)
    with open(sub, "w") as fh:
        fh.write("#!/bin/bash\n")
        fh.write("# %s の %d 構成を投入する。\n" % (args.phase, len(names)))
        fh.write("# 引数で絞り込み可: ./submit_%s.sh 軸名\n" % args.phase)
        fh.write('cd "$(dirname "$0")"\n')
        fh.write('FILTER="$*"\nn=0\n')
        fh.write("JOBS=\"\n%s\n\"\n" % "\n".join(names))
        fh.write('for f in $JOBS; do\n')
        fh.write('  if [ ! -e "$f" ]; then echo "見つからない: $f" >&2; continue; fi\n')
        fh.write('  if [ -n "$FILTER" ]; then\n')
        fh.write("    hit=0\n")
        fh.write('    for k in $FILTER; do case "$f" in *${k}*) hit=1;; esac; done\n')
        fh.write('    [ "$hit" = "1" ] || continue\n')
        fh.write("  fi\n")
        fh.write('  echo "qsub $f"; qsub "$f"; n=$((n+1))\n')
        fh.write("done\n")
        fh.write('echo "投入: $n 本"\n')
    os.chmod(sub, 0o755)
    print("\n生成: PL %d 本" % made["PL"])
    print("投入: %s" % sub)


if __name__ == "__main__":
    main()
