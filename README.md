# Risk-averse stockpiling of medicines for disasters considering ingredient and therapeutic substitutability

Code used for the numerical experiments in

> H. Ohba, L. Y. Tun, S. Hori, J. Kamei, K. Kumamaru, S. Mizuno,
> *Risk-averse stockpiling of medicines for disasters considering ingredient and therapeutic substitutability*, 2026 (manuscript).

[English](#english) | [日本語](#日本語)

---

## English

### 1. Overview

This repository contains the programs that were actually used to produce the results of the paper: a risk-averse two-stage stochastic programming model (CVaR-LP) for pre-disaster stockpiling of pharmaceuticals in Shizuoka Prefecture, Japan.

* Product-level (PL) model. Stock decisions are made per product (brand/generic, strength, dosage form). Substitution between products is organized in a Tier 0 / Tier 1 / Tier 2 hierarchy defined by the ATC classification (same ATC level 5 / level 4 / level 3), with penalties `(0, π1, π2)`.
* Pooling reformulation. Only the second-stage substitution is aggregated to the ingredient (ATC code) level, which keeps the first stage at the product level while reducing the LP to a tractable size (32.44 million columns for the antidiabetic group).
* Solver. HiGHS interior-point method with crossover (`highspy`), with the LP built as sparse `scipy` matrices.
* Canonicalization of the optimal solution (unmet-demand fairness across nodes and minimum Tier 0 reassignment), so that reported breakdowns do not depend on the solver's arbitrary choice.
* Three therapeutic groups. A10B antidiabetics (314 products, 32 ATC codes), N03A antiepileptics and B01A antithrombotics (15 ATC codes each), computed on identical nodes, warehouses, scenarios and population shares.

The computations were run on the SQUID supercomputer (Cybermedia Center, Osaka University; 76 cores / 200 GB per node) with Python 3.8.

### 2. Repository layout

```
.
├── README.md / LICENSE / requirements.txt
├── data/                                   data preparation (raw sources are NOT included)
│   └── 20260509_災害時における医療に緯度経度付加.ipynb   geocoding of the medical-relief hospitals
└── SQUID/
    ├── 01_可視化_eng.ipynb                 all figures and tables of the paper
    ├── product_map.csv                     class_id -> product name / price / generic flag (3 groups)
    ├── tools/
    │   ├── make_dataset.py                 builds the model CSVs for N03A / B01A from the A10B set
    │   ├── curated_atc.csv                 hand-curated ingredient -> ATC code / RDD table (N03A, B01A)
    │   ├── curated_dose_override.csv       dose overrides used by make_dataset.py
    │   └── make_product_map.py             builds product_map.csv
    ├── v9/        A10B antidiabetics      <- the model code (see below)
    ├── v9_N03A/   N03A antiepileptics     same code; only common/config.py (target ATC codes,
    └── v9_B01A/   B01A antithrombotics       high-risk ingredients) and the tag prefix differ
```

Each `v9*/` folder is self-contained (`data/` and `PL/jobs/` are created when needed and are not tracked):

| Path | Role |
|---|---|
| `common/config.py` | data paths (`PL_DATA`, `PL_RESULTS`), baseline parameters, target ATC codes, high-risk ingredients |
| `common/pl_params.py` | builds the product-level problem data (demand, costs, ω, π, capacities, substitution pairs) |
| `common/pool_lp.py` | builds the pooled two-stage CVaR-LP as sparse matrices (κp / κc capacity rows, Tier 2 cap) |
| `common/driver.py` | solves one configuration and appends a one-row summary to `results/v9_results.csv` |
| `common/solvers.py` | HiGHS wrapper (also a PDLP wrapper that was used only for early comparisons) |
| `common/save.py` | writes `storage_plan.csv`, `unmet_demand.csv`, `substitutions.csv`, Tier 0 tables, `summary.json`, `solution_z.npz` |
| `common/canonicalize.py` | unmet-demand fairness (bounded proportional allocation) and minimum-Tier 0 transportation problem |
| `PL/solve_one.py` | thin wrapper around `common/driver.py` |
| `gen_jobs.py` | generates one NQSV job script per configuration into `PL/jobs/` (`phase1`, `phase1_5`, `phase2`, `retry`) and a `submit_*.sh` |
| `postprocess_canonical.py` | re-derives the reported breakdowns from a saved solution (`results/<tag>_canon/`) |
| `results/v9_results.csv` | one-row summary of every configuration that was solved (tracked in git) |

### 3. Requirements

```bash
pip install -r requirements.txt
```

The model itself needs only `numpy`, `scipy`, `pandas` and `highspy` (versions pinned to those used on SQUID with Python 3.8). `openpyxl` and `requests` are used by the data notebooks, `matplotlib` and `geopandas` by the figure notebook.

### 4. Data

Input data are not distributed with the code. All sources are official statistics, administrative open data or published documents:

| Layer | Dataset | Use |
|---|---|---|
| Secondary medical areas | Correspondence table of secondary medical areas and municipalities (e-Gov) | medical-area attribute of each node |
| Population | Population Census, small-area tabulation (by town/block) | population share of each node |
| Medical institutions | 8th Shizuoka Prefecture Health and Medical Care Plan, medical-relief hospitals | hospital nodes (geocoded by `data/20260509_*.ipynb`) |
| Evacuation shelters | Designated evacuation shelters open data (GSI) | shelter nodes |
| Drug use | 9th NDB Open Data (quantities by prefecture; drug price listing) | annual use and price per product |
| Stockpile warehouses | Pharmaceutical stockpile centers / Red Cross blood centers (2020) | warehouse set |
| Disaster scenarios | Fujinokuni Disaster Guidebook | S1/S2/S3 demand multipliers |

The model reads the following CSV files from `SQUID/<group>/data/` (or the directory given by `PL_DATA`):

```
nodes.csv  demand.csv  baseline_demand.csv  scenarios.csv  warehouses.csv
drug_usage_df.csv  rdd_table.csv  dose_mg_override.csv
```

How they were built:

1. A10B (antidiabetics). Run `data/20260805_実データ正規化_v3病院重複修正.ipynb` in a folder that contains the raw sources; it writes `csvv3/`. Create `SQUID/v9/data/` and copy the files above into it.
2. N03A / B01A. `python SQUID/tools/make_dataset.py N03A` (or `B01A`) writes `SQUID/v9_N03A/data/` (`v9_B01A/data/`). Nodes, warehouses, scenarios and the population shares are copied/back-calculated from the A10B set so that the three groups are strictly comparable; ingredient definitions and RDDs come from `tools/curated_atc.csv`.

### 5. Running the model

Work inside one group folder, e.g. `SQUID/v9/`.

Quick local check (small subset of ingredients):

```bash
cd SQUID/v9
export PL_DATA=$PWD/data PL_RESULTS=/tmp/pl_test
python PL/solve_one.py --tag test --tier2 on --kappa-p inf --kappa-c 1 --subset --threads 4
```

Full experiments on SQUID (one job per configuration):

1. Edit `GROUP`, `MAIL` and `VENV` at the top of `gen_jobs.py` (placeholders `GXXXXX`, `your_mail@example.com`).
2. Generate the job scripts and the submit script, then submit:

```bash
python gen_jobs.py phase2 --dry-run --skip-done   # list the configurations
python gen_jobs.py phase2 --skip-done             # writes PL/jobs/*.sh and submit_phase2.sh
./submit_phase2.sh            # all; or ./submit_phase2.sh kappa / ./submit_phase2.sh budget ...
python gen_jobs.py retry      # re-generates the configurations that did not reach "Optimal"
```

`phase2` covers the baseline and the sensitivity axes used in the paper:

| Axis | Parameter | Values (baseline in bold) |
|---|---|---|
| `kappa` | product-level procurement multiplier κp (κc = 1) | 1 / 2 / 5 / ∞ |
| `kappac` | ingredient-level procurement multiplier κc (κp = ∞) | 1 / 2 / 5 / ∞ |
| `budget` | budget ratio β | 0.05 / 0.15 / 0.3 / 0.5 / 0.7 / 1.0 |
| `alpha` | CVaR confidence level α | 0.3 / 0.5 / 0.7 / 0.9 |
| `pm` | substitution allowance ratio p_m | 0.2 / 0.4 / 0.6 / 0.8 |
| `rhigh` | high-risk weight multiplier r | 1 / 3 / 5 / 7.5 / 10 |
| `delta` | penalty coefficients (δ1, δ2) | δ1: 0.25 / 0.5 / 0.75, δ2: 0.75 / 1.0 / 1.5 / 2.0 |
| `t2cap` | Tier 2 acceptance cap p_T2 | 0.1 / 0.2 / 0.3 (no cap) |
| `--tier2 off` | Tier 2 prohibited | run for every axis |

Fixed: γ = 5, λ = 1, scenario probabilities (0.5, 0.35, 0.15). The `retry` phase adds `--no-crossover` (crossover failed) or `--presolve off` (presolve declared infeasible at α = 0.9); for those runs only the objective value and the CVaR are reported in the paper.

Canonicalization of a saved solution:

```bash
python postprocess_canonical.py PL_gran_t2on_kpinf_kc1_b0p5_r5_a0p5_pm0p4_d10p5_d20p75
# -> results/<tag>_canon/  (unmet_demand.csv, tier0_by_component.csv, tier0_by_product.csv, canonical_summary.json)
```

### 6. Figures and tables

`SQUID/01_可視化_eng.ipynb` reproduces every figure and table of the paper from `v9*/results/`. Run it with `SQUID/` as the working directory; set `FIGDIR` in the setup cell to where the files should be written (by default a `figs2_1_eng/` folder of the LaTeX project). The municipality map additionally needs a municipality boundary GeoJSON (`N03` boundary data) at `data/境界/shizuoka_municipality.geojson`.

`tools/make_product_map.py` rebuilds `product_map.csv` (product names shown in the Tier 0 figures).

### 7. Notes

* With probabilities (0.5, 0.35, 0.15) and α = 0.5 the CVaR equals `0.7·f(S2) + 0.3·f(S3)`; the normal-time scenario S1 is outside the objective, so its second-stage allocation is not determined by optimality. Quantities in the paper are therefore aggregated with the CVaR weights (S2: 0.7, S3: 0.3).
* `--no-crossover` returns an interior solution whose second-stage breakdown is not comparable with vertex solutions; it was used only where crossover failed.
* Capacity rows: `Σ_k s[w,k] ≤ Cap_w` (warehouse), `Σ_{k∈c} s[w,k] ≤ κc·cap_c` (ingredient) and `s[w,k] ≤ κp·cap_k` (product). With κp = 1 no product can be stocked beyond its own 7-day demand, so Tier 0 substitution cannot arise; the baseline (κp, κc) = (∞, 1) keeps the ingredient totals and frees the product mix.
* The docstrings of `common/canonicalize.py` and `common/pool_lp.py` document the canonicalization and the LP layout in detail (Japanese).

### 8. License

MIT License. See `LICENSE`.

---

## 日本語

### 1. 概要

本リポジトリは、論文「成分・薬効代替性を考慮した災害用医薬品供給のロバスト最適化」（英題: *Risk-averse stockpiling of medicines for disasters considering ingredient and therapeutic substitutability*）の数値実験で実際に使用したプログラムをまとめたものです。静岡県を対象に、災害前の医薬品備蓄を決める リスク回避型2段階確率計画モデル（CVaR-LP） を扱います。

* 製品レベル（PL）モデル。 備蓄は製品（先発/後発・規格・剤形）単位で決めます。製品間の代替は ATC 分類に基づく Tier 0 / Tier 1 / Tier 2（ATC レベル5 / 4 / 3 が同一）の階層で表し、ペナルティは `(0, π1, π2)` です。
* プール化定式化。 第2段階の代替だけを成分（ATC コード）粒度に集約し、第1段階は製品粒度のまま LP を解ける規模に抑えます（糖尿病用剤で 3,244 万列）。
* ソルバ。 HiGHS 内点法 + crossover（`highspy`）。LP は `scipy` の疎行列で構築します。
* 解の正準化。 拠点間の未充足率の公平化と Tier 0 付け替え量の最小化により、報告値がソルバの任意性に依存しないようにします。
* 3薬効群。 A10B 糖尿病用剤（314製品・32 ATC コード）、N03A 抗てんかん薬、B01A 抗血栓薬（各15 ATC コード）。拠点・倉庫・シナリオ・人口シェアは3群で同一です。

計算は大阪大学サイバーメディアセンターのスーパーコンピュータ SQUID（1ノード 76コア / 200 GB）上で Python 3.8 を用いて行いました。

### 2. フォルダ構成

```
.
├── README.md / LICENSE / requirements.txt
├── data/                                   データ作成（元データは含みません）
│   └── 20260509_災害時における医療に緯度経度付加.ipynb   医療救護病院のジオコーディング
└── SQUID/
    ├── 01_可視化_eng.ipynb                 論文の全図表
    ├── product_map.csv                     class_id → 医薬品名・薬価・後発品フラグ（3薬効群）
    ├── tools/
    │   ├── make_dataset.py                 A10B のデータから N03A / B01A のモデル用 CSV を作成
    │   ├── curated_atc.csv                 手作業で確認した成分 → ATC コード / RDD 対応表（N03A, B01A）
    │   ├── curated_dose_override.csv       make_dataset.py が使う用量の上書き表
    │   └── make_product_map.py             product_map.csv の作成
    ├── v9/        A10B 糖尿病用剤          ← モデル本体（下記）
    ├── v9_N03A/   N03A 抗てんかん薬        コードは同一。common/config.py（対象 ATC コード・
    └── v9_B01A/   B01A 抗血栓薬               高リスク成分）とタグの接頭辞だけが異なる
```

各 `v9*/` フォルダは単独で動作します（`data/` と `PL/jobs/` は必要時に作られ、git では管理しません）。

| パス | 役割 |
|---|---|
| `common/config.py` | データパス（`PL_DATA`, `PL_RESULTS`）、基準パラメータ、対象 ATC コード、高リスク成分 |
| `common/pl_params.py` | 製品レベルの問題データ（需要・費用・ω・π・容量・代替ペア）を構築 |
| `common/pool_lp.py` | プール化 2段階 CVaR-LP を疎行列で構築（κp / κc の容量行、Tier 2 上限） |
| `common/driver.py` | 1構成を解き、`results/v9_results.csv` に1行追記 |
| `common/solvers.py` | HiGHS ラッパ（初期比較にのみ使った PDLP ラッパも含む） |
| `common/save.py` | `storage_plan.csv`, `unmet_demand.csv`, `substitutions.csv`, Tier 0 表, `summary.json`, `solution_z.npz` を保存 |
| `common/canonicalize.py` | 未充足の公平化（上限つき比例配分）と Tier 0 最小化の輸送問題 |
| `PL/solve_one.py` | `common/driver.py` の薄いラッパ |
| `gen_jobs.py` | 構成ごとの NQSV ジョブスクリプトを `PL/jobs/` に、投入用の `submit_*.sh` を生成（`phase1`, `phase1_5`, `phase2`, `retry`） |
| `postprocess_canonical.py` | 保存済みの解から報告用の内訳を作り直す（`results/<tag>_canon/`） |
| `results/v9_results.csv` | 解いた全構成の1行サマリ（git で管理） |

### 3. 必要環境

```bash
pip install -r requirements.txt
```

モデル本体に必要なのは `numpy`, `scipy`, `pandas`, `highspy` だけです（バージョン固定は SQUID の Python 3.8 環境に合わせたもの）。`openpyxl`, `requests` はデータ作成ノートブック、`matplotlib`, `geopandas` は図表ノートブックで使います。

### 4. データ

入力データはリポジトリに含めていません。いずれも公的統計・行政オープンデータ・公開資料に基づきます。

| 層 | データ | 用途 |
|---|---|---|
| 二次医療圏 | 二次医療圏と市区町村の対応表（e-Gov） | 各拠点の医療圏属性 |
| 人口 | 国勢調査 小地域集計（町丁・字別人口） | 各拠点の人口シェア |
| 医療機関 | 第8次静岡県保健医療計画 医療救護病院 | 病院拠点（`data/20260509_*.ipynb` でジオコーディング） |
| 避難所 | 指定緊急避難場所・指定避難所オープンデータ（国土地理院） | 避難所拠点 |
| 医薬品使用量 | 第9回 NDB オープンデータ（都道府県別数量、薬価） | 製品別の年間使用量と薬価 |
| 備蓄倉庫 | 医療品等備蓄センター／赤十字血液センター（2020） | 倉庫集合 |
| 災害シナリオ | ふじのくに防災ガイドブック | S1/S2/S3 の需要倍率 |

モデルは `SQUID/<薬効群>/data/`（または環境変数 `PL_DATA` のディレクトリ）から次の CSV を読みます。

```
nodes.csv  demand.csv  baseline_demand.csv  scenarios.csv  warehouses.csv
drug_usage_df.csv  rdd_table.csv  dose_mg_override.csv
```

作成手順:

1. A10B（糖尿病用剤）。 元データを置いたフォルダで `data/20260805_実データ正規化_v3病院重複修正.ipynb` を実行すると `csvv3/` ができます。`SQUID/v9/data/` を作成し、上記のファイルをコピーします。
2. N03A / B01A。 `python SQUID/tools/make_dataset.py N03A`（または `B01A`）で `SQUID/v9_N03A/data/`（`v9_B01A/data/`）を作成します。拠点・倉庫・シナリオ・人口シェアは A10B のデータから複製／逆算し、3群の比較が拠点条件の違いに影響されないようにしています。成分の定義と RDD は `tools/curated_atc.csv` に従います。

### 5. モデルの実行

薬効群のフォルダ（例: `SQUID/v9/`）の中で作業します。

ローカルでの動作確認（成分のサブセット）:

```bash
cd SQUID/v9
export PL_DATA=$PWD/data PL_RESULTS=/tmp/pl_test
python PL/solve_one.py --tag test --tier2 on --kappa-p inf --kappa-c 1 --subset --threads 4
```

SQUID での本計算（1構成 = 1ジョブ）:

1. `gen_jobs.py` 冒頭の `GROUP`, `MAIL`, `VENV` を自分の環境に書き換えます（プレースホルダ `GXXXXX`, `your_mail@example.com`）。
2. ジョブスクリプトと投入スクリプトを生成して投入します。

```bash
python gen_jobs.py phase2 --dry-run --skip-done   # 構成一覧の確認
python gen_jobs.py phase2 --skip-done             # PL/jobs/*.sh と submit_phase2.sh を生成
./submit_phase2.sh            # 全部。./submit_phase2.sh kappa / ./submit_phase2.sh budget のように軸で絞り込み可
python gen_jobs.py retry      # Optimal に到達しなかった構成を再生成
```

`phase2` は論文で用いた基準ケースと感度分析の全軸を含みます。

| 軸 | パラメータ | 値（太字が基準） |
|---|---|---|
| `kappa` | 製品単位の調達可能倍率 κp（κc = 1） | 1 / 2 / 5 / ∞ |
| `kappac` | 成分単位の調達可能倍率 κc（κp = ∞） | 1 / 2 / 5 / ∞ |
| `budget` | 予算率 β | 0.05 / 0.15 / 0.3 / 0.5 / 0.7 / 1.0 |
| `alpha` | CVaR 信頼水準 α | 0.3 / 0.5 / 0.7 / 0.9 |
| `pm` | 代替許容率 p_m | 0.2 / 0.4 / 0.6 / 0.8 |
| `rhigh` | 高リスク重み倍率 r | 1 / 3 / 5 / 7.5 / 10 |
| `delta` | ペナルティ係数 (δ1, δ2) | δ1: 0.25 / 0.5 / 0.75, δ2: 0.75 / 1.0 / 1.5 / 2.0 |
| `t2cap` | Tier 2 受入上限 p_T2 | 0.1 / 0.2 / 0.3（上限なし） |
| `--tier2 off` | Tier 2 禁止 | 全軸で実行 |

固定値: γ = 5、λ = 1、シナリオ確率 (0.5, 0.35, 0.15)。`retry` では `--no-crossover`（crossover 失敗時）または `--presolve off`（α = 0.9 で presolve が実行不可能と誤判定した場合）を付けます。これらの構成について論文では目的値と CVaR のみを報告しています。

保存済みの解の正準化:

```bash
python postprocess_canonical.py PL_gran_t2on_kpinf_kc1_b0p5_r5_a0p5_pm0p4_d10p5_d20p75
# → results/<tag>_canon/  (unmet_demand.csv, tier0_by_component.csv, tier0_by_product.csv, canonical_summary.json)
```

### 6. 図表

`SQUID/01_可視化_eng.ipynb` が `v9*/results/` から論文の全図表を再現します。`SQUID/` をカレントディレクトリにして実行し、セットアップセルの `FIGDIR` を出力先に合わせてください（既定は LaTeX プロジェクトの `figs2_1_eng/`）。市区町村地図の図には行政区域境界（国土数値情報 N03）の GeoJSON を `data/境界/shizuoka_municipality.geojson` に置く必要があります。

`tools/make_product_map.py` は `product_map.csv`（Tier 0 の図で使う製品名）を作成します。

### 7. 注意点

* シナリオ確率 (0.5, 0.35, 0.15) と α = 0.5 の組合せでは CVaR = `0.7·f(S2) + 0.3·f(S3)` となり、平常時シナリオ S1 は目的関数に入りません。そのため S1 の第2段階配分は最適性から定まらず、論文の数量は CVaR 加重（S2: 0.7, S3: 0.3）で集計しています。
* `--no-crossover` は内点解を返すため、第2段階の内訳を頂点解と比較できません。crossover が失敗した構成にのみ使用しました。
* 容量制約は `Σ_k s[w,k] ≤ Cap_w`（倉庫）、`Σ_{k∈c} s[w,k] ≤ κc·cap_c`（成分）、`s[w,k] ≤ κp·cap_k`（製品）の3段です。κp = 1 では各製品を自分の7日分需要を超えて置けないため Tier 0 代替は構造的に発生しません。基準ケース (κp, κc) = (∞, 1) は成分の総量を据え置き、成分内の製品構成だけを自由にします。
* 正準化と LP の列配置の詳細は `common/canonicalize.py` と `common/pool_lp.py` の docstring に記しています。

### 8. ライセンス

MIT License（`LICENSE` を参照）。
