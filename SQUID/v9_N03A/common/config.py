# -*- coding: utf-8 -*-
"""SQUID 実行共通設定（N03A 抗てんかん薬）。データパス・基準パラメータ・対象ATCコード。"""
import os

# データディレクトリ（このフォルダ直下の data/ に csv を置く）
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # SQUID/
DATA = os.environ.get("PL_DATA", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("PL_RESULTS", os.path.join(ROOT, "results"))

# 必要なデータファイル（csvv2 または csvv3 からコピー）
REQUIRED_FILES = [
    "nodes.csv", "demand.csv", "scenarios.csv", "warehouses.csv",
    "drug_usage_df.csv", "rdd_table.csv", "dose_mg_override.csv",
    "baseline_demand.csv",   # c̄ 計算に必要
]

# ── 基準ケースの政策パラメータ ─────────────────────────────
GAMMA = 5.0          # 未充足の重大性  ω_base = γ·c̄
R_HIGH = 5.0         # 高リスク成分の重み倍率
DELTA1 = 0.5         # Tier1 代替ペナルティ係数  π1 = δ1·γ·c̄
DELTA2 = 0.75        # Tier2 代替ペナルティ係数  π2 = δ2·γ·c̄
BUDGET_RATE = 0.5    # 予算率 β
P_M = 0.4            # 代替許容率
ALPHA = 0.5          # CVaR 信頼水準
RHO = 0.0            # 最低備蓄比率
LAM = 1.0            # CVaR 重み λ

# ── 容量制約 ─────────────────────────────────────────────
#   Σ_k s[w,k]     ≤ Cap_w          倉庫の総容量（κ によらず一定）
#   Σ_{k∈c} s[w,k] ≤ κ_c · cap_c    成分（薬クラス）単位の調達可能上限
#   s[w,k]         ≤ κ_p · cap_k    製品（SKU）単位の調達可能上限
#
# cap_k はその製品の7日分需要／倉庫数、cap_c = Σ_{k∈c} cap_k。
#
# κ_p … 「1つの製品を通常使用量の何倍まで確保できるか」
# κ_c … 「1つの薬クラスを通常使用量の何倍まで確保できるか」
#
#   (κ_p, κ_c) = (1, 1)     … v1_3 と厳密に一致
#   (κ_p, κ_c) = (inf, 1)   … クラス総量は据え置きのまま、**クラス内の製品構成だけ自由**
#                             ATC は薬クラス=成分なので κ_c=1 が効いて v1_3 のまま変わらない。
#                             PL だけが「クラス内で安い後発品に寄せる」自由を得る
#                             ＝ 粒度の効果だけを取り出した比較
#   (κ_p, κ_c) = (inf, inf) … クラス横断の集中も許す（両モデルに効く別の効果が混ざる）
#
# κ_p=1 では成分内の製品間代替（Tier0）は構造的に発生しえない
# （どの製品も自分の需要を超えて積めないため、製品を自分の需要先へ配れば足りる）。
KAPPA_P = 1.0
KAPPA_C = float("inf")

# 高リスク成分
#
# 一次情報は日本災害医学会「災害時超急性期における必須医薬品リスト
# （DMATによる救命救急医療用医薬品を除く）2026年改訂版」93項目
#   https://jadm.or.jp/contents/model/
# NDB の「災害フラグ」はこのリスト由来。規則は「災害フラグ=1 かつ単剤」で、
# 糖尿病版の高リスク3成分もこの規則に一致する。
# レベチラセタムは JADM に明記されており、フラグでも拾えている。
HIGH_RISK_COMPS = [
    'フェノバルビタール',
    'レベチラセタム',
]

# 対象 ATCcode（15成分）。PLでは各コードに属する全 class_id(製品) が対象。
ALL_CODES = [
    'N03AA02',
    'N03AB02',
    'N03AE01',
    'N03AF01',
    'N03AF03',
    'N03AG01',
    'N03AX',
    'N03AX03',
    'N03AX09',
    'N03AX11',
    'N03AX12',
    'N03AX14',
    'N03AX15',
    'N03AX18',
    'N03AX22',
]
ALL_32 = ALL_CODES   # 後方互換

# 動作確認用の小サブセット（数量上位4成分）。--subset で使用。
SUBSET_SMALL = ['N03AG01', 'N03AX14', 'N03AF01', 'N03AE01']
