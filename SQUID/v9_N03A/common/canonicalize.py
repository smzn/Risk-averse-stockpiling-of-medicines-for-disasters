# -*- coding: utf-8 -*-
"""解の一意化（正準化）。
"""
import numpy as np
from scipy.optimize import linprog


def waterfill(total, weight, cap, tol=1e-9, max_iter=200):
    """total を weight に比例して配分する。ただし各要素は cap を超えない。

    上限に当たった要素を固定し、残りを再度比例配分する（water-filling）。
    上限のもとで (値/weight) の分散を最小にする配分に一致する。
    """
    weight = np.asarray(weight, dtype=float)
    cap = np.asarray(cap, dtype=float)
    out = np.zeros_like(weight)
    free = (weight > tol) & (cap > tol)
    rest = float(total)
    for _ in range(max_iter):
        if rest <= tol or not free.any():
            break
        w = weight[free]
        share = rest * w / w.sum()
        over = share > cap[free] - tol
        if not over.any():
            out[free] = share
            rest = 0.0
            break
        idx = np.where(free)[0][over]
        out[idx] = cap[idx]
        rest -= cap[idx].sum()
        free[idx] = False
    if rest > tol:
        # 上限の合計が total に届かない（理論上は起きない）。残りを比例で押し込む。
        rem = np.where(cap > out + tol)[0]
        if len(rem):
            out[rem] += rest * (cap[rem] - out[rem]) / (cap[rem] - out[rem]).sum()
    return out


def fair_unmet(u_nc, recv_nc, D_nc):
    """未充足を拠点間で公平化する（成分ごとに合計を保存）。

    u_nc, recv_nc, D_nc : (nN, nC)
      u    … 現在の未充足
      recv … 代替受入量 R[n,c]（自己消費非負制約の上限計算に必要）
      D    … 成分別需要 Dc[n,c]
    戻り値: 再配分後の u（合計は成分ごとに不変）
    """
    out = np.zeros_like(u_nc)
    cap = np.maximum(0.0, D_nc - recv_nc)         # u ≤ D − R
    for c in range(u_nc.shape[1]):
        tot = float(u_nc[:, c].sum())
        if tot <= 1e-9:
            continue
        out[:, c] = waterfill(tot, D_nc[:, c], cap[:, c])
    return out


def min_tier0_component(own_n, D_nk, S_k, donate=0.0):
    """1成分1シナリオの最小 Tier0 と、それを達成する配送 Y[n,k] を返す。

    own_n  : (nN,)     自成分の需要を自成分の製品で満たす量（= Dc − R − u）
    D_nk   : (nN, m)   製品別需要
    S_k    : (m,)      製品別の総在庫
    donate : float     この成分から他成分へ供出する総量（在庫を残しておく必要がある）
    """
    nN, m = D_nk.shape
    T = float(own_n.sum())
    if T <= 1e-9:
        return 0.0, np.zeros((nN, m))
    if m == 1:
        # 製品が 1 つなら付け替える相手がいない。own ≤ D なので Tier0 は 0。
        return 0.0, np.minimum(own_n, D_nk[:, 0]).reshape(-1, 1)

    # max Σ Y  s.t. Σ_k Y ≤ own_n, Σ_n Y ≤ S_k, ΣY ≤ ΣS_k − donate, 0 ≤ Y ≤ D
    nv = nN * m
    rows_n = np.repeat(np.arange(nN), m)
    rows_k = nN + np.tile(np.arange(m), nN)
    cols = np.arange(nv)
    import scipy.sparse as sp
    rows_all = np.full(nv, nN + m)                       # 供出ぶんの在庫を残す行
    A_ub = sp.csr_matrix(
        (np.ones(3 * nv),
         (np.concatenate([rows_n, rows_k, rows_all]),
          np.concatenate([cols, cols, cols]))),
        shape=(nN + m + 1, nv))
    b_ub = np.concatenate([own_n, S_k,
                           [max(0.0, float(S_k.sum()) - float(donate))]])
    res = linprog(-np.ones(nv), A_ub=A_ub, b_ub=b_ub,
                  bounds=list(zip(np.zeros(nv), D_nk.reshape(-1))),
                  method="highs")
    if not res.success:
        raise RuntimeError("min_tier0 LP failed: %s" % res.message)
    Y = res.x.reshape(nN, m)
    tier0 = T - float(Y.sum())

    # 余り（Tier0 分）を、在庫に余裕のある製品から需要比で割り付けて X を作る
    X = Y.copy()
    short = own_n - Y.sum(axis=1)                  # 各拠点で足りない分
    left = S_k - Y.sum(axis=0)                     # 各製品の残在庫
    left = np.maximum(left, 0.0)
    idx = np.where(short > 1e-9)[0]
    for n in idx:
        need = short[n]
        avail = left.sum()
        if avail <= 1e-12:
            break
        take = left * (need / avail) if avail > need else left.copy()
        X[n] += take
        left -= take
        left = np.maximum(left, 0.0)
    return tier0, X
