# -*- coding: utf-8 -*-
"""LP dict（pool_lp.build_pool_ef の戻り値）を各ソルバで解く共通関数。"""
import time
import numpy as np
import scipy.sparse as sp


def _row_form(lp):
    """A_ub(≤), A_eq(=) を1つの行制約に統合し、CSC と行上下限を返す。"""
    A = sp.vstack([lp["A_ub"], lp["A_eq"]]).tocsc()
    n_ub = lp["A_ub"].shape[0]
    r_lo = np.concatenate([np.full(n_ub, -np.inf), lp["b_eq"]])
    r_hi = np.concatenate([lp["b_ub"], lp["b_eq"]])
    return A, r_lo, r_hi


def write_mps(lp, path):
    """highspy を用いて MPS を書き出す（PDLP/PIPS 等での再利用向け）。"""
    import highspy
    h = _highs_from_lp(lp, threads=1, silent=True)
    h.writeModel(path)
    return path


def _highs_from_lp(lp, threads=0, presolve="on", silent=False, time_limit=None):
    import highspy
    INF = highspy.kHighsInf
    A, r_lo, r_hi = _row_form(lp)
    col_lo = np.where(np.isneginf(lp["lb"]), -INF, lp["lb"]).astype(float)
    col_hi = np.where(np.isposinf(lp["ub"]), INF, lp["ub"]).astype(float)
    r_lo = np.where(np.isneginf(r_lo), -INF, r_lo).astype(float)
    r_hi = np.where(np.isposinf(r_hi), INF, r_hi).astype(float)

    model = highspy.HighsLp()
    model.num_col_ = int(A.shape[1])
    model.num_row_ = int(A.shape[0])
    model.col_cost_ = lp["c"].astype(float)
    model.col_lower_ = col_lo
    model.col_upper_ = col_hi
    model.row_lower_ = r_lo
    model.row_upper_ = r_hi
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_ = A.indptr.astype(np.int32)
    model.a_matrix_.index_ = A.indices.astype(np.int32)
    model.a_matrix_.value_ = A.data.astype(float)

    h = highspy.Highs()
    if silent:
        h.setOptionValue("output_flag", False)
    if threads:
        h.setOptionValue("threads", int(threads))
    h.setOptionValue("presolve", presolve)
    if time_limit:
        h.setOptionValue("time_limit", float(time_limit))
    h.passModel(model)
    return h


def solve_highs(lp, threads=0, method="ipm", presolve="on", crossover=True, time_limit=None):
    """
    HiGHS で解く。method: 'ipm'(内点), 'simplex', 'choose'。
    戻り値: (z, info)
    """
    import highspy
    t0 = time.time()
    h = _highs_from_lp(lp, threads=threads, presolve=presolve, time_limit=time_limit)
    h.setOptionValue("solver", method)   # 'simplex' | 'ipm' | 'choose'
    if method == "ipm":
        h.setOptionValue("run_crossover", "on" if crossover else "off")
    h.run()
    z = np.array(h.getSolution().col_value)
    st = h.getModelStatus()
    info = dict(solver="HiGHS", method=method,
                status=str(h.modelStatusToString(st)),
                objective=float(h.getObjectiveValue()),
                walltime=time.time() - t0)
    return z, info


def solve_pdlp(lp, time_limit=3600.0, tol=1e-6, threads=0):
    """
    OR-Tools PDLP（一次法）で解く。巨大LP・GPU向けの比較用。
    戻り値: (z, info)
    """
    from ortools.linear_solver import pywraplp
    t0 = time.time()
    solver = pywraplp.Solver.CreateSolver("PDLP")
    if solver is None:
        raise RuntimeError("PDLP solver not available in this OR-Tools build")
    INF = solver.infinity()
    ncol = lp["ncol"]
    col_lo = lp["lb"]; col_hi = lp["ub"]
    xs = []
    for j in range(ncol):
        lo = -INF if np.isneginf(col_lo[j]) else float(col_lo[j])
        hi = INF if np.isposinf(col_hi[j]) else float(col_hi[j])
        xs.append(solver.NumVar(lo, hi, "z%d" % j))
    # 目的
    obj = solver.Objective()
    cvec = lp["c"]
    nz = np.nonzero(cvec)[0]
    for j in nz:
        obj.SetCoefficient(xs[int(j)], float(cvec[j]))
    obj.SetMinimization()
    # 制約（≤ と =）を行ごとに追加
    def add_rows(A, lo_arr, hi_arr):
        Acsr = A.tocsr()
        for i in range(Acsr.shape[0]):
            ct = solver.Constraint(float(lo_arr[i]), float(hi_arr[i]))
            s, e = Acsr.indptr[i], Acsr.indptr[i + 1]
            for idx in range(s, e):
                ct.SetCoefficient(xs[int(Acsr.indices[idx])], float(Acsr.data[idx]))
    add_rows(lp["A_ub"], np.full(lp["A_ub"].shape[0], -INF), lp["b_ub"])
    add_rows(lp["A_eq"], lp["b_eq"], lp["b_eq"])
    solver.SetTimeLimit(int(time_limit * 1000))
    st = solver.Solve()
    z = np.array([xs[j].solution_value() for j in range(ncol)])
    info = dict(solver="PDLP", status=int(st), objective=solver.Objective().Value(),
                walltime=time.time() - t0)
    return z, info
