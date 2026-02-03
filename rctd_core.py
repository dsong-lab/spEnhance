from __future__ import annotations

"""Core RCTD likelihoods and optimization routines."""

from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

DEFAULT_QMAT_PATH = os.path.join(os.path.dirname(__file__), "data", "qmat.npz")
DEFAULT_MIN_CHANGE = 1e-3
DEFAULT_IRWLS_ITERS = 50
DEFAULT_IRWLS_ITERS_SCORE = 25
EPS_LAM = 1e-4
DELTA_LAM = 1e-6
EPS_QP = 1e-7
EPS_QP_CHOL = 1e-8

_FIT_CONTEXT: Dict[str, object] = {}


def _init_fit_worker(context: Dict[str, object]) -> None:
    global _FIT_CONTEXT
    _FIT_CONTEXT = context


def _fit_spot(index: int) -> Tuple[int, np.ndarray]:
    ctx = _FIT_CONTEXT
    bead = ctx["counts"][:, index]
    result = process_bead_multi(
        ctx["cell_type_info"],
        ctx["gene_idx"],
        float(ctx["n_umi"][index]),
        bead,
        ctx["cache"],
        constrain=ctx["constrain"],
        min_change=ctx["min_change"],
        max_types=ctx["max_types"],
        confidence_threshold=ctx["confidence_threshold"],
        doublet_threshold=ctx["doublet_threshold"],
        cell_type_profiles_base=ctx["cell_type_profiles_base"],
        solver=ctx["solver"],
        initial_weight_thresh=ctx["initial_weight_thresh"],
    )
    spot_weights = np.zeros(ctx["n_cell_types"], dtype=float)
    for idx, w in zip(result["cell_type_list"], result["sub_weights"]):
        spot_weights[idx] = w
    return index, spot_weights


@dataclass(frozen=True)
class LikelihoodCache:
    """Cached likelihood tables for spline interpolation."""

    q_mat: np.ndarray
    x_vals: np.ndarray
    sq_mat: np.ndarray
    k_val: int
    qmat_mode: str


def load_qmat_npz(path: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load q-matrix data from npz."""
    if path is None:
        path = DEFAULT_QMAT_PATH
    data = np.load(path, allow_pickle=True)
    x_vals = data["X_vals"]
    qmat_all = {k: data[k] for k in data.files if k != "X_vals"}
    return qmat_all, x_vals


def solve_sq(q_mat: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    """Spline second-derivative table for q-matrix interpolation."""
    n = q_mat.shape[1] - 1
    deltas = np.diff(x_vals)
    m_mat = np.zeros((n - 1, n - 1), dtype=float)
    diag = 2 * (deltas[0 : n - 1] + deltas[1:n])
    np.fill_diagonal(m_mat, diag)
    idx = np.arange(1, n - 1)
    m_mat[idx, idx - 1] = deltas[1 : n - 1]
    m_mat[idx - 1, idx] = deltas[1 : n - 1]

    f_b = np.diff(q_mat.T, axis=0) / deltas[:, None]
    f_bd = 6 * np.diff(f_b, axis=0)
    sq = np.linalg.solve(m_mat, f_bd).T
    pad = np.zeros((q_mat.shape[0], 1), dtype=float)
    return np.concatenate([pad, sq, pad], axis=1)


def _prepare_qmat(q_mat: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    mode = mode.lower()
    if mode not in {"neglog", "log", "prob", "raw", "auto"}:
        raise ValueError(f"Unknown qmat mode: {mode}")

    if mode == "raw":
        return q_mat, mode

    if mode == "auto":
        q_min = float(np.nanmin(q_mat))
        q_max = float(np.nanmax(q_mat))
        if q_min >= 0.0 and q_max <= 1.0:
            mode = "prob"
        elif q_min >= 0.0:
            mode = "neglog"
        else:
            mode = "log"

    if mode == "prob":
        q_mat = np.clip(q_mat, 1e-300, 1.0)
        q_mat = np.log(q_mat)
    elif mode == "neglog":
        q_mat = -q_mat
    # log/raw are used as-is (spacexr stores log p values)
    return q_mat, mode


def set_likelihood_vars(q_mat: np.ndarray, x_vals: np.ndarray, qmat_mode: str = "auto") -> LikelihoodCache:
    """Prepare likelihood cache (q-matrix, spline, k cap)."""
    q_mat_log, final_mode = _prepare_qmat(q_mat, qmat_mode)
    return LikelihoodCache(
        q_mat=q_mat_log,
        x_vals=x_vals,
        sq_mat=solve_sq(q_mat_log, x_vals),
        k_val=q_mat_log.shape[0] - 3,
        qmat_mode=final_mode,
    )


def _calc_q_all(y: np.ndarray, lam: np.ndarray, cache: LikelihoodCache) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    lam = np.asarray(lam, dtype=float)
    y = np.minimum(y, cache.k_val)

    x_max = float(cache.x_vals[-1])
    lam = np.clip(lam, EPS_LAM, x_max - EPS_LAM)

    l_val = np.floor(np.sqrt(lam / DELTA_LAM)).astype(int)
    m_r = np.minimum(l_val - 9, 40) + np.maximum(
        np.ceil(np.sqrt(np.maximum(l_val - 48.7499, 0) * 4)) - 2, 0
    )
    m_r = m_r.astype(int)
    m = np.clip(m_r - 1, 0, cache.x_vals.size - 2)

    ti1 = cache.x_vals[m]
    ti = cache.x_vals[m + 1]
    hi = ti - ti1

    fti1 = cache.q_mat[y, m]
    fti = cache.q_mat[y, m + 1]
    zi1 = cache.sq_mat[y, m]
    zi = cache.sq_mat[y, m + 1]

    diff1 = lam - ti1
    diff2 = ti - lam
    diff3 = fti / hi - zi * hi / 6.0
    diff4 = fti1 / hi - zi1 * hi / 6.0
    zdi = zi / hi
    zdi1 = zi1 / hi

    d0 = zdi * diff1**3 / 6.0 + zdi1 * diff2**3 / 6.0 + diff3 * diff1 + diff4 * diff2
    d1 = zdi * diff1**2 / 2.0 - zdi1 * diff2**2 / 2.0 + diff3 - diff4
    d2 = zdi * diff1 + zdi1 * diff2
    return d0, d1, d2


def calc_log_l_vec(lam: np.ndarray, y: np.ndarray, cache: LikelihoodCache, return_vec: bool = False) -> float | np.ndarray:
    d0, _, _ = _calc_q_all(y, lam, cache)
    log_l = -d0
    if return_vec:
        return log_l
    return float(np.sum(log_l))


def _get_der_fast(
    s_mat: np.ndarray,
    s_mat_cross: np.ndarray,
    b: np.ndarray,
    prediction: np.ndarray,
    cache: LikelihoodCache,
    bulk_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if bulk_mode:
        d1_vec = -2 * (np.log(prediction) - np.log(b)) / prediction
        d2_vec = -2 * (1 - np.log(prediction) + np.log(b)) / (prediction**2)
    else:
        _, d1_vec, d2_vec = _calc_q_all(b, prediction, cache)

    grad = -d1_vec @ s_mat
    hess_c = -d2_vec @ s_mat_cross
    n_types = s_mat.shape[1]
    hess = np.zeros((n_types, n_types), dtype=float)
    counter = 0
    for i in range(n_types):
        length = n_types - i
        hess[i, i:] = hess_c[counter : counter + length]
        hess[i, i] = hess[i, i] / 2.0
        counter += length
    hess = hess + hess.T
    return grad, hess


def _psd(h_mat: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(h_mat)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _build_s_mat(s_mat: np.ndarray) -> np.ndarray:
    n_types = s_mat.shape[1]
    cols = []
    for i in range(n_types):
        for j in range(i, n_types):
            cols.append(s_mat[:, i] * s_mat[:, j])
    return np.column_stack(cols)


def _project_simplex(x: np.ndarray, s: float) -> np.ndarray:
    if s <= 0:
        return np.zeros_like(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, u.size + 1) > (cssv - s))[0]
    if rho.size == 0:
        return np.zeros_like(x)
    rho = rho[-1]
    theta = (cssv[rho] - s) / float(rho + 1)
    return np.maximum(x - theta, 0.0)


def _qp_to_lsq(d_mat: np.ndarray, d_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Convert 1/2 x^T D x - d^T x into 1/2 ||L x - y||^2
    # with L^T L = D and y = solve(L^T, d).
    d_mat = d_mat + EPS_QP_CHOL * np.eye(d_mat.shape[0])
    try:
        l_mat = np.linalg.cholesky(d_mat)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(d_mat)
        eigvals = np.maximum(eigvals, EPS_QP_CHOL)
        l_mat = eigvecs @ np.diag(np.sqrt(eigvals))
    y_vec = np.linalg.solve(l_mat.T, d_vec)
    return l_mat, y_vec


def _solve_qp(
    d_mat: np.ndarray,
    d_vec: np.ndarray,
    bounds: List[Tuple[float, Optional[float]]],
    eq_sum: Optional[float],
    solver: str = "quadprog",
) -> np.ndarray:
    solver = solver.lower()

    if solver == "quadprog":
        try:
            import quadprog  # type: ignore

            n = d_vec.size
            # Build C and b for constraints C^T x >= b
            lower = np.array([b[0] for b in bounds], dtype=float)
            c_mat = np.eye(n)
            b_vec = lower.copy()
            meq = 0
            if eq_sum is not None:
                c_mat = np.column_stack([np.ones(n), c_mat])
                b_vec = np.concatenate([[eq_sum], b_vec])
                meq = 1
            sol = quadprog.solve_qp(d_mat, d_vec, c_mat, b_vec, meq)[0]
            return sol
        except Exception:
            solver = "slsqp"

    if solver == "cvxopt":
        try:
            from cvxopt import matrix, solvers  # type: ignore

            solvers.options["show_progress"] = False
            n = d_vec.size
            p = matrix(d_mat)
            q = matrix(-d_vec)
            g_list = []
            h_list = []
            for i, (lo, hi) in enumerate(bounds):
                row = np.zeros(n)
                row[i] = -1.0
                g_list.append(row)
                h_list.append(-lo if lo is not None else 0.0)
                if hi is not None:
                    row = np.zeros(n)
                    row[i] = 1.0
                    g_list.append(row)
                    h_list.append(hi)
            g = matrix(np.vstack(g_list)) if g_list else None
            h = matrix(np.array(h_list)) if h_list else None
            a = None
            b = None
            if eq_sum is not None:
                a = matrix(np.ones((1, n)))
                b = matrix(np.array([eq_sum]))
            sol = solvers.qp(p, q, g, h, a, b)
            return np.asarray(sol["x"]).ravel()
        except Exception:
            solver = "slsqp"

    if solver == "osqp":
        try:
            import osqp  # type: ignore
            import scipy.sparse as sp

            n = d_vec.size
            p = sp.csc_matrix(d_mat)
            q = -d_vec
            l = np.full(n, -np.inf)
            u = np.full(n, np.inf)
            for i, (lo, hi) in enumerate(bounds):
                if lo is not None:
                    l[i] = lo
                if hi is not None:
                    u[i] = hi
            if eq_sum is not None:
                a = sp.vstack([sp.eye(n), np.ones((1, n))]).tocsc()
                l = np.concatenate([l, [eq_sum]])
                u = np.concatenate([u, [eq_sum]])
            else:
                a = sp.eye(n, format="csc")
            prob = osqp.OSQP()
            prob.setup(p, q, a, l, u, verbose=False)
            res = prob.solve()
            if res.x is not None:
                return res.x
        except Exception:
            solver = "slsqp"

    if solver in {"lsq_linear", "nnls"}:
        l_mat, y_vec = _qp_to_lsq(d_mat, d_vec)
        if solver == "lsq_linear":
            from scipy.optimize import lsq_linear

            lower = np.array([b[0] for b in bounds], dtype=float)
            upper = np.array([b[1] if b[1] is not None else np.inf for b in bounds], dtype=float)
            res = lsq_linear(l_mat, y_vec, bounds=(lower, upper), lsmr_tol="auto")
            x = res.x
        else:
            from scipy.optimize import nnls

            x, _ = nnls(l_mat, y_vec)
            for i, (_, hi) in enumerate(bounds):
                if hi is not None:
                    x[i] = min(x[i], hi)
        if eq_sum is not None:
            x = _project_simplex(np.maximum(x, 0.0), eq_sum)
        return x

    # default SLSQP
    def obj(x: np.ndarray) -> float:
        return 0.5 * float(x @ d_mat @ x) - float(d_vec @ x)

    def grad(x: np.ndarray) -> np.ndarray:
        return d_mat @ x - d_vec

    constraints = []
    if eq_sum is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x: np.sum(x) - eq_sum,
                "jac": lambda x: np.ones_like(x),
            }
        )
    x0 = np.zeros_like(d_vec)
    res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=constraints, method="SLSQP")
    if not res.success:
        return res.x
    return res.x


def solve_ols(s_mat: np.ndarray, b: np.ndarray, constrain: bool = True, solver: str = "quadprog") -> np.ndarray:
    d_mat = s_mat.T @ s_mat
    d_vec = s_mat.T @ b
    norm_factor = np.linalg.norm(d_mat, 2)
    if norm_factor > 0:
        d_mat = d_mat / norm_factor
        d_vec = d_vec / norm_factor
    d_mat = d_mat + EPS_QP * np.eye(d_mat.shape[0])
    bounds = [(0.0, None)] * d_vec.size
    eq_sum = 1.0 if constrain else None
    return _solve_qp(d_mat, d_vec, bounds, eq_sum, solver=solver)


def solve_wls(
    s_mat: np.ndarray,
    s_mat_cross: np.ndarray,
    b: np.ndarray,
    initial_sol: np.ndarray,
    n_umi: float,
    cache: LikelihoodCache,
    bulk_mode: bool = False,
    constrain: bool = False,
    solver: str = "quadprog",
) -> np.ndarray:
    solution = np.maximum(initial_sol, 0)
    prediction = np.abs(s_mat @ solution)
    threshold = max(EPS_LAM, n_umi * 1e-7)
    prediction[prediction < threshold] = threshold

    grad, hess = _get_der_fast(s_mat, s_mat_cross, b, prediction, cache, bulk_mode=bulk_mode)
    d_vec = -grad
    d_mat = _psd(hess)
    norm_factor = np.linalg.norm(d_mat, 2)
    if norm_factor > 0:
        d_mat = d_mat / norm_factor
        d_vec = d_vec / norm_factor
    d_mat = d_mat + EPS_QP * np.eye(d_mat.shape[0])

    bounds = [(-float(val), None) for val in solution]
    eq_sum = (1.0 - float(np.sum(solution))) if constrain else None
    delta = _solve_qp(d_mat, d_vec, bounds, eq_sum, solver=solver)
    alpha = 0.3
    return solution + alpha * delta


def solve_irwls_weights(
    s_mat: np.ndarray,
    b: np.ndarray,
    n_umi: float,
    cache: LikelihoodCache,
    ols: bool = False,
    constrain: bool = True,
    n_iter: int = DEFAULT_IRWLS_ITERS,
    min_change: float = DEFAULT_MIN_CHANGE,
    bulk_mode: bool = False,
    solver: str = "quadprog",
) -> Dict[str, np.ndarray]:
    b = np.asarray(b, dtype=float)
    if not bulk_mode:
        b = np.minimum(b, cache.k_val)
    solution = np.full(s_mat.shape[1], 1.0 / s_mat.shape[1], dtype=float)
    if ols:
        return {"weights": solve_ols(s_mat, b, constrain=constrain, solver=solver), "converged": True}

    iterations = 0
    change = 1.0
    s_mat_cross = _build_s_mat(s_mat)
    while change > min_change and iterations < n_iter:
        new_solution = solve_wls(
            s_mat,
            s_mat_cross,
            b,
            solution,
            n_umi,
            cache,
            bulk_mode=bulk_mode,
            constrain=constrain,
            solver=solver,
        )
        change = float(np.sum(np.abs(new_solution - solution)))
        solution = new_solution
        iterations += 1
    return {"weights": solution, "converged": change <= min_change}


def decompose_sparse(
    cell_type_profiles: np.ndarray,
    n_umi: float,
    bead: np.ndarray,
    cache: LikelihoodCache,
    custom_list: Iterable[int],
    score_mode: bool = False,
    constrain: bool = True,
    min_change: float = 1e-3,
    solver: str = "quadprog",
) -> Dict[str, np.ndarray] | float:
    cell_types = list(custom_list)
    reg_data = cell_type_profiles[:, cell_types]
    n_iter = DEFAULT_IRWLS_ITERS_SCORE if score_mode else DEFAULT_IRWLS_ITERS
    results = solve_irwls_weights(
        reg_data,
        bead,
        n_umi,
        cache,
        ols=False,
        constrain=constrain,
        n_iter=n_iter,
        min_change=min_change,
        solver=solver,
    )
    if not score_mode:
        weights = results["weights"]
        total = float(np.sum(weights))
        if total > 0:
            weights = weights / total
        results["weights"] = weights
        return results
    prediction = reg_data @ results["weights"]
    return calc_log_l_vec(prediction, bead, cache)


def decompose_full(
    cell_type_profiles: np.ndarray,
    n_umi: float,
    bead: np.ndarray,
    cache: LikelihoodCache,
    constrain: bool = True,
    ols: bool = False,
    n_iter: int = DEFAULT_IRWLS_ITERS,
    min_change: float = 1e-3,
    bulk_mode: bool = False,
    solver: str = "quadprog",
) -> Dict[str, np.ndarray]:
    return solve_irwls_weights(
        cell_type_profiles,
        bead,
        n_umi,
        cache,
        ols=ols,
        constrain=constrain,
        n_iter=n_iter,
        min_change=min_change,
        bulk_mode=bulk_mode,
        solver=solver,
    )


def process_bead_multi(
    cell_type_info: Tuple[np.ndarray, List[str], int],
    gene_list: np.ndarray,
    n_umi: float,
    bead: np.ndarray,
    cache: LikelihoodCache,
    constrain: bool = True,
    min_change: float = 1e-3,
    max_types: int = 4,
    confidence_threshold: float = 5.0,
    doublet_threshold: float = 20.0,
    cell_type_profiles_base: Optional[np.ndarray] = None,
    solver: str = "quadprog",
    debug: bool = False,
    initial_weight_thresh: float = 0.01,
) -> Dict[str, object]:
    if cell_type_profiles_base is None:
        cell_type_profiles_base = cell_type_info[0][gene_list, :]
    cell_type_profiles = cell_type_profiles_base * n_umi
    cell_type_profiles = np.asarray(cell_type_profiles, dtype=float)
    score_cache: Dict[Tuple[int, ...], float] = {}
    weight_cache: Dict[Tuple[int, ...], Dict[str, np.ndarray]] = {}

    def _key(types: Iterable[int]) -> Tuple[int, ...]:
        return tuple(types)

    def _score(types: Iterable[int]) -> float:
        key = _key(types)
        if key in score_cache:
            return score_cache[key]
        val = float(
            decompose_sparse(
                cell_type_profiles,
                n_umi,
                bead,
                cache,
                custom_list=list(key),
                score_mode=True,
                constrain=constrain,
                min_change=min_change,
                solver=solver,
            )
        )
        score_cache[key] = val
        return val

    def _weights(types: Iterable[int]) -> Dict[str, np.ndarray]:
        key = _key(types)
        if key in weight_cache:
            return weight_cache[key]
        res = decompose_sparse(
            cell_type_profiles,
            n_umi,
            bead,
            cache,
            custom_list=list(key),
            score_mode=False,
            constrain=constrain,
            min_change=min_change,
            solver=solver,
        )
        weight_cache[key] = res
        return res

    results_all = decompose_full(
        cell_type_profiles,
        n_umi,
        bead,
        cache,
        constrain=constrain,
        min_change=min_change,
        solver=solver,
    )
    all_weights = results_all["weights"]
    conv_all = results_all["converged"]

    candidates = np.where(all_weights > initial_weight_thresh)[0]
    if len(candidates) == 0:
        max_w = float(np.max(all_weights)) if all_weights.size else 0.0
        raise ValueError(
            "process_bead_multi: no cell types passed weight threshold on full mode. "
            f"Check that enough counts are present for each spot. "
            f"max_weight={max_w:.6f}, threshold={initial_weight_thresh:.6f}"
        )
    candidates = candidates.tolist()

    cell_type_list: List[int] = []
    curr_score = 1.0e10
    for _ in range(max_types):
        min_score = curr_score
        best_type = None
        for idx in candidates:
            cur_list = cell_type_list + [idx]
            score = _score(cur_list)
            if score < min_score:
                best_type = idx
                min_score = score
        if best_type is None or min_score > curr_score - doublet_threshold:
            break
        cell_type_list.append(best_type)
        candidates.remove(best_type)
        curr_score = min_score

    conf_list = {idx: True for idx in cell_type_list}
    for idx in cell_type_list:
        for new_idx in candidates:
            cur_list = [i for i in cell_type_list if i != idx] + [new_idx]
            score = _score(cur_list)
            if score < curr_score + confidence_threshold:
                conf_list[idx] = False
                break

    sub_results = _weights(cell_type_list)
    sub_weights = sub_results["weights"]
    conv_sub = sub_results["converged"]
    out = {
        "all_weights": all_weights,
        "cell_type_list": cell_type_list,
        "conf_list": conf_list,
        "sub_weights": sub_weights,
        "min_score": curr_score,
        "conv_all": conv_all,
        "conv_sub": conv_sub,
    }
    if debug:
        out["score_cache"] = score_cache
        out["weight_cache"] = weight_cache
    return out


def _align_genes(
    ref_genes: Iterable[str],
    spatial_genes: Iterable[str],
    allowed_genes: Optional[Iterable[str]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    spatial_index = {gene: i for i, gene in enumerate(spatial_genes)}
    ref_index = {gene: i for i, gene in enumerate(ref_genes)}
    if allowed_genes is None:
        gene_list = [gene for gene in ref_genes if gene in spatial_index]
    else:
        allowed_set = set(allowed_genes)
        gene_list = [gene for gene in ref_genes if gene in spatial_index and gene in allowed_set]
    ref_idx = np.array([ref_index[g] for g in gene_list], dtype=int)
    spatial_idx = np.array([spatial_index[g] for g in gene_list], dtype=int)
    return gene_list, ref_idx, spatial_idx


def fit(
    reference: Mapping[str, object],
    spatial: Mapping[str, object],
    *,
    qmat_path: Optional[str] = None,
    sigma: float = 1.0,
    qmat_mode: str = "auto",
    gene_list: Optional[Iterable[str]] = None,
    max_types: int = 4,
    confidence_threshold: float = 5.0,
    doublet_threshold: float = 20.0,
    min_change: float = 1e-3,
    constrain: bool = False,
    n_jobs: int = 1,
    use_float32: bool = False,
    solver: str = "quadprog",
    initial_weight_thresh: float = 0.01,
    n_umi: Optional[np.ndarray] = None,
    debug_spot: Optional[str] = None,
    debug_out: Optional[str] = None,
    return_conf: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
    qmat_all, x_vals = load_qmat_npz(qmat_path)
    sigma_key = str(int(round(sigma * 100)))
    if sigma_key not in qmat_all:
        raise ValueError(f"sigma={sigma} (key {sigma_key}) is not available in qmat data.")
    cache = set_likelihood_vars(qmat_all[sigma_key], x_vals, qmat_mode=qmat_mode)

    ref_genes = list(reference["gene_names"])
    spatial_genes = list(spatial["gene_names"])
    gene_list, ref_idx, spatial_idx = _align_genes(ref_genes, spatial_genes, allowed_genes=gene_list)
    if len(gene_list) == 0:
        raise ValueError("No overlapping genes between reference and spatial datasets.")

    dtype = np.float32 if use_float32 else float
    cell_type_means = np.asarray(reference["cell_type_means"], dtype=dtype)[ref_idx, :]
    cell_type_names = list(reference["cell_type_names"])
    counts = np.asarray(spatial["counts"], dtype=dtype)
    if counts.shape[0] != len(spatial_genes) and counts.shape[1] == len(spatial_genes):
        counts = counts.T
    if counts.shape[0] != len(spatial_genes):
        raise ValueError("Spatial counts shape does not align with provided gene_names.")
    counts = counts[spatial_idx, :]
    spot_names = list(spatial.get("spot_names", [f"spot_{i}" for i in range(counts.shape[1])]))

    if n_umi is None:
        n_umi = np.sum(counts, axis=0)
    else:
        n_umi = np.asarray(n_umi, dtype=float)
        if n_umi.shape[0] != counts.shape[1]:
            raise ValueError("Provided n_umi does not match number of spatial spots.")
    cell_type_info = (cell_type_means, cell_type_names, len(cell_type_names))
    cell_type_profiles_base = cell_type_means

    weights = np.zeros((counts.shape[1], len(cell_type_names)), dtype=float)
    gene_idx = np.arange(cell_type_profiles_base.shape[0])
    if n_jobs > 1 and debug_spot is not None:
        raise ValueError("debug_spot requires n_jobs=1.")
    if n_jobs > 1 and return_conf:
        raise ValueError("return_conf requires n_jobs=1.")
    if n_jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        context = {
            "counts": counts,
            "n_umi": n_umi,
            "cell_type_info": cell_type_info,
            "gene_idx": gene_idx,
            "cell_type_profiles_base": cell_type_profiles_base,
            "cache": cache,
            "constrain": constrain,
            "min_change": min_change,
            "max_types": max_types,
            "confidence_threshold": confidence_threshold,
            "doublet_threshold": doublet_threshold,
            "n_cell_types": len(cell_type_names),
            "solver": solver,
            "initial_weight_thresh": initial_weight_thresh,
        }
        with ctx.Pool(processes=n_jobs, initializer=_init_fit_worker, initargs=(context,)) as pool:
            for idx, spot_weights in pool.imap_unordered(_fit_spot, range(counts.shape[1]), chunksize=4):
                weights[idx, :] = spot_weights
    else:
        debug_payload = None
        conf_lists: List[Dict[int, bool]] = [] if return_conf else []
        for i in range(counts.shape[1]):
            bead = counts[:, i]
            debug_flag = debug_spot is not None and spot_names[i] == debug_spot
            result = process_bead_multi(
                cell_type_info,
                gene_idx,
                float(n_umi[i]),
                bead,
                cache,
                constrain=constrain,
                min_change=min_change,
                max_types=max_types,
                confidence_threshold=confidence_threshold,
                doublet_threshold=doublet_threshold,
                cell_type_profiles_base=cell_type_profiles_base,
                solver=solver,
                debug=debug_flag,
                initial_weight_thresh=initial_weight_thresh,
            )
            if debug_flag:
                debug_payload = {
                    "spot": spot_names[i],
                    "all_weights": result.get("all_weights"),
                    "cell_type_list": result.get("cell_type_list"),
                    "conf_list": result.get("conf_list"),
                    "sub_weights": result.get("sub_weights"),
                    "min_score": result.get("min_score"),
                    "conv_all": result.get("conv_all"),
                    "conv_sub": result.get("conv_sub"),
                    "score_cache": result.get("score_cache"),
                }
            if return_conf:
                conf_lists.append(result.get("conf_list", {}))
            spot_weights = np.zeros(len(cell_type_names), dtype=float)
            for idx, w in zip(result["cell_type_list"], result["sub_weights"]):
                spot_weights[idx] = w
            weights[i, :] = spot_weights
        if debug_payload is not None and debug_out is not None:
            import json

            def _to_list(obj):
                if isinstance(obj, dict):
                    return {str(k): _to_list(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_list(v) for v in obj]
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                return obj

            with open(debug_out, "w") as f:
                json.dump(_to_list(debug_payload), f, indent=2)

    if return_conf:
        return weights, cell_type_names, spot_names, conf_lists
    return weights, cell_type_names, spot_names
