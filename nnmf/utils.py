import math
from typing import List, Optional, Sequence

import numpy as np


def dist_index(x: np.ndarray, index: Sequence[int]) -> np.ndarray:
    target = x[index]
    if target.ndim == 1:
        r = np.sum((x - target) ** 2, axis=1)
    else:
        r = np.sum((x[:, None, :] - target[None, :, :]) ** 2, axis=2)
    if np.any(r < 0):
        r = np.maximum(r, 0)
    return r


def dist_fun(x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    if y is None:
        x2 = np.sum(x ** 2, axis=1)
        r = x2[:, None] - 2 * (x @ x.T) + x2[None, :]
    else:
        if x.shape[1] != y.shape[1]:
            raise ValueError("The number of columns in x and y need to be the same.")
        x2 = np.sum(x ** 2, axis=1)
        y2 = np.sum(y ** 2, axis=1)
        r = x2[:, None] - 2 * (x @ y.T) + y2[None, :]
    if np.any(r < 0):
        r = np.maximum(r, 0)
    return r


def groupondist(location: np.ndarray, size: Optional[int] = None, no_groups: Optional[int] = None) -> List[str]:
    n = location.shape[0]
    left = list(range(n))
    if no_groups is None and size is None:
        raise ValueError("You must determine the size or number of groups")
    if size is None:
        size = int(math.ceil(n / no_groups))

    i = 1
    batch_vec = ["b0"] * n
    rng = np.random.default_rng()

    while len(left) > size:
        start = rng.integers(0, len(left))
        dist = dist_index(location[left], [start])
        if dist.ndim > 1:
            dist = dist[:, 0]
        batch_index = np.argsort(dist)[:size]
        for idx in batch_index:
            batch_vec[left[idx]] = f"b{i}"
        i += 1
        batch_set = set(batch_index.tolist())
        left = [left[idx] for idx in range(len(left)) if idx not in batch_set]

    return batch_vec


def topfeatures(signatures: np.ndarray, feature_names: Sequence[str], ntop: int = 10) -> np.ndarray:
    if signatures.shape[1] != len(feature_names):
        raise ValueError("feature_names need to be the same length as the number of columns in signatures.")
    dat = signatures.T
    dat_new = []
    for ii in range(dat.shape[0]):
        rr = dat[ii, :]
        m1 = int(np.argmax(rr))
        rr_ex = np.delete(rr, m1)
        m2 = int(np.argmax(rr_ex)) if rr_ex.size else m1
        m2_idx = m2 if m2 < m1 else m2 + 1
        mm = np.full(rr.shape, rr[m1])
        mm[m1] = rr[m2_idx]
        ns = rr * np.log(1 + rr / (mm + 1e-10))
        dat_new.append(ns)
    dat_new = np.vstack(dat_new)

    weight_topgene = []
    for topic in range(dat.shape[1]):
        idx = np.argsort(dat_new[:, topic])[::-1]
        weighting = [feature_names[i] for i in idx[:ntop]]
        weight_topgene.append([topic + 1] + weighting)
    return np.array(weight_topgene, dtype=object)


def estimate_lengthscale(
    data: np.ndarray,
    location: Optional[np.ndarray] = None,
    max_avg_nn: int = 20,
    batch: Sequence[int] = (1,),
    max_pct: Optional[float] = None,
    dist: Optional[np.ndarray] = None,
    column_ls: bool = False,
) -> np.ndarray:
    if location is not None and data.shape[0] != location.shape[0]:
        raise ValueError("The number of rows in location must match the number of rows in data.")

    unique_batches = np.unique(batch)
    if len(unique_batches) == 1:
        if data.shape[0] > 50000:
            raise ValueError(
                "Too many observations to run in one batch. Use groupondist() with size 20000 or smaller."
            )
    else:
        if data.shape[0] != len(batch):
            raise ValueError("The length of batch must match the number of rows in data.")
        batch_mask = np.asarray(batch) == unique_batches[0]
        data = data[batch_mask]
        if location is not None:
            location = location[batch_mask]

    if dist is None:
        if location is None:
            raise ValueError("Specify either location or dist (distance) for your data points.")
        dist = dist_fun(location).astype(float)
        np.fill_diagonal(dist, np.inf)
    else:
        if dist.shape[0] != data.shape[0] or dist.shape[1] != data.shape[0]:
            raise ValueError("The distance matrix must match the number of rows in data.")
        dist = dist.astype(float, copy=True)
        np.fill_diagonal(dist, np.inf)

    dist = np.sqrt(dist)
    min_val = float(np.min(dist))

    if max_pct is None:
        if max_avg_nn < 1 or int(max_avg_nn) != max_avg_nn:
            raise ValueError("max_avg_nn should be a positive integer.")
        max_val = float(np.mean(np.sort(dist, axis=1)[:, int(max_avg_nn) - 1]))
    else:
        if max_pct > 1 or max_pct <= 0:
            raise ValueError("max_pct needs to be a value between 0 and 1.")
        idx = int(math.floor(dist.shape[1] * max_pct))
        max_val = float(np.mean(np.sort(dist, axis=1)[:, idx]))

    lengthscale = np.linspace(min_val, max_val, num=10)
    data_norm = data / np.sum(data, axis=1, keepdims=True)

    test_error = []
    for ls in lengthscale:
        if ls > 0:
            sigma = np.exp(-(dist ** 2) / (ls ** 2))
        else:
            sigma = np.exp(-np.inf * (dist + 1))
        sigma[sigma < 0.1] = 0
        r_sum = np.sum(sigma, axis=1, keepdims=True)
        sigma[r_sum[:, 0] == 0, :] = 1
        weights = sigma / np.sum(sigma, axis=1, keepdims=True)
        if column_ls:
            fit_group = np.mean((data_norm - weights @ data_norm) ** 2, axis=0)
            test_error.append(fit_group)
        else:
            fit_group = float(np.mean((data_norm - weights @ data_norm) ** 2))
            test_error.append(fit_group)

    return np.column_stack([lengthscale, np.array(test_error, dtype=object)])


def nn_adj(
    location: np.ndarray,
    celltype: Sequence[str],
    nn: int = 5,
    sampleid: Optional[Sequence[str]] = None,
) -> np.ndarray:
    celltype = np.asarray(celltype)
    sampleid_arr = None if sampleid is None else np.asarray(sampleid)
    celltypes = np.unique(celltype)
    sum_cell = {ct: np.sum(celltype == ct) for ct in celltypes}

    def collect_neighbors(cell_label: str) -> dict:
        neighbors = []
        if sampleid_arr is None:
            indices = np.where(celltype == cell_label)[0]
            for idx in indices:
                dist = np.sum((location - location[idx]) ** 2, axis=1)
                nearest = np.argsort(dist)[1 : nn + 1]
                neighbors.extend(celltype[nearest])
        else:
            for sid in np.unique(sampleid_arr):
                mask = sampleid_arr == sid
                location_sub = location[mask]
                celltype_sub = celltype[mask]
                indices = np.where(celltype_sub == cell_label)[0]
                for idx in indices:
                    dist = np.sum((location_sub - location_sub[idx]) ** 2, axis=1)
                    nearest = np.argsort(dist)[1 : nn + 1]
                    neighbors.extend(celltype_sub[nearest])

        counts = {ct: 0 for ct in celltypes}
        for n in neighbors:
            counts[n] += 1
        nn_norm = {ct: 1 / sum_cell[ct] for ct in celltypes}
        for ct in celltypes:
            if counts[ct] > 0:
                nn_norm[ct] = counts[ct] * nn_norm[ct]
        total = sum(nn_norm.values())
        if total == 0:
            return {ct: 0.0 for ct in celltypes}
        return {ct: nn_norm[ct] / total for ct in celltypes}

    adj = []
    for ct in celltypes:
        row = collect_neighbors(ct)
        adj.append([row[c] for c in celltypes])

    adj_mat = np.array(adj).T
    return adj_mat


def select_no_signatures(
    data: np.ndarray,
    location: Optional[np.ndarray] = None,
    k_seq: Sequence[int] = tuple(range(2, 31)),
    method: str = "nnmf",
    maxiter: int = 300,
    initial: int = 3,
    small_iter: int = 30,
    seed: Optional[int] = None,
) -> dict:
    results = []
    for k in k_seq:
        if method == "nnmf":
            from .core import nnmf

            res = nnmf(
                data=data,
                no_signatures=int(k),
                location=location,
                maxiter=maxiter,
                initial=initial,
                small_iter=small_iter,
                seed=seed,
            )
            weights = res["weights"]
            err = float(res.get("error", np.nan))
        elif method == "spatial_pnmf":
            from .core import spatial_pnmf

            res = spatial_pnmf(
                data=data,
                no_signatures=int(k),
                location=location,
                maxiter=maxiter,
                initial=initial,
                small_iter=small_iter,
                seed=seed,
            )
            weights = res["weights"]
            err = float(np.linalg.norm(data - weights @ res["signatures"]))
        else:
            raise ValueError("method must be 'nnmf' or 'spatial_pnmf'.")

        wnorm = weights / np.maximum(np.linalg.norm(weights, axis=0, keepdims=True), 1e-12)
        dev_ortho = float(np.linalg.norm(wnorm.T @ wnorm - np.eye(wnorm.shape[1]), ord="fro") / wnorm.shape[1])
        results.append((int(k), dev_ortho, err))

    arr = np.array(results, dtype=float)
    ks = arr[:, 0]
    dev = arr[:, 1]
    err = arr[:, 2]

    if len(err) >= 3:
        second = np.diff(err, n=2)
        best_idx = int(np.argmax(np.abs(second))) + 1
    else:
        best_idx = int(np.argmin(err))

    return {
        "best_k": int(ks[best_idx]),
        "k": ks,
        "dev_ortho": dev,
        "error": err,
    }


def auto_lambda(
    data: np.ndarray,
    location: np.ndarray,
    no_signatures: int,
    lambdas: Sequence[float] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    smoothness_metric: str = "lap_energy",
    smoothness_target: float = 0.95,
    error_tolerance: float = 0.05,
    seed: Optional[int] = None,
    maxiter: int = 1000,
    kernel_cutoff: float = 0.1,
) -> dict:
    from .core import spatial_pnmf

    if smoothness_metric not in {"lap_energy"}:
        raise ValueError("smoothness_metric must be 'lap_energy'.")

    dist = dist_fun(location)
    dist_sqrt = np.sqrt(dist)
    idx = min(14, dist_sqrt.shape[1] - 1)
    lengthscale = float(np.mean(np.sort(dist_sqrt, axis=1)[:, idx]))
    sigma = np.exp(-dist / (lengthscale ** 2))
    sigma[sigma < kernel_cutoff] = 0
    sigma = sigma / np.sum(sigma, axis=1, keepdims=True)
    lap = np.eye(sigma.shape[0], dtype=float) - sigma

    def lap_energy(scores: np.ndarray) -> float:
        vals = []
        for i in range(scores.shape[1]):
            s = scores[:, i]
            vals.append(float(s.T @ lap @ s) / s.shape[0])
        return float(np.mean(vals))

    rows = []
    for lam in lambdas:
        res = spatial_pnmf(
            data=data,
            no_signatures=no_signatures,
            location=location,
            seed=seed,
            maxiter=maxiter,
            use_smoothing=(lam > 0),
            lambda_smooth=float(lam),
        )
        scores = res["scores"]
        weights = res["weights"]
        data_proc = (data / np.sum(data, axis=1, keepdims=True)) * 1e4
        data_proc = np.log1p(data_proc)
        recon_error = float(np.linalg.norm(data_proc - (scores @ weights)))
        smooth_val = lap_energy(scores)
        rows.append((float(lam), recon_error, smooth_val))

    arr = np.array(rows, dtype=float)
    lam_vals = arr[:, 0]
    err_vals = arr[:, 1]
    smooth_vals = arr[:, 2]

    # For lap_energy, lower is smoother; convert to a 0..1 smoothness score
    s_min = float(np.min(smooth_vals))
    s_max = float(np.max(smooth_vals))
    smooth_score = (s_max - smooth_vals) / max(s_max - s_min, 1e-12)

    err_min = float(np.min(err_vals))
    err_ok = err_vals <= (1.0 + error_tolerance) * err_min
    smooth_ok = smooth_score >= smoothness_target
    feasible = np.where(err_ok & smooth_ok)[0]
    if feasible.size:
        best_idx = int(feasible[0])
    else:
        # fallback: minimize a weighted sum
        err_score = (err_vals - err_min) / max(np.max(err_vals) - err_min, 1e-12)
        total = err_score + (1.0 - smooth_score)
        best_idx = int(np.argmin(total))

    return {
        "best_lambda": float(lam_vals[best_idx]),
        "lambda": lam_vals,
        "recon_error": err_vals,
        "smoothness": smooth_vals,
        "smooth_score": smooth_score,
    }
