from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse

from .utils import dist_fun, estimate_lengthscale


EPS = 1e-10
ERR_EPS = 1e-6


class _LegacyRNG:
    def __init__(self, seed: int):
        self._rs = np.random.RandomState(seed)

    def random(self, size: Tuple[int, ...]) -> np.ndarray:
        return self._rs.random_sample(size)


class _RRngMT:
    def __init__(self, state: List[int], mti: int):
        self.n = 624
        self.m = 397
        self.matrix_a = 0x9908B0DF
        self.upper_mask = 0x80000000
        self.lower_mask = 0x7FFFFFFF
        self.mt = [s & 0xFFFFFFFF for s in state]
        self.mti = mti

    def _genrand_int32(self) -> int:
        mag01 = [0x0, self.matrix_a]
        if self.mti >= self.n:
            for kk in range(self.n - self.m):
                y = (self.mt[kk] & self.upper_mask) | (self.mt[kk + 1] & self.lower_mask)
                self.mt[kk] = self.mt[kk + self.m] ^ (y >> 1) ^ mag01[y & 0x1]
            for kk in range(self.n - self.m, self.n - 1):
                y = (self.mt[kk] & self.upper_mask) | (self.mt[kk + 1] & self.lower_mask)
                self.mt[kk] = self.mt[kk + (self.m - self.n)] ^ (y >> 1) ^ mag01[y & 0x1]
            y = (self.mt[self.n - 1] & self.upper_mask) | (self.mt[0] & self.lower_mask)
            self.mt[self.n - 1] = self.mt[self.m - 1] ^ (y >> 1) ^ mag01[y & 0x1]
            self.mti = 0
        y = self.mt[self.mti]
        self.mti += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def random(self, size: Tuple[int, ...]) -> np.ndarray:
        total = int(np.prod(size))
        vals = np.empty(total, dtype=float)
        for i in range(total):
            vals[i] = self._genrand_int32() * (1.0 / 4294967296.0)
        return vals.reshape(size)


def _r_state_from_seed(seed: int) -> Optional[Tuple[List[int], int]]:
    rscript = os.environ.get("NNMF_RSCRIPT") or shutil.which("Rscript")
    if not rscript:
        return None
    cmd = [
        rscript,
        "-e",
        (
            "RNGkind('Mersenne-Twister','Inversion','Rejection');"
            f"set.seed({seed});"
            "cat(paste(.Random.seed, collapse=','))"
        ),
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return None
    parts = [p for p in proc.stdout.strip().split(",") if p]
    if len(parts) < 626:
        return None
    values = [int(p) for p in parts]
    mti = values[1]
    state = values[2 : 2 + 624]
    return state, mti


def _make_rng(seed: Optional[int]) -> np.random.Generator | _LegacyRNG | _RRngMT:
    if seed is None:
        return np.random.default_rng()
    r_state = _r_state_from_seed(seed)
    if r_state is not None:
        state, mti = r_state
        return _RRngMT(state, mti)
    return _LegacyRNG(seed)


def _gkl_error(y: np.ndarray, mu: np.ndarray) -> float:
    y_adj = y + ERR_EPS
    mu_adj = mu + ERR_EPS
    return float(np.sum(y_adj * (np.log(y_adj) - np.log(mu_adj)) - y_adj + mu_adj))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    row_sum = np.sum(x, axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return x / row_sum


def _normalize_cols(x: np.ndarray) -> np.ndarray:
    col_sum = np.sum(x, axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1
    return x / col_sum


def _randu_matrix(rng, n_rows: int, n_cols: int) -> np.ndarray:
    vals = rng.random((n_rows * n_cols,))
    return np.reshape(vals, (n_rows, n_cols), order="F")


def _project_signatures(weights: np.ndarray, data: np.ndarray) -> np.ndarray:
    signatures = weights.T @ data
    return np.clip(signatures, EPS, None)


def _apply_score_ortho(exposures: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return exposures
    exp_norm = exposures / np.maximum(np.linalg.norm(exposures, axis=0, keepdims=True), EPS)
    gram = exp_norm.T @ exp_norm
    off = gram - np.diag(np.diag(gram))
    penalty = 1.0 + strength * np.sum(np.abs(off), axis=1, keepdims=True)
    exposures = exposures / penalty.T
    return np.clip(exposures, EPS, None)


def _nmf1(
    data: np.ndarray,
    no_signatures: int,
    rng: np.random.Generator,
    init_exposures: Optional[np.ndarray] = None,
    init_signatures: Optional[np.ndarray] = None,
    iterations: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    genomes, mut_types = data.shape
    if init_exposures is not None:
        exposures = init_exposures.copy()
    else:
        exposures = _randu_matrix(rng, genomes, no_signatures)
    if init_signatures is not None:
        signatures = init_signatures.copy()
    else:
        signatures = _randu_matrix(rng, no_signatures, mut_types)
    estimate = exposures @ signatures

    for t in range(iterations):
        signatures *= exposures.T @ (data / estimate)
        signatures = _normalize_rows(signatures)
        signatures = np.clip(signatures, EPS, None)
        estimate = exposures @ signatures
        exposures *= (data / estimate) @ signatures.T
        exposures = np.clip(exposures, EPS, None)
        estimate = exposures @ signatures

    gkl = _gkl_error(data.ravel(), estimate.ravel())
    return exposures, signatures, gkl


def _nmfgen(
    data: np.ndarray,
    no_signatures: int,
    rng: np.random.Generator,
    init_exposures: Optional[np.ndarray] = None,
    init_signatures: Optional[np.ndarray] = None,
    init_exposures_list: Optional[List[np.ndarray]] = None,
    init_signatures_list: Optional[List[np.ndarray]] = None,
    maxiter: int = 10000,
    tolerance: float = 1e-8,
    initial: int = 10,
    small_iter: int = 100,
    error_freq: int = 10,
    score_ortho: float = 0.0,
) -> Dict[str, np.ndarray]:
    init_count = initial
    if init_exposures_list is not None and init_signatures_list is not None:
        init_count = min(len(init_exposures_list), len(init_signatures_list))
    exposures, signatures, gkl_value = _nmf1(
        data,
        no_signatures,
        rng,
        init_exposures=init_exposures_list[0] if init_exposures_list else init_exposures,
        init_signatures=init_signatures_list[0] if init_signatures_list else init_signatures,
        iterations=small_iter,
    )
    for i in range(1, init_count):
        exp_new, sig_new, gkl_new = _nmf1(
            data,
            no_signatures,
            rng,
            init_exposures=init_exposures_list[i] if init_exposures_list else None,
            init_signatures=init_signatures_list[i] if init_signatures_list else None,
            iterations=small_iter,
        )
        if gkl_new < gkl_value:
            gkl_value = gkl_new
            exposures = exp_new
            signatures = sig_new

    estimate = exposures @ signatures
    gkl_old = _gkl_error(data.ravel(), estimate.ravel())
    gkl_new = 2 * gkl_old
    gklvalues = np.zeros(maxiter, dtype=float)

    for t in range(maxiter):
        signatures *= exposures.T @ (data / estimate)
        signatures = _normalize_rows(signatures)
        signatures = np.clip(signatures, EPS, None)
        estimate = exposures @ signatures
        exposures *= (data / estimate) @ signatures.T
        exposures = np.clip(exposures, EPS, None)
        exposures = _apply_score_ortho(exposures, score_ortho)
        estimate = exposures @ signatures
        gklvalues[t] = gkl_old

        if t % error_freq == 0:
            gkl_new = _gkl_error(data.ravel(), estimate.ravel())
            if (2 * (gkl_old - gkl_new) / (0.1 + abs(2 * gkl_new)) < tolerance) and (t > error_freq):
                break
            gkl_old = gkl_new

    gkl_new = _gkl_error(data.ravel(), estimate.ravel())
    return {"exposures": exposures, "signatures": signatures, "gkl": gkl_new, "gklvalues": gklvalues}


def _nmfspatial(
    data: np.ndarray,
    no_signatures: int,
    weight: scipy.sparse.spmatrix,
    rng: np.random.Generator,
    init_exposures: Optional[np.ndarray] = None,
    init_signatures: Optional[np.ndarray] = None,
    init_exposures_list: Optional[List[np.ndarray]] = None,
    init_signatures_list: Optional[List[np.ndarray]] = None,
    maxiter: int = 10000,
    tolerance: float = 1e-8,
    initial: int = 5,
    small_iter: int = 100,
    error_freq: int = 10,
    score_ortho: float = 0.0,
) -> Dict[str, np.ndarray]:
    init_count = initial
    if init_exposures_list is not None and init_signatures_list is not None:
        init_count = min(len(init_exposures_list), len(init_signatures_list))
    exposures, signatures, gkl_value = _nmf1(
        data,
        no_signatures,
        rng,
        init_exposures=init_exposures_list[0] if init_exposures_list else init_exposures,
        init_signatures=init_signatures_list[0] if init_signatures_list else init_signatures,
        iterations=small_iter,
    )
    for i in range(1, init_count):
        exp_new, sig_new, gkl_new = _nmf1(
            data,
            no_signatures,
            rng,
            init_exposures=init_exposures_list[i] if init_exposures_list else None,
            init_signatures=init_signatures_list[i] if init_signatures_list else None,
            iterations=small_iter,
        )
        if gkl_new < gkl_value:
            gkl_value = gkl_new
            exposures = exp_new
            signatures = sig_new

    estimate = exposures @ signatures
    gkl_old = _gkl_error(data.ravel(), estimate.ravel())
    gkl_new = 2 * gkl_old
    gklvalues = np.zeros(maxiter, dtype=float)

    for t in range(maxiter):
        exposures *= (data / estimate) @ signatures.T
        exposures = np.clip(exposures, EPS, None)
        estimate = exposures @ signatures

        signatures *= exposures.T @ (data / estimate)
        signatures = np.clip(signatures, EPS, None)
        signatures = _normalize_rows(signatures)
        estimate = exposures @ signatures

        exposures *= (data / estimate) @ signatures.T
        exp_sum = np.sum(exposures, axis=1, keepdims=True)
        exp_sum[exp_sum == 0] = 1
        exposures_norm = exposures / exp_sum
        exposures_norm = weight @ exposures_norm
        exposures = exposures_norm * exp_sum
        exposures = np.clip(exposures, EPS, None)
        exposures = _normalize_cols(exposures)
        exposures = _apply_score_ortho(exposures, score_ortho)
        estimate = exposures @ signatures

        signatures *= exposures.T @ (data / estimate)
        signatures = np.clip(signatures, EPS, None)
        estimate = exposures @ signatures

        if t % error_freq == 0:
            gkl_new = _gkl_error(data.ravel(), estimate.ravel())
            if (2 * (gkl_old - gkl_new) / (0.1 + abs(2 * gkl_new)) < tolerance) and (t > error_freq):
                break
            gkl_old = gkl_new
        gklvalues[t] = gkl_old

    rsum = np.sum(signatures, axis=1, keepdims=True)
    rsum[rsum == 0] = 1
    exposures = exposures * rsum.T
    signatures = signatures / rsum
    return {"exposures": exposures, "signatures": signatures, "gkl": gkl_new, "gklvalues": gklvalues}


def _nmfspatialbatch(
    data: np.ndarray,
    no_signatures: int,
    weight: List[scipy.sparse.spmatrix],
    batch: List[np.ndarray],
    rng: np.random.Generator,
    init_exposures: Optional[np.ndarray] = None,
    init_signatures: Optional[np.ndarray] = None,
    init_exposures_list: Optional[List[np.ndarray]] = None,
    init_signatures_list: Optional[List[np.ndarray]] = None,
    maxiter: int = 10000,
    tolerance: float = 1e-8,
    initial: int = 10,
    small_iter: int = 100,
    error_freq: int = 10,
    score_ortho: float = 0.0,
) -> Dict[str, np.ndarray]:
    init_count = initial
    if init_exposures_list is not None and init_signatures_list is not None:
        init_count = min(len(init_exposures_list), len(init_signatures_list))
    exposures, signatures, gkl_value = _nmf1(
        data,
        no_signatures,
        rng,
        init_exposures=init_exposures_list[0] if init_exposures_list else init_exposures,
        init_signatures=init_signatures_list[0] if init_signatures_list else init_signatures,
        iterations=small_iter,
    )
    for i in range(init_count):
        if i == 0:
            continue
        exp_new, sig_new, gkl_new = _nmf1(
            data,
            no_signatures,
            rng,
            init_exposures=init_exposures_list[i] if init_exposures_list else None,
            init_signatures=init_signatures_list[i] if init_signatures_list else None,
            iterations=small_iter,
        )
        if gkl_new < gkl_value:
            gkl_value = gkl_new
            exposures = exp_new
            signatures = sig_new

    estimate = exposures @ signatures
    gkl_old = _gkl_error(data.ravel(), estimate.ravel())
    gkl_new = 2 * gkl_old
    gklvalues = np.zeros(maxiter, dtype=float)

    for t in range(maxiter):
        exposures *= (data / estimate) @ signatures.T
        exposures = np.clip(exposures, EPS, None)
        estimate = exposures @ signatures

        signatures *= exposures.T @ (data / estimate)
        signatures = np.clip(signatures, EPS, None)
        signatures = _normalize_rows(signatures)
        estimate = exposures @ signatures

        exposures *= (data / estimate) @ signatures.T
        for b, batch_index in enumerate(batch):
            exposures_batch = exposures[batch_index]
            exp_sum = np.sum(exposures_batch, axis=1, keepdims=True)
            exp_sum[exp_sum == 0] = 1
            exposures_batch_norm = exposures_batch / exp_sum
            exposures_batch_norm = weight[b] @ exposures_batch_norm
            exposures_batch = exposures_batch_norm * exp_sum
            exposures[batch_index] = exposures_batch

        exposures = np.clip(exposures, EPS, None)
        exposures = _normalize_cols(exposures)
        exposures = _apply_score_ortho(exposures, score_ortho)
        estimate = exposures @ signatures

        signatures *= exposures.T @ (data / estimate)
        signatures = np.clip(signatures, EPS, None)
        estimate = exposures @ signatures

        gklvalues[t] = gkl_old
        if t % error_freq == 0:
            gkl_new = _gkl_error(data.ravel(), estimate.ravel())
            if (2 * (gkl_old - gkl_new) / (0.1 + abs(2 * gkl_new)) < tolerance) and (t > error_freq):
                break
            gkl_old = gkl_new

    rsum = np.sum(signatures, axis=1, keepdims=True)
    rsum[rsum == 0] = 1
    exposures = exposures * rsum.T
    signatures = signatures / rsum
    return {"exposures": exposures, "signatures": signatures, "gkl": gkl_new, "gklvalues": gklvalues}


def _nmfspatialbatch2(
    data: np.ndarray,
    no_signatures: int,
    weight: List[scipy.sparse.spmatrix],
    batch: List[np.ndarray],
    rng: np.random.Generator,
    init_exposures: Optional[np.ndarray] = None,
    init_signatures: Optional[np.ndarray] = None,
    maxiter: int = 10000,
    tolerance: float = 1e-8,
    error_freq: int = 10,
    score_ortho: float = 0.0,
) -> Dict[str, np.ndarray]:
    genomes, mut_types = data.shape
    if init_exposures is not None:
        exposures = init_exposures.copy()
    else:
        exposures = _randu_matrix(rng, genomes, no_signatures)
    if init_signatures is not None:
        signatures = init_signatures.copy()
    else:
        signatures = _randu_matrix(rng, no_signatures, mut_types)
    estimate = exposures @ signatures

    gkl_old = _gkl_error(data.ravel(), estimate.ravel())
    gkl_new = 2 * gkl_old
    gklvalues = np.zeros(maxiter, dtype=float)

    for t in range(maxiter):
        exposures *= (data / estimate) @ signatures.T
        exposures = np.clip(exposures, EPS, None)
        estimate = exposures @ signatures

        signatures *= exposures.T @ (data / estimate)
        signatures = np.clip(signatures, EPS, None)
        signatures = _normalize_rows(signatures)
        estimate = exposures @ signatures

        exposures *= (data / estimate) @ signatures.T
        for b, batch_index in enumerate(batch):
            exposures_batch = exposures[batch_index]
            exp_sum = np.sum(exposures_batch, axis=1, keepdims=True)
            exp_sum[exp_sum == 0] = 1
            exposures_batch_norm = exposures_batch / exp_sum
            exposures_batch_norm = weight[b] @ exposures_batch_norm
            exposures_batch = exposures_batch_norm * exp_sum
            exposures[batch_index] = exposures_batch

        exposures = np.clip(exposures, EPS, None)
        exposures = _normalize_cols(exposures)
        exposures = _apply_score_ortho(exposures, score_ortho)
        estimate = exposures @ signatures

        signatures *= exposures.T @ (data / estimate)
        signatures = np.clip(signatures, EPS, None)
        estimate = exposures @ signatures

        if t % error_freq == 0:
            gkl_new = _gkl_error(data.ravel(), estimate.ravel())
            if (2 * (gkl_old - gkl_new) / (0.1 + abs(2 * gkl_new)) < tolerance) and (t > error_freq):
                break
            gkl_old = gkl_new
        gklvalues[t] = gkl_old

    rsum = np.sum(signatures, axis=1, keepdims=True)
    rsum[rsum == 0] = 1
    exposures = exposures * rsum.T
    signatures = signatures / rsum
    return {"exposures": exposures, "signatures": signatures, "gkl": gkl_new, "gklvalues": gklvalues}


def _spectral_norm(mat: np.ndarray) -> float:
    vals = np.linalg.svd(mat, compute_uv=False)
    return float(vals[0]) if vals.size else 1.0


def _pnmf_euc_update(data: np.ndarray, weights: np.ndarray, zerotol: float) -> np.ndarray:
    xx = data @ data.T
    xx_w = xx @ weights
    scl = weights @ (weights.T @ xx_w) + (xx_w @ (weights.T @ weights))
    scl = np.maximum(scl, zerotol)
    weights = weights * (xx_w / scl)
    weights /= _spectral_norm(weights)
    weights = np.where(weights < zerotol, 0.0, weights)
    return weights


def spatial_pnmf(
    data: np.ndarray,
    no_signatures: int,
    location: Optional[np.ndarray] = None,
    lengthscale: Optional[float] = None,
    batch: Sequence[int] | int = 1,
    maxiter: int = 1000,
    tolerance: float = 1e-6,
    initial: int = 3,
    small_iter: int = 50,
    error_freq: int = 10,
    kernel_cutoff: float = 0.1,
    normalize: bool = True,
    seed: Optional[int] = None,
    zerotol: float = 1e-10,
    use_smoothing: bool = True,
    use_estimate_lengthscale: bool = False,
    lambda_smooth: float = 1.0,
    init_pca: bool = False,
    dtype: Optional[np.dtype] = None,
    fast: bool = True,
) -> Dict[str, np.ndarray]:
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")
    if np.any(np.sum(data, axis=0) == 0):
        raise ValueError("Remove columns in the data that only contain zeroes.")
    if np.any(np.sum(data, axis=1) == 0):
        raise ValueError("Remove rows in the data that only contain zeroes.")

    if fast and dtype is None:
        dtype = np.float32

    if normalize:
        data = (data / np.sum(data, axis=1, keepdims=True)) * 1e4
        print("Normalized the data, so each row sums to 10000.")
    data = np.log1p(data)
    if dtype is not None:
        data = np.ascontiguousarray(data, dtype=dtype)

    rng = _make_rng(seed)

    ls_value = lengthscale
    dist = None
    if location is not None:
        dist = dist_fun(location)
        if lengthscale is None:
            if use_estimate_lengthscale:
                est = estimate_lengthscale(data=data, dist=dist, max_avg_nn=20)
                ls_idx = int(np.argmin(est[:, 1].astype(float)))
                ls_value = float(est[ls_idx, 0])
                print(f"Lengthscale set to {ls_value} using estimate_lengthscale().")
            else:
                dist_sqrt = np.sqrt(dist)
                idx = min(14, dist_sqrt.shape[1] - 1)
                ls_value = float(np.mean(np.sort(dist_sqrt, axis=1)[:, idx]))
                print(
                    f"Lengthscale set to {ls_value}. Adjust after assessing results or use estimate_lengthscale()."
                )

    def run_with_k(k: int):
        data_used = data
        n, p = data_used.shape
        if init_pca:
            data_centered = data_used - np.mean(data_used, axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(data_centered, full_matrices=False)
            W = np.abs(vt[:k, :])
            W = np.clip(W, EPS, None)
        else:
            W = _randu_matrix(rng, k, p)

        sigma = None
        lap = None
        if location is not None and use_smoothing:
            dist_mat = dist if dist is not None else dist_fun(location)
            if ls_value is None:
                dist_sqrt = np.sqrt(dist_mat)
                idx = min(14, dist_sqrt.shape[1] - 1)
                ls = float(np.mean(np.sort(dist_sqrt, axis=1)[:, idx]))
            else:
                ls = ls_value
            sigma = np.exp(-dist_mat / (ls ** 2))
            sigma[sigma < kernel_cutoff] = 0
            sigma = sigma / np.sum(sigma, axis=1, keepdims=True)
            lap = np.eye(n, dtype=data_used.dtype) - sigma

        # Initialize scores
        S = data_used @ W.T
        S = np.clip(S, EPS, None)

        err_old = None
        for t in range(maxiter):
            # Update scores with smoothing penalty
            denom_s = S @ (W @ W.T)
            if lap is not None and lambda_smooth > 0:
                denom_s = denom_s + lambda_smooth * (lap @ S)
            denom_s = np.maximum(denom_s, zerotol)
            S = S * ((data_used @ W.T) / denom_s)
            S = np.clip(S, EPS, None)

            # Update W
            denom_w = (S.T @ S) @ W
            denom_w = np.maximum(denom_w, zerotol)
            W = W * ((S.T @ data_used) / denom_w)
            W = np.clip(W, EPS, None)

            if error_freq > 0 and t % error_freq == 0:
                estimate = S @ W
                err_new = float(np.linalg.norm(data_used - estimate))
                if err_old is not None:
                    if (2 * (err_old - err_new) / (0.1 + abs(2 * err_new)) < tolerance) and (t > error_freq):
                        break
                err_old = err_new

        return W, S

    weights, scores = run_with_k(no_signatures)
    return {
        "weights": weights,
        "signatures": weights,
        "scores": scores,
    }


def nnmf(
    data: np.ndarray,
    no_signatures: int,
    location: Optional[np.ndarray] = None,
    lengthscale: Optional[float] = None,
    batch: Sequence[int] | int = 1,
    maxiter: int = 1000,
    tolerance: float = 1e-10,
    initial: int = 3,
    small_iter: int = 50,
    error_freq: int = 10,
    kernel_cutoff: float = 0.1,
    normalize: bool = True,
    not_sc: bool = False,
    seed: Optional[int] = None,
    init_exposures: Optional[np.ndarray] = None,
    init_signatures: Optional[np.ndarray] = None,
    init_exposures_list: Optional[List[np.ndarray]] = None,
    init_signatures_list: Optional[List[np.ndarray]] = None,
    score_ortho: float = 0.0,
    dtype: Optional[np.dtype] = None,
    fast: bool = True,
) -> Dict[str, np.ndarray]:
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")
    if np.any(np.sum(data, axis=0) == 0):
        raise ValueError("Remove columns in the data that only contain zeroes.")
    if np.any(np.sum(data, axis=1) == 0):
        raise ValueError("Remove rows in the data that only contain zeroes.")

    if fast and dtype is None:
        dtype = np.float32

    if normalize:
        data = (data / np.sum(data, axis=1, keepdims=True)) * data.shape[1]
        print(f"Normalized the data, so each row sums to {data.shape[1]}.")
    if dtype is not None:
        data = np.ascontiguousarray(data, dtype=dtype)

    mean_nn = 0.0
    rng = _make_rng(seed)

    if location is None:
        print("Running regular NMF, as no locations were specified.")
        out = _nmfgen(
            data=data,
            no_signatures=no_signatures,
            rng=rng,
            init_exposures=init_exposures,
            init_signatures=init_signatures,
            init_exposures_list=init_exposures_list,
            init_signatures_list=init_signatures_list,
            maxiter=maxiter,
            tolerance=tolerance,
            initial=initial,
            small_iter=small_iter,
            error_freq=error_freq,
            score_ortho=score_ortho,
        )
    else:
        if data.shape[0] != location.shape[0]:
            raise ValueError("The number of rows in location must match the number of rows in data.")

        batch_arr = np.asarray(batch)
        unique_batches = np.unique(batch_arr)

        if unique_batches.size == 1:
            print(f"All {data.shape[0]} observations are run in one batch.")
            if data.shape[0] > 50000:
                raise ValueError("Too many observations to run in one batch. Use groupondist(size<=20000).")

            dist = dist_fun(location)
            if lengthscale is None:
                dist_sqrt = np.sqrt(dist)
                idx = min(14, dist_sqrt.shape[1] - 1)
                lengthscale = float(np.mean(np.sort(dist_sqrt, axis=1)[:, idx]))
                print(
                    f"Lengthscale set to {lengthscale}. Adjust after assessing results or use estimate_lengthscale()."
                )

            sigma = np.exp(-dist / (lengthscale ** 2))
            sigma[sigma < kernel_cutoff] = 0
            mean_nn = float(np.mean(np.sum(sigma > 0, axis=1) - 1))
            sigma = sigma / np.sum(sigma, axis=1, keepdims=True)
            weight = scipy.sparse.csr_matrix(sigma)
            out = _nmfspatial(
                data=data,
                no_signatures=no_signatures,
                weight=weight,
                rng=rng,
                init_exposures=init_exposures,
                init_signatures=init_signatures,
                init_exposures_list=init_exposures_list,
                init_signatures_list=init_signatures_list,
                maxiter=maxiter,
                tolerance=tolerance,
                initial=initial,
                small_iter=small_iter,
                error_freq=error_freq,
                score_ortho=score_ortho,
            )
        else:
            if data.shape[0] != batch_arr.shape[0]:
                raise ValueError("The length of batch must match the number of rows in data.")

            weights = []
            batch_list = []
            first_batch = True
            for b in unique_batches:
                index = np.where(batch_arr == b)[0]
                batch_list.append(index)
                dist = dist_fun(location[index])
                if lengthscale is None:
                    dist_sqrt = np.sqrt(dist)
                    idx = min(14, dist_sqrt.shape[1] - 1)
                    lengthscale = float(np.mean(np.sort(dist_sqrt, axis=1)[:, idx]))
                    print(f"Lengthscale set to {lengthscale}. Adjust after assessing results.")

                sigma = np.exp(-dist / (lengthscale ** 2))
                sigma[sigma < kernel_cutoff] = 0
                if first_batch:
                    mean_nn = float(np.mean(np.sum(sigma > 0, axis=1) - 1))
                    first_batch = False
                else:
                    mean_nn = 0.5 * mean_nn + 0.5 * float(np.mean(np.sum(sigma > 0, axis=1) - 1))

                sigma = sigma / np.sum(sigma, axis=1, keepdims=True)
                weights.append(scipy.sparse.csr_matrix(sigma))

            if initial == 1:
                out = _nmfspatialbatch2(
                    data=data,
                    no_signatures=no_signatures,
                    weight=weights,
                    batch=batch_list,
                    rng=rng,
                    init_exposures=init_exposures,
                    init_signatures=init_signatures,
                    maxiter=maxiter,
                    tolerance=tolerance,
                    error_freq=error_freq,
                    score_ortho=score_ortho,
                )
            else:
                out = _nmfspatialbatch(
                    data=data,
                    no_signatures=no_signatures,
                    weight=weights,
                    batch=batch_list,
                    rng=rng,
                    init_exposures=init_exposures,
                    init_signatures=init_signatures,
                    init_exposures_list=init_exposures_list,
                    init_signatures_list=init_signatures_list,
                    maxiter=maxiter,
                    tolerance=tolerance,
                    initial=initial,
                    small_iter=small_iter,
                    error_freq=error_freq,
                    score_ortho=score_ortho,
                )

        if not_sc:
            estimate = out["exposures"] @ out["signatures"]
            adjust = (data / np.maximum(estimate, EPS)) @ out["signatures"].T
            out["exposures"] = out["exposures"] * adjust

    return {
        "weights": out["exposures"],
        "signatures": out["signatures"],
        "error": out["gkl"],
        "errorvalues": out["gklvalues"],
        "avg_nn": mean_nn,
        "lengthscale": lengthscale,
        "batch": batch,
    }
