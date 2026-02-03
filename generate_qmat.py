#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import erf, gammaln, logsumexp


def _default_sigma_values() -> List[int]:
    return list(range(10, 71)) + [i * 2 for i in range(36, 101)]


def _parse_sigma_list(raw: Optional[str], grid: str) -> List[int]:
    if raw:
        vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not vals:
            raise ValueError("Empty --sigma-list.")
        return sorted(set(vals))
    if grid == "rctd":
        return _default_sigma_values()
    if grid == "full":
        return list(range(10, 201))
    raise ValueError(f"Unknown sigma grid: {grid}")


def _build_x_vals(x_max: float, delta: float = 1e-6) -> np.ndarray:
    if x_max <= 0:
        raise ValueError("x_max must be positive.")
    l_max = int(np.floor(np.sqrt(x_max / delta)))
    if l_max < 10:
        l_max = 10

    def m_from_l(l_val: int) -> int:
        m_r = min(l_val - 9, 40) + max(int(np.ceil(np.sqrt(max(l_val - 48.7499, 0) * 4))) - 2, 0)
        return max(m_r - 1, 0)

    x_vals = []
    prev_m = None
    for l_val in range(10, l_max + 1):
        m = m_from_l(l_val)
        if prev_m is None or m != prev_m:
            x_vals.append(delta * l_val * l_val)
            prev_m = m
    x_vals.append(delta * (l_max + 1) ** 2)
    return np.asarray(x_vals, dtype=float)


def _qmat_for_sigma(
    sigma: float,
    y_vals: np.ndarray,
    x_vals: np.ndarray,
    gh_n: int,
    *,
    progress: bool = False,
) -> np.ndarray:
    nodes, weights = hermgauss(gh_n)
    zs = np.sqrt(2.0) * nodes
    log_w = np.log(weights) - 0.5 * np.log(np.pi)
    exp_sig_z = np.exp(sigma * zs)
    log_x = np.log(x_vals)
    q_mat = np.empty((y_vals.size, x_vals.size), dtype=float)

    for yi, y in enumerate(y_vals):
        log_fact = gammaln(y + 1.0)
        log_terms = (
            log_w[None, :]
            + y * (log_x[:, None] + sigma * zs[None, :])
            - x_vals[:, None] * exp_sig_z[None, :]
            - log_fact
        )
        log_p = logsumexp(log_terms, axis=1)
        q_mat[yi, :] = -log_p
        if progress and yi % 100 == 0:
            print(f"  y={int(y)} / {int(y_vals[-1])}")
    return q_mat


def _ht_pdf_norm(x: np.ndarray) -> np.ndarray:
    a = 4.0 / 9.0 * np.exp(-9.0 / 2.0) / np.sqrt(2.0 * np.pi)
    c = 7.0 / 3.0
    C = 1.0 / ((a / (3.0 - c) - 0.5 * (1.0 + erf(-3.0 / np.sqrt(2.0)))) * 2.0 + 1.0)
    p = np.zeros_like(x, dtype=float)
    mask = np.abs(x) < 3.0
    p[mask] = C / np.sqrt(2.0 * np.pi) * np.exp(-(x[mask] ** 2) / 2.0)
    p[~mask] = C * a / (np.abs(x[~mask]) - c) ** 2
    return p


def _ht_pdf(z: np.ndarray, sigma: float) -> np.ndarray:
    x = z / sigma
    return _ht_pdf_norm(x) / sigma


def _qmat_for_sigma_rctd(
    sigma: float,
    y_vals: np.ndarray,
    x_vals: np.ndarray,
    *,
    ny: int = 5000,
    gamma: float = 0.004,
    log_output: bool = True,
    progress: bool = False,
) -> np.ndarray:
    y_grid = np.arange(-ny, ny + 1, dtype=float) * gamma
    log_p = np.log(_ht_pdf(y_grid, sigma))
    exp_y = np.exp(y_grid)
    log_x = np.log(x_vals)
    q_mat = np.empty((y_vals.size, x_vals.size), dtype=float)
    for yi, y in enumerate(y_vals):
        log_S = -np.outer(exp_y, x_vals)
        log_S = log_S + (y * y_grid)[:, None] + log_p[:, None]
        log_S = log_S - gammaln(y + 1.0)
        log_S = log_S + (y * log_x)[None, :]
        S = np.exp(log_S)
        vals = np.sum(S, axis=0) * gamma
        q_mat[yi, :] = -np.log(vals) if log_output else vals
        if progress and yi % 20 == 0:
            print(f"  y={int(y)} / {int(y_vals[-1])}")
    return q_mat


def _infer_ranges_from_counts(path: str, pad: float) -> tuple[int, float]:
    cnts = pd.read_csv(path, index_col=0)
    max_count = float(np.nanmax(cnts.to_numpy()))
    if max_count <= 0:
        raise ValueError("Counts file appears to be empty or non-positive.")
    k_max = int(np.ceil(max_count * pad))
    x_max = float(max_count * pad)
    return k_max, x_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qmat.npz for Python RCTD.")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "qmat.npz"))
    parser.add_argument("--k-max", type=int, default=None, help="Max count (k). qmat rows will be k_max+3.")
    parser.add_argument("--x-max", type=float, default=None, help="Max lambda value for x grid.")
    parser.add_argument("--from-counts", default=None, help="Counts CSV to infer k_max/x_max.")
    parser.add_argument("--pad", type=float, default=1.2, help="Padding factor when inferring from counts.")
    parser.add_argument("--sigma-list", default=None, help="Comma-separated sigma integers (e.g., 10,12,14).")
    parser.add_argument("--sigma-grid", choices=["rctd", "full"], default="rctd")
    parser.add_argument("--gh-n", type=int, default=30, help="Gauss-Hermite nodes (trade-off speed/accuracy).")
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--x-vals-csv", default=None, help="CSV file with X_vals column to use directly.")
    parser.add_argument("--method", choices=["gh", "rctd"], default="gh")
    parser.add_argument("--ny", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.004)
    parser.add_argument("--rctd-no-log", action="store_true", help="Store raw Q (no -log) for rctd method.")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.from_counts and (args.k_max is None or args.x_max is None):
        k_max, x_max = _infer_ranges_from_counts(args.from_counts, args.pad)
        args.k_max = args.k_max if args.k_max is not None else k_max
        args.x_max = args.x_max if args.x_max is not None else x_max

    if args.k_max is None:
        args.k_max = 300
    if args.x_max is None:
        args.x_max = 200.0

    sigma_vals = _parse_sigma_list(args.sigma_list, args.sigma_grid)
    y_vals = np.arange(args.k_max + 3, dtype=int)
    if args.x_vals_csv:
        x_vals = pd.read_csv(args.x_vals_csv).iloc[:, 0].to_numpy(dtype=float)
    else:
        x_vals = _build_x_vals(args.x_max, delta=args.delta)

    print(f"k_max={args.k_max} (rows={y_vals.size}), x_max={args.x_max}, x_vals={x_vals.size}")
    print(f"sigma values: {sigma_vals[0]}..{sigma_vals[-1]} (count={len(sigma_vals)})")
    print(f"output: {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    qmat_dict = {"X_vals": x_vals}
    for s in sigma_vals:
        print(f"Generating sigma={s}...")
        if args.method == "rctd":
            qmat = _qmat_for_sigma_rctd(
                sigma=s / 100.0,
                y_vals=y_vals,
                x_vals=x_vals,
                ny=args.ny,
                gamma=args.gamma,
                log_output=not args.rctd_no_log,
                progress=args.progress,
            )
        else:
            qmat = _qmat_for_sigma(
                sigma=s / 100.0, y_vals=y_vals, x_vals=x_vals, gh_n=args.gh_n, progress=args.progress
            )
        qmat_dict[str(s)] = qmat

    np.savez_compressed(args.out, **qmat_dict)
    print("Done.")


if __name__ == "__main__":
    main()
