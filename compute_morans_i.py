import argparse
import os


def _sanitize_thread_env() -> None:
    keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            parsed = int(str(value).strip())
            if parsed <= 0:
                raise ValueError
        except Exception:
            os.environ[key] = "1"


_sanitize_thread_env()

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from utils import load_csv, save_csv

try:
    from esda.moran import Moran
    from libpysal.weights import KNN

    HAS_ESDA = True
except Exception:
    HAS_ESDA = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="dataset directory, e.g. breast-cancer-5k/")
    parser.add_argument("--locs-file", type=str, default="locs.csv")
    parser.add_argument("--cnts-file", type=str, default="cnts.csv")
    parser.add_argument("--output-file", type=str, default="morans_i.csv")
    parser.add_argument("--k", type=int, default=8, help="number of nearest neighbors")
    parser.add_argument("--block-size", type=int, default=256, help="genes per compute block")
    parser.add_argument("--symmetrize", action="store_true", help="symmetrize kNN graph before row normalization")
    return parser.parse_args()


def normalize_prefix(prefix):
    prefix = str(prefix)
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def extract_coords(locs_df):
    lower_map = {str(col).lower(): col for col in locs_df.columns}
    if "y" in lower_map and "x" in lower_map:
        return locs_df[[lower_map["y"], lower_map["x"]]].to_numpy(dtype=np.float64)

    if locs_df.shape[1] < 2:
        raise ValueError("locs.csv must contain at least two columns for coordinates.")
    return locs_df.iloc[:, :2].to_numpy(dtype=np.float64)


def build_knn_weight_matrix(coords, k=8, symmetrize=False):
    n = coords.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 spots to compute Moran's I.")

    k_eff = min(int(k), n - 1)
    if k_eff < 1:
        raise ValueError("Effective k must be at least 1.")

    tree = cKDTree(coords)
    _, nn_idx = tree.query(coords, k=k_eff + 1)
    nn_idx = np.asarray(nn_idx)[:, 1:]

    row_idx = np.repeat(np.arange(n), k_eff)
    col_idx = nn_idx.reshape(-1)
    data = np.ones(row_idx.shape[0], dtype=np.float64)

    w = coo_matrix((data, (row_idx, col_idx)), shape=(n, n)).tocsr()
    w.setdiag(0.0)
    w.eliminate_zeros()

    if symmetrize:
        w = w.maximum(w.T).tocsr()

    row_sums = np.asarray(w.sum(axis=1)).reshape(-1)
    row_sums[row_sums == 0] = 1.0
    inv_row = 1.0 / row_sums
    w = w.multiply(inv_row[:, None]).tocsr()
    s0 = float(w.sum())
    return w, s0


def compute_morans_i_for_counts(counts_df, weight_matrix, s0, block_size=256):
    x = counts_df.to_numpy(dtype=np.float64, copy=False)
    n_spots, n_genes = x.shape
    if n_spots < 2:
        raise ValueError("Need at least 2 spots to compute Moran's I.")

    out = np.full(n_genes, np.nan, dtype=np.float64)
    scale = float(n_spots) / max(s0, 1e-12)

    for start in range(0, n_genes, block_size):
        end = min(start + block_size, n_genes)
        block = x[:, start:end]
        mean = np.mean(block, axis=0, keepdims=True)
        centered = block - mean
        denom = np.sum(centered * centered, axis=0)
        wz = weight_matrix @ centered
        numer = np.sum(centered * wz, axis=0)

        valid = denom > 1e-12
        block_i = np.full(end - start, np.nan, dtype=np.float64)
        block_i[valid] = scale * numer[valid] / denom[valid]
        out[start:end] = block_i

    return out


def compute_morans_i_esda(counts_df, coords, k=8):
    w = KNN.from_array(coords, k=min(int(k), coords.shape[0] - 1))
    w.transform = "R"

    out_i = np.full(counts_df.shape[1], np.nan, dtype=np.float64)
    out_p = np.full(counts_df.shape[1], np.nan, dtype=np.float64)
    x = counts_df.to_numpy(dtype=np.float64, copy=False)

    for j in range(x.shape[1]):
        values = x[:, j]
        if np.allclose(values, values[0]):
            continue
        moran = Moran(values, w, two_tailed=True)
        out_i[j] = float(moran.I)
        out_p[j] = float(moran.p_norm)

    return out_i, out_p


def main():
    args = get_args()
    prefix = normalize_prefix(args.prefix)

    locs = load_csv(f"{prefix}{args.locs_file}")
    cnts = load_csv(f"{prefix}{args.cnts_file}")

    coords = extract_coords(locs)
    if len(coords) != len(cnts):
        raise ValueError(
            f"locs rows ({len(coords)}) and cnts rows ({len(cnts)}) must match."
        )

    if HAS_ESDA:
        morans_i, p_values = compute_morans_i_esda(
            counts_df=cnts,
            coords=coords,
            k=args.k,
        )
        method = "esda_knn_row_standardized"
    else:
        weight_matrix, s0 = build_knn_weight_matrix(
            coords=coords,
            k=args.k,
            symmetrize=args.symmetrize,
        )
        morans_i = compute_morans_i_for_counts(
            counts_df=cnts,
            weight_matrix=weight_matrix,
            s0=s0,
            block_size=max(int(args.block_size), 1),
        )
        p_values = np.full_like(morans_i, np.nan, dtype=np.float64)
        method = "manual_knn_row_standardized"

    out_df = pd.DataFrame(
        {
            "gene": cnts.columns.to_list(),
            "morans_i": morans_i,
            "p_norm": p_values,
            "method": method,
        }
    )
    out_df = out_df.sort_values("morans_i", ascending=False, na_position="last").reset_index(drop=True)
    save_csv(out_df, f"{prefix}{args.output_file}", index=False)


if __name__ == "__main__":
    main()
