import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial import distance

from .core import nnmf, spatial_pnmf
from .utils import select_no_signatures


def _load_gene_names(path: Path) -> list[str]:
    raw = path.read_text().strip()
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                if data and isinstance(data[0], list):
                    return [str(g) for group in data for g in group]
                return [str(g) for g in data]
        except json.JSONDecodeError:
            pass
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines:
        return lines
    df = pd.read_csv(path, header=None, sep=None, engine="python")
    return df.iloc[:, 0].astype(str).tolist()


def run_pipeline(
    path: Path,
    no_signatures: int,
    k: int,
    seed: int | None,
    auto_k: bool,
    k_min: int,
    k_max: int,
    method: str,
) -> None:
    cnts = pd.read_csv(path / "cnts.csv", index_col=0)
    locs = pd.read_csv(path / "locs.csv", index_col=0)
    gene_names = _load_gene_names(path / "gene-names.txt")

    cnts = cnts[gene_names]
    nonzero_cols = cnts.sum(axis=0) > 0
    cnts_filtered = cnts.loc[:, nonzero_cols]
    nonzero_rows = cnts_filtered.sum(axis=1) > 0
    cnts_filtered = cnts_filtered.loc[nonzero_rows]
    locs = locs.loc[nonzero_rows]

    if auto_k:
        k_seq = list(range(int(k_min), int(k_max) + 1))
        auto = select_no_signatures(
            data=cnts_filtered.to_numpy(),
            location=locs.to_numpy(),
            k_seq=k_seq,
            method=method,
            seed=seed,
        )
        no_signatures = int(auto["best_k"])
        print(f"[auto-k] selected noSignatures={no_signatures}")

    if method == "nnmf":
        res = nnmf(
            data=cnts_filtered.to_numpy(),
            no_signatures=no_signatures,
            location=locs.to_numpy(),
            not_sc=False,
            seed=seed,
        )
    elif method == "spatial_pnmf":
        res = spatial_pnmf(
            data=cnts_filtered.to_numpy(),
            no_signatures=no_signatures,
            location=locs.to_numpy(),
            seed=seed,
        )
    else:
        raise ValueError("method must be 'nnmf' or 'spatial_pnmf'.")

    genes = list(cnts_filtered.columns)
    features = res["signatures"]
    mat = np.asarray(features)

    mat_new = np.zeros_like(mat)
    for g in range(mat.shape[1]):
        for kk in range(mat.shape[0]):
            w_ki = mat[kk, g]
            max_other = np.max(np.delete(mat[:, g], kk)) if mat.shape[0] > 1 else 0
            mat_new[kk, g] = w_ki * np.log(1 + w_ki / (max_other + 1e-9))

    gene_mat = mat_new.T
    mean = gene_mat.mean(axis=0, keepdims=True)
    std = gene_mat.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1
    gene_mat_scaled = (gene_mat - mean) / std

    corr = np.corrcoef(gene_mat_scaled, rowvar=True)
    dist_mat = 1 - corr
    condensed = distance.squareform(dist_mat, checks=False)
    hc = hierarchy.linkage(condensed, method="ward")
    clusters = hierarchy.fcluster(hc, t=k, criterion="maxclust")

    gene_list = {}
    for gene, cluster_id in zip(genes, clusters):
        gene_list.setdefault(cluster_id, []).append(gene)

    ordered = [gene_list[idx] for idx in sorted(gene_list)]
    formatted = json.dumps(ordered)
    (path / "gene-names-group.txt").write_text(formatted)
    print("Results saved!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NNMF gene clustering pipeline.")
    parser.add_argument("--path", type=Path, required=True, help="Input folder path")
    parser.add_argument("--noSignatures", type=int, default=10, help="Number of signatures for NNMF")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters for hierarchical clustering")
    parser.add_argument("--seed", type=int, default=123, help="Seed for NNMF initialization")
    parser.add_argument("--auto-k", action="store_true", help="Automatically select noSignatures")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum noSignatures for auto-k grid")
    parser.add_argument("--k-max", type=int, default=30, help="Maximum noSignatures for auto-k grid")
    parser.add_argument("--method", type=str, default="nnmf", help="Method: nnmf or spatial_pnmf")
    args = parser.parse_args()

    run_pipeline(
        path=args.path,
        no_signatures=args.noSignatures,
        k=args.k,
        seed=args.seed,
        auto_k=args.auto_k,
        k_min=args.k_min,
        k_max=args.k_max,
        method=args.method,
    )


if __name__ == "__main__":
    main()
