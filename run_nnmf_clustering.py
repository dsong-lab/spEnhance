#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))


def _load_gene_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text().strip().splitlines()
    if not raw:
        return []
    return [line.strip().split()[0] for line in raw if line.strip()]


def reweight_features(features: np.ndarray) -> np.ndarray:
    """
    Apply R-version reweighting:
    w_ki * log(1 + w_ki / max_other)
    features: (K x G)
    """
    K, G = features.shape
    mat_new = np.zeros_like(features)

    for g in range(G):
        for k in range(K):
            w_ki = features[k, g]

            # max over other signatures
            other = np.delete(features[:, g], k)
            if other.size == 0:
                max_other = 0.0
            else:
                max_other = np.max(other)

            mat_new[k, g] = w_ki * np.log(1 + w_ki / (max_other + 1e-9))

    return mat_new


def gene_clustering(reweighted: np.ndarray, genes: list[str], k: int) -> list[list[str]]:
    """
    Replicates R clustering:
    - transpose
    - scale per gene
    - 1 - Pearson correlation
    - Ward.D2 hierarchical clustering
    - cutree
    """

    # transpose → gene x K
    gene_mat = reweighted.T  # shape: (G x K)

    # scale by gene (row-wise z-score)
    mean = gene_mat.mean(axis=1, keepdims=True)
    std = gene_mat.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    gene_mat_scaled = (gene_mat - mean) / std

    # compute correlation matrix (gene-gene)
    corr = np.corrcoef(gene_mat_scaled)

    # convert to distance
    dist = 1 - corr

    # ensure numerical stability
    np.fill_diagonal(dist, 0)

    # convert to condensed form
    dist_condensed = squareform(dist, checks=False)

    # hierarchical clustering (Ward)
    Z = linkage(dist_condensed, method="ward")

    clusters = fcluster(Z, k, criterion="maxclust")

    gene_groups = {}
    for gene, cl in zip(genes, clusters):
        gene_groups.setdefault(cl, []).append(gene)

    # sort cluster IDs for deterministic output
    sorted_groups = [gene_groups[c] for c in sorted(gene_groups.keys())]

    return sorted_groups


def save_gene_groups(gene_groups: list[list[str]], outfile: Path) -> None:
    """
    Save identical format as R:
    [
      ["gene1","gene2"],
      ["geneA","geneB"]
    ]
    """
    formatted = "["
    formatted += ", ".join(
        ["[\"" + "\",\"".join(group) + "\"]" for group in gene_groups]
    )
    formatted += "]"

    outfile.write_text(formatted)


def run(
    path: Path,
    no_signatures: int,
    k_clusters: int,
    seed: int,
    out_features: Path,
) -> None:

    from nnmf import nnmf

    print("Loading data...")

    cnts = pd.read_csv(path / "cnts.csv", index_col=0)
    locs = pd.read_csv(path / "locs.csv", index_col=0)
    gene_names = _load_gene_names(path / "gene-names.txt")

    if gene_names:
        cnts = cnts.loc[:, gene_names]

    # filter zero genes
    cnts = cnts.loc[:, cnts.sum(axis=0) > 0]

    # filter zero spots
    nonzero_rows = cnts.sum(axis=1) > 0
    cnts = cnts.loc[nonzero_rows]
    locs = locs.loc[nonzero_rows]

    print("Running NNMF...")
    res = nnmf(
        data=cnts.to_numpy(),
        no_signatures=no_signatures,
        location=locs.to_numpy(),
        not_sc=False,
        seed=seed,
        n_jobs=1,
    )

    features = res["signatures"]  # K x G
    genes = list(cnts.columns)

    # Save raw features
    out_features.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features, columns=genes).to_csv(out_features, index=False)
    print(f"Saved features → {out_features}")

    # Reweighting
    print("Reweighting features...")
    reweighted = reweight_features(features)

    # Clustering
    print("Clustering genes...")
    gene_groups = gene_clustering(reweighted, genes, k_clusters)

    # Save clustering result
    out_groups = path / "gene-names-group.txt"
    save_gene_groups(gene_groups, out_groups)

    print(f"Saved gene groups → {out_groups}")
    print("Done.")


def main():
    _ensure_repo_on_path()

    parser = argparse.ArgumentParser(description="Run NNMF + Gene Clustering (Python version of R pipeline)")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--noSignatures", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("features_py.csv"))

    args = parser.parse_args()

    run(
        path=args.path,
        no_signatures=args.noSignatures,
        k_clusters=args.k,
        seed=args.seed,
        out_features=args.out,
    )


if __name__ == "__main__":
    main()