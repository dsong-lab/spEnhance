import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))


def _load_gene_names(path: Path) -> list[str]:
    raw = path.read_text().strip().splitlines()
    if not raw:
        return []
    return [line.strip().split()[0] for line in raw if line.strip()]


def run(
    path: Path,
    no_signatures: int,
    auto_k: bool,
    k_min: int,
    k_max: int,
    seed: int,
    out: Path,
) -> None:
    from nnmf import nnmf, select_no_signatures

    cnts = pd.read_csv(path / "cnts.csv", index_col=0)
    locs = pd.read_csv(path / "locs.csv", index_col=0)
    gene_names = _load_gene_names(path / "gene-names.txt")
    if gene_names:
        cnts = cnts.loc[:, gene_names]

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
            method="nnmf",
            seed=seed,
        )
        no_signatures = int(auto["best_k"])
        print(f"[auto-k] selected noSignatures={no_signatures}")

    res = nnmf(
        data=cnts_filtered.to_numpy(),
        no_signatures=no_signatures,
        location=locs.to_numpy(),
        not_sc=False,
        seed=seed,
    )

    features = res["signatures"]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features, columns=cnts_filtered.columns).to_csv(out, index=False)
    print(f"Saved {out}")


def main() -> None:
    _ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Run NNMF to mirror run_nnmf.R")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--noSignatures", type=int, default=10)
    parser.add_argument("--auto-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("features_py.csv"))
    args = parser.parse_args()

    run(
        path=args.path,
        no_signatures=args.noSignatures,
        auto_k=args.auto_k,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()
