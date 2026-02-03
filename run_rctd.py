#!/usr/bin/env python3
#!/usr/bin/env python3
"""Python RCTD runner aligned with spacexr reference construction."""

import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from rctd_core import calc_log_l_vec, decompose_full, fit, load_qmat_npz, set_likelihood_vars

UMI_MIN = 100
UMI_MAX = 20_000_000
COUNTS_MIN = 10
MIN_OBS_BULK = 10
DEFAULT_CONF_THRESH = 5.0
DEFAULT_DOUBLET_THRESH = 20.0
DEFAULT_MAX_TYPES = 4
DEFAULT_REF_N_CELLS_MAX = 10000
DEFAULT_MIN_CHANGE_BULK = 1e-4
DEFAULT_IRWLS_ITERS_BULK = 100


def _to_dense(x: np.ndarray) -> np.ndarray:
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def get_cell_type_info(
    counts: np.ndarray,
    cell_types: pd.Series,
    n_umi: pd.Series,
    *,
    ref_umi_min: int = 100,
    ref_n_cells_min: int = 25,
    ref_n_cells_max: int = 10000,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[pd.DataFrame, List[str], int]:
    """Compute cell-type mean expression (UMI-normalized) and names."""
    valid = cell_types.notna() & (cell_types != "")
    cell_types = cell_types[valid].copy()
    n_umi = n_umi[valid].copy()
    counts = counts[:, valid.to_numpy()]

    # Match R: factor() ordering (alphabetical by default).
    if pd.api.types.is_categorical_dtype(cell_types):
        categories = list(cell_types.cat.categories)
    else:
        categories = sorted(pd.unique(cell_types).tolist())
    cell_types = pd.Series(pd.Categorical(cell_types, categories=categories, ordered=False), index=cell_types.index)

    keep = n_umi >= ref_umi_min
    cell_types = cell_types[keep]
    n_umi = n_umi[keep]
    counts = counts[:, keep.to_numpy()]

    counts_per_type = cell_types.value_counts().reindex(categories, fill_value=0)
    missing = counts_per_type[counts_per_type == 0].index.tolist()
    if missing:
        print("Reference: missing cell types with no occurrences:", ", ".join(missing))

    if counts_per_type.max() > ref_n_cells_max:
        if rng is None:
            rng = np.random.RandomState()
        keep_mask = np.zeros(cell_types.shape[0], dtype=bool)
        for ct in categories:
            idx = np.where(cell_types.to_numpy() == ct)[0]
            if idx.size == 0:
                continue
            take = min(ref_n_cells_max, idx.size)
            chosen = rng.choice(idx, size=take, replace=False)
            keep_mask[chosen] = True
        cell_types = cell_types.iloc[keep_mask]
        n_umi = n_umi.iloc[keep_mask]
        counts = counts[:, keep_mask]

    counts_per_type = cell_types.value_counts().reindex(categories, fill_value=0)
    if counts_per_type.min() < ref_n_cells_min:
        raise ValueError(
            f"Reference: need at least {ref_n_cells_min} cells per cell type; "
            f"min observed = {counts_per_type.min()}."
        )

    def cell_mean(ct: str) -> np.ndarray:
        idx = (cell_types == ct).to_numpy()
        mat = counts[:, idx]
        umi = n_umi[idx].to_numpy()
        norm = mat / umi
        return norm.sum(axis=1) / norm.shape[1]

    means = np.column_stack([cell_mean(ct) for ct in categories])
    return pd.DataFrame(means, index=None, columns=categories), categories, len(categories)


def get_de_genes(
    cell_type_means: pd.DataFrame,
    cell_type_names: List[str],
    puck_counts: pd.DataFrame,
    fc_thresh: float = 1.25,
    expr_thresh: float = 0.00015,
    min_obs: int = 3,
) -> List[str]:
    """Select DE genes consistent with spacexr thresholds."""
    epsilon = 1e-9
    bulk_vec = puck_counts.sum(axis=1)
    gene_list = cell_type_means.index.tolist()
    prev_num_genes = min(len(gene_list), bulk_vec.shape[0])
    if any(g.startswith("mt-") for g in gene_list):
        print("get_de_genes: Filtering out mitochondrial genes.")
        gene_list = [g for g in gene_list if not g.startswith("mt-")]
    gene_list = [g for g in gene_list if g in bulk_vec.index]
    if len(gene_list) == 0:
        raise ValueError("get_de_genes: 0 common genes between SpatialRNA and Reference.")
    gene_list = [g for g in gene_list if bulk_vec[g] >= min_obs]
    if len(gene_list) < 0.1 * prev_num_genes:
        raise ValueError(
            "get_de_genes: at least 90% of genes do not match between SpatialRNA and Reference."
        )

    total_gene_idx = set()
    for ct in cell_type_names:
        other_types = [c for c in cell_type_names if c != ct]
        other_mean = cell_type_means.loc[gene_list, other_types].mean(axis=1)
        logfc = np.log(cell_type_means.loc[gene_list, ct] + epsilon) - np.log(other_mean + epsilon)
        type_gene_list = np.where(
            (logfc > fc_thresh) & (cell_type_means.loc[gene_list, ct] > expr_thresh)
        )[0]
        total_gene_idx.update(type_gene_list.tolist())

    if len(total_gene_idx) < 10:
        raise ValueError("Fewer than 10 differentially expressed genes found")
    return [gene_list[i] for i in sorted(total_gene_idx)]


def get_norm_ref(
    puck_counts: pd.DataFrame,
    cell_type_means: pd.DataFrame,
    gene_list: List[str],
    proportions: np.ndarray,
    total_umi: float,
) -> pd.DataFrame:
    """Renormalize reference to match bulk composition."""
    bulk_vec = puck_counts.sum(axis=1)
    weight_avg = (cell_type_means.loc[gene_list].to_numpy() * (proportions / proportions.sum())).sum(axis=1)
    target_means = bulk_vec.loc[gene_list].to_numpy() / float(total_umi)
    renorm = cell_type_means.loc[gene_list].to_numpy() / (weight_avg / target_means)[:, None]
    return pd.DataFrame(renorm, index=gene_list, columns=cell_type_means.columns)


def choose_sigma_c(
    puck_counts: pd.DataFrame,
    gene_list_reg: List[str],
    cell_type_means: pd.DataFrame,
    n_umi: pd.Series,
    qmat_all: Dict[str, np.ndarray],
    x_vals: np.ndarray,
    qmat_mode: str,
    rng: Optional[np.random.RandomState] = None,
    fit_idx: Optional[List[str]] = None,
    n_fit: int = 1000,
    n_epoch: int = 8,
    umi_min_sigma: int = 300,
) -> Tuple[float, List[str]]:
    """Choose sigma using spacexr-compatible heuristic."""
    sigma = 100
    sigma_ind = list(range(10, 71)) + [i * 2 for i in range(36, 101)]
    mult_fac_vec = [i / 10 for i in range(8, 13)]

    fit_spots = n_umi[n_umi > umi_min_sigma]
    if fit_spots.shape[0] == 0:
        raise ValueError("choose_sigma_c: no spots above UMI_min_sigma.")
    if fit_idx is None:
        if rng is None:
            rng = np.random.RandomState()
        fit_idx = rng.choice(fit_spots.index, size=min(n_fit, fit_spots.shape[0]), replace=False).tolist()

    counts = puck_counts.loc[gene_list_reg, fit_idx].to_numpy()
    n_umi_fit = n_umi.loc[fit_idx].to_numpy()

    for _ in range(n_epoch):
        cache = set_likelihood_vars(qmat_all[str(sigma)], x_vals, qmat_mode=qmat_mode)
        weights = []
        for i in range(counts.shape[1]):
            res = decompose_full(
                cell_type_means.loc[gene_list_reg].to_numpy() * n_umi_fit[i],
                float(n_umi_fit[i]),
                counts[:, i],
                cache,
                constrain=False,
                min_change=1e-3,
            )
            weights.append(res["weights"])
        weights = np.vstack(weights)
        prediction = cell_type_means.loc[gene_list_reg].to_numpy() @ weights.T
        prediction = prediction * n_umi_fit
        # Match R's column-major vectorization in chooseSigma.
        pred_vec = prediction.ravel(order="F")
        pred_vec = np.maximum(pred_vec, 1e-4)
        count_vec = counts.ravel(order="F")
        num_sample = min(1_000_000, pred_vec.shape[0])
        if num_sample < pred_vec.shape[0]:
            if rng is None:
                rng = np.random.RandomState()
            use_ind = rng.choice(pred_vec.shape[0], size=num_sample, replace=False)
            pred_vec = pred_vec[use_ind]
            count_vec = count_vec[use_ind]

        si = sigma_ind.index(int(round(sigma))) if int(round(sigma)) in sigma_ind else 0
        sigma_window = sigma_ind[max(0, si - 8) : min(si + 9, len(sigma_ind))]

        def score_for_sigma(sig):
            cache_sig = set_likelihood_vars(qmat_all[str(sig)], x_vals, qmat_mode=qmat_mode)
            best = None
            for mult in mult_fac_vec:
                val = calc_log_l_vec(pred_vec * mult, count_vec, cache_sig)
                best = val if best is None else min(best, val)
            return best

        score_vec = {s: score_for_sigma(s) for s in sigma_window}
        new_sigma = min(score_vec, key=score_vec.get)
        if new_sigma == sigma:
            break
        sigma = new_sigma
    return sigma / 100.0, fit_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_h5ad")
    parser.add_argument("cnts_csv")
    parser.add_argument("locs_csv")
    parser.add_argument("outdir")
    parser.add_argument("num_cores", type=int)
    parser.add_argument("--cell-type-col", default="celltype")
    parser.add_argument("--max-spots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--reference-csv", type=str, default=None)
    parser.add_argument("--cell-type-means-csv", type=str, default=None)
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--ref-n-cells-max", type=int, default=DEFAULT_REF_N_CELLS_MAX)
    parser.add_argument("--gene-list-reg", default=None)
    parser.add_argument("--gene-list-bulk", default=None)
    parser.add_argument("--reference-barcodes", default=None)
    parser.add_argument("--fit-indices", default=None)
    parser.add_argument("--constrain", action="store_true")
    parser.add_argument("--max-types", type=int, default=DEFAULT_MAX_TYPES)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONF_THRESH)
    parser.add_argument("--doublet-threshold", type=float, default=DEFAULT_DOUBLET_THRESH)
    parser.add_argument("--solver", type=str, default="quadprog")
    parser.add_argument("--debug-spot", type=str, default=None)
    parser.add_argument("--debug-out", type=str, default=None)
    parser.add_argument("--confidence-filter", action="store_true")
    parser.add_argument("--qmat", type=str, default=None)
    parser.add_argument("--strict-float64", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dump-intermediate", action="store_true")
    parser.add_argument("--compare-to-r", type=str, default=None)
    parser.add_argument(
        "--qmat-mode",
        type=str,
        default="auto",
        choices=["neglog", "log", "prob", "raw", "auto"],
        help="Input qmat encoding: neglog (-log p), log (log p), prob (p), raw (no conversion), auto (heuristic).",
    )
    parser.add_argument(
        "--initial-weight-thresh",
        type=float,
        default=0.01,
        help="Minimum weight from full model to keep cell type as candidate.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed) if args.seed is not None else np.random.RandomState()
    if args.deterministic and args.seed is not None:
        np.random.seed(args.seed)

    print("Loading single-cell reference...")
    adata = ad.read_h5ad(args.reference_h5ad)
    if "counts" in adata.layers:
        counts = _to_dense(adata.layers["counts"])
    else:
        counts = _to_dense(adata.X)
    counts = counts.T  # genes x cells
    if args.strict_float64:
        counts = counts.astype(np.float64, copy=False)
    counts_full = counts
    genes = adata.var_names.to_list()
    genes_full = genes
    cells = adata.obs_names.to_list()
    obs = adata.obs
    if args.cell_type_col not in obs.columns:
        raise ValueError(f"Missing cell type column: {args.cell_type_col}")
    cell_types = obs[args.cell_type_col]
    n_umi = obs["nUMI"] if "nUMI" in obs.columns else pd.Series(counts.sum(axis=0), index=cells)
    if args.strict_float64:
        n_umi = n_umi.astype(np.float64)
    cell_types_full = cell_types.copy()
    if args.reference_barcodes:
        with open(args.reference_barcodes) as f:
            keep_barcodes = {line.strip() for line in f if line.strip()}
        keep_mask = np.array([c in keep_barcodes for c in cells])
        counts = counts[:, keep_mask]
        cell_types = cell_types.iloc[keep_mask]
        n_umi = n_umi.iloc[keep_mask]
        cells = list(np.array(cells)[keep_mask])
        obs = obs.iloc[keep_mask]

    print("Loading spatial transcriptomics data...")
    cnts = pd.read_csv(args.cnts_csv, index_col=0)
    locs = pd.read_csv(args.locs_csv, index_col=0)

    var_genes = cnts.var(axis=0)
    cnts = cnts.loc[:, var_genes > 0]
    cnts = cnts.T  # genes x spots
    cnts = cnts.dropna(axis=0, how="any")
    if args.strict_float64:
        cnts = cnts.astype(np.float64)
    if not all(locs.index == cnts.columns):
        raise ValueError("Spatial coords and counts do not align.")
    cnts_all_genes = cnts.copy()

    print("Computing cell type profiles...")
    if args.reference_csv:
        ref_df = pd.read_csv(args.reference_csv, index_col=0)
        ct_names = ref_df.index.tolist()
        ct_means = ref_df.T
        genes = ct_means.index.tolist()
    elif args.cell_type_means_csv:
        ct_means = pd.read_csv(args.cell_type_means_csv, index_col=0)
        ct_names = ct_means.columns.tolist()
        genes = ct_means.index.tolist()
    else:
        ct_means, ct_names, _ = get_cell_type_info(
            counts,
            cell_types,
            n_umi,
            ref_n_cells_max=args.ref_n_cells_max,
            rng=rng,
        )
        ct_means.index = genes
    if args.strict_float64:
        ct_means = ct_means.astype(np.float64)

    print("Filtering spatial spots...")
    umi_min = UMI_MIN
    umi_max = UMI_MAX
    counts_min = COUNTS_MIN
    n_umi_spots_all = cnts.sum(axis=0)
    keep_spots_all = (n_umi_spots_all >= umi_min) & (n_umi_spots_all <= umi_max) & (n_umi_spots_all >= counts_min)
    cnts_restricted = cnts.loc[:, keep_spots_all]

    print("Selecting DE genes...")
    if args.gene_list_reg:
        with open(args.gene_list_reg) as f:
            gene_list_reg = [line.strip() for line in f if line.strip()]
    else:
        gene_list_reg = get_de_genes(ct_means, ct_names, cnts_restricted, fc_thresh=0.75, expr_thresh=0.0002, min_obs=3)
    if args.gene_list_bulk:
        with open(args.gene_list_bulk) as f:
            gene_list_bulk = [line.strip() for line in f if line.strip()]
    else:
        gene_list_bulk = get_de_genes(ct_means, ct_names, cnts_restricted, fc_thresh=0.5, expr_thresh=0.000125, min_obs=3)

    counts_tot = cnts_restricted.loc[gene_list_bulk].sum(axis=0)
    n_umi_spots = cnts_restricted.sum(axis=0)
    keep_spots = (n_umi_spots >= umi_min) & (n_umi_spots <= umi_max) & (counts_tot >= counts_min)
    cnts = cnts_restricted.loc[:, keep_spots]
    locs = locs.loc[cnts.columns]
    n_umi_spots = n_umi_spots.loc[cnts.columns]

    if args.max_spots is not None and cnts.shape[1] > args.max_spots:
        chosen = rng.choice(cnts.columns.to_numpy(), size=args.max_spots, replace=False)
        cnts = cnts.loc[:, chosen]
        locs = locs.loc[chosen]
        n_umi_spots = n_umi_spots.loc[chosen]

    print("Normalizing reference...")
    qmat_all, x_vals = load_qmat_npz(args.qmat)
    bulk_counts_full = cnts.loc[gene_list_bulk]
    bulk_vec_series = bulk_counts_full.sum(axis=1)
    bulk_gene_list = [g for g in gene_list_bulk if bulk_vec_series[g] >= MIN_OBS_BULK]
    bulk_counts_fit = bulk_counts_full.loc[bulk_gene_list]
    bulk_vec = bulk_vec_series.loc[bulk_gene_list].to_numpy()
    if args.skip_normalize:
        ct_means_renorm = ct_means
    else:
        bulk_X = ct_means.loc[bulk_gene_list].to_numpy() * n_umi_spots.sum()
        cache = set_likelihood_vars(qmat_all["100"], x_vals, qmat_mode=args.qmat_mode)
        bulk_res = decompose_full(
            bulk_X,
            float(n_umi_spots.sum()),
            bulk_vec,
            cache,
            constrain=False,
            bulk_mode=True,
            n_iter=DEFAULT_IRWLS_ITERS_BULK,
            min_change=DEFAULT_MIN_CHANGE_BULK,
        )
        proportions = bulk_res["weights"]
        ct_means_renorm = get_norm_ref(bulk_counts_full, ct_means, gene_list_bulk, proportions, total_umi=float(n_umi_spots.sum()))
    if args.strict_float64:
        ct_means_renorm = ct_means_renorm.astype(np.float64)

    print("Choosing sigma...")
    fit_idx = None
    if args.fit_indices:
        with open(args.fit_indices) as f:
            fit_idx = [line.strip() for line in f if line.strip()]
    if args.sigma is not None:
        sigma = args.sigma
        fit_idx_used = fit_idx
    else:
        sigma, fit_idx_used = choose_sigma_c(
            cnts,
            gene_list_reg,
            ct_means_renorm,
            n_umi_spots,
            qmat_all,
            x_vals,
            args.qmat_mode,
            rng=rng,
            fit_idx=fit_idx,
        )

    print("Running Python RCTD...")
    reference = {
        "cell_type_means": ct_means_renorm.loc[gene_list_reg].to_numpy(dtype=np.float64),
        "gene_names": gene_list_reg,
        "cell_type_names": ct_names,
    }
    spatial = {
        "counts": cnts.loc[gene_list_reg].to_numpy(dtype=np.float64),
        "gene_names": gene_list_reg,
        "spot_names": cnts.columns.to_list(),
    }
    if args.strict_float64:
        n_umi_spots = n_umi_spots.astype(np.float64)
    if args.confidence_filter and args.num_cores != 1:
        print("confidence-filter with num_cores>1: running a second single-core pass for conf_list.")

    fit_out = fit(
        reference,
        spatial,
        qmat_path=args.qmat,
        qmat_mode=args.qmat_mode,
        sigma=sigma,
        gene_list=gene_list_reg,
        n_jobs=args.num_cores,
        constrain=args.constrain,
        max_types=args.max_types,
        confidence_threshold=args.confidence_threshold,
        doublet_threshold=args.doublet_threshold,
        solver=args.solver,
        initial_weight_thresh=args.initial_weight_thresh,
        n_umi=n_umi_spots.to_numpy(),
        debug_spot=args.debug_spot if args.num_cores == 1 else None,
        debug_out=args.debug_out if args.num_cores == 1 else None,
        return_conf=False,
    )
    weights, cell_type_names, spot_names = fit_out

    conf_lists = None
    if args.confidence_filter:
        fit_out_conf = fit(
            reference,
            spatial,
            qmat_path=args.qmat,
            qmat_mode=args.qmat_mode,
            sigma=sigma,
            gene_list=gene_list_reg,
            n_jobs=1,
            constrain=args.constrain,
            max_types=args.max_types,
            confidence_threshold=args.confidence_threshold,
            doublet_threshold=args.doublet_threshold,
            solver=args.solver,
            initial_weight_thresh=args.initial_weight_thresh,
            n_umi=n_umi_spots.to_numpy(),
            debug_spot=args.debug_spot,
            debug_out=args.debug_out,
            return_conf=True,
        )
        _, _, _, conf_lists = fit_out_conf

    weights_df = pd.DataFrame(weights, index=spot_names, columns=cell_type_names)
    if args.confidence_filter and conf_lists is not None:
        weights_arr = weights_df.to_numpy()
        for i, conf in enumerate(conf_lists):
            conf_idx = [idx for idx, ok in conf.items() if ok]
            if len(conf_idx) == 0:
                weights_arr[i, :] = 0.0
                continue
            mask = np.zeros(weights_arr.shape[1], dtype=bool)
            mask[conf_idx] = True
            weights_arr[i, ~mask] = 0.0
            s = weights_arr[i, :].sum()
            if s > 0:
                weights_arr[i, :] = weights_arr[i, :] / s
        weights_df.iloc[:, :] = weights_arr
    common_spots = locs.index.intersection(weights_df.index)
    weights_df = weights_df.loc[common_spots]
    locs = locs.loc[common_spots]

    weights_df.to_csv(os.path.join(args.outdir, "proportion_celltype.csv"))
    locs.to_csv(os.path.join(args.outdir, "locs_celltype.csv"))

    # Build reference matrix to mirror spacexr: mean normalized expression per cell type.
    if args.reference_csv:
        ref_df = pd.read_csv(args.reference_csv, index_col=0)
        ref_df = ref_df.loc[:, ref_df.columns.isin(cnts_all_genes.index)]
    else:
        # If we loaded precomputed means, use them directly.
        if args.cell_type_means_csv:
            ref_df = pd.read_csv(args.cell_type_means_csv, index_col=0)
            if set(ref_df.columns).issuperset(set(ct_names)) and not set(ref_df.index).issuperset(set(ct_names)):
                ref_df = ref_df.T
            ref_df = ref_df.loc[:, ref_df.columns.isin(cnts_all_genes.index)]
        else:
            # Mirror run_rctd.R: raw mean counts per cell type.
            valid_cells = cell_types_full.notna() & (cell_types_full != "")
            ct_series = cell_types_full[valid_cells]
            expr = counts_full[:, valid_cells.to_numpy()]
            ct_order = pd.unique(ct_series)
            ref_cols = []
            for ct in ct_order:
                mask = (ct_series == ct).to_numpy()
                ref_cols.append(expr[:, mask].mean(axis=1))
            ref_matrix = np.vstack(ref_cols)
            ref_df = pd.DataFrame(ref_matrix, index=ct_order, columns=genes_full)
            ref_df = ref_df.loc[:, ref_df.columns.isin(cnts_all_genes.index)]
    ref_df.to_csv(os.path.join(args.outdir, "reference.csv"))

    if args.dump_intermediate:
        dump_dir = os.path.join(args.outdir, "intermediate")
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "gene_list_reg.txt"), "w") as f:
            f.write("\n".join(gene_list_reg) + "\n")
        with open(os.path.join(dump_dir, "gene_list_bulk.txt"), "w") as f:
            f.write("\n".join(gene_list_bulk) + "\n")
        if fit_idx_used:
            with open(os.path.join(dump_dir, "fit_indices.txt"), "w") as f:
                f.write("\n".join(fit_idx_used) + "\n")
        with open(os.path.join(dump_dir, "sigma.txt"), "w") as f:
            f.write(f"{sigma}\n")
        pd.Series(n_umi_spots).to_csv(os.path.join(dump_dir, "n_umi_spots.csv"))
        ct_means.to_csv(os.path.join(dump_dir, "ct_means_raw.csv"))
        ct_means_renorm.to_csv(os.path.join(dump_dir, "ct_means_renorm.csv"))
        bulk_vec_series.to_csv(os.path.join(dump_dir, "bulk_vec_full.csv"))
        bulk_counts_full.to_csv(os.path.join(dump_dir, "bulk_counts_full.csv"))

    if args.compare_to_r:
        import json
        r_dir = args.compare_to_r
        compare = {}
        r_ref = pd.read_csv(os.path.join(r_dir, "reference.csv"), index_col=0)
        compare["reference_shape"] = [int(x) for x in r_ref.shape]
        compare["reference_diff_mean_abs"] = float(
            (ref_df.reindex_like(r_ref) - r_ref).abs().to_numpy().mean()
        )
        r_prop = pd.read_csv(os.path.join(r_dir, "proportion_celltype.csv"), index_col=0)
        r_al, p_al = r_prop.align(weights_df, join="inner", axis=0)
        r_al, p_al = r_al.align(p_al, join="inner", axis=1)
        compare["proportion_shape"] = [int(x) for x in p_al.shape]
        diff = (p_al - r_al).abs()
        compare["proportion_mean_abs"] = float(diff.to_numpy().mean())
        compare["proportion_max_abs"] = float(diff.to_numpy().max())
        compare["proportion_spots_diff_gt_1e-6"] = int((diff.max(axis=1) > 1e-6).sum())
        with open(os.path.join(args.outdir, "compare_to_r.json"), "w") as f:
            json.dump(compare, f, indent=2)

    print("Done. Results saved to:", args.outdir)


if __name__ == "__main__":
    main()
