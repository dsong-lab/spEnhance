import argparse
import gc
import math
import os
import pickle
import shutil
import tempfile


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
from PIL import Image
from sklearn.decomposition import IncrementalPCA, NMF, TruncatedSVD
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None


def load_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def save_pickle(x, filename):
    with open(filename, "wb") as file:
        pickle.dump(x, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved", filename)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix", type=str)

    parser.add_argument(
        "--mode",
        type=str,
        default="combined",
        choices=["combined", "uncombined"],
    )

    parser.add_argument(
        "--normalize",
        type=str,
        default="celltype",
        choices=["none", "celltype", "gene-zscore", "gene-sum"],
    )

    # ============================================================
    # Expression embedding dimension selection:
    # Choose either fixed dimension OR target variance/reconstruction ratio.
    # ============================================================
    dim_group = parser.add_mutually_exclusive_group()

    dim_group.add_argument(
        "--expr-dim",
        type=int,
        default=None,
        help=(
            "Manually set the number of expression embedding dimensions/components. "
            "Cannot be used together with --expr-target-variance."
        ),
    )

    dim_group.add_argument(
        "--expr-target-variance",
        type=float,
        default=None,
        help=(
            "Manually set the target retained variance/reconstruction ratio for expression embedding. "
            "For IncrementalPCA/TruncatedSVD, this means cumulative explained variance ratio. "
            "For NMF, this means reconstruction ratio. Example: 0.95 or 0.99. "
            "Cannot be used together with --expr-dim."
        ),
    )

    parser.add_argument(
        "--max-expr-dim",
        type=int,
        default=256,
        help=(
            "Maximum number of expression components to fit when using --expr-target-variance. "
            "Also used as the default fixed expression dimension if neither --expr-dim nor "
            "--expr-target-variance is provided."
        ),
    )

    parser.add_argument(
        "--svd-method",
        type=str,
        default="nmf",
        choices=["incremental-pca", "truncated-svd", "nmf"],
    )

    parser.add_argument(
        "--block_h",
        type=int,
        default=64,
        help="height block size for block-wise computation",
    )

    parser.add_argument(
        "--mask-file",
        type=str,
        default=None,
        help="optional mask file; if omitted, will auto-use prefix/mask-small.png when present",
    )

    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="force using all pixels and ignore any mask file",
    )

    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="directory to store temporary memmap files",
    )

    parser.add_argument("--keep-temp", action="store_true")

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )

    args = parser.parse_args()
    return args


def list_pickle_files(pickle_dir):
    files = [f for f in os.listdir(pickle_dir) if f.endswith(".pickle")]
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No .pickle files found in {pickle_dir}")
    return files


def infer_hw_from_first_pickle(pickle_dir, files):
    arr = load_pickle(os.path.join(pickle_dir, files[0]))
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Probability map should be 2D, got shape={arr.shape}")
    h, w = arr.shape
    del arr
    gc.collect()
    return h, w


def prepare_prob_memmap(pickle_dir, files, workdir, dtype=np.float32):
    h, w = infer_hw_from_first_pickle(pickle_dir, files)
    n = len(files)

    prob_path = os.path.join(workdir, "celltype_prob_maps.dat")
    prob_maps = np.memmap(prob_path, mode="w+", dtype=dtype, shape=(n, h, w))

    celltypes = []
    for i, fname in enumerate(tqdm(files, desc="Loading probability maps to memmap")):
        celltype = os.path.splitext(fname)[0]
        arr = load_pickle(os.path.join(pickle_dir, fname))
        arr = np.asarray(arr, dtype=dtype)
        if arr.shape != (h, w):
            raise ValueError(f"Inconsistent shape for {fname}: got {arr.shape}, expected {(h, w)}")
        prob_maps[i] = arr
        celltypes.append(celltype)
        del arr
        gc.collect()

    prob_maps.flush()
    return prob_maps, np.array(celltypes), h, w


def build_pred_ids_streaming(prob_maps, h, w, block_h):
    pred_ids = np.memmap(
        prob_maps.filename.replace("celltype_prob_maps.dat", "pred_ids.dat"),
        mode="w+",
        dtype=np.int32,
        shape=(h, w),
    )

    n_blocks = math.ceil(h / block_h)
    for b in tqdm(range(n_blocks), desc="Computing argmax celltype IDs"):
        h0 = b * block_h
        h1 = min((b + 1) * block_h, h)
        block = prob_maps[:, h0:h1, :]
        pred_ids[h0:h1, :] = np.argmax(block, axis=0).astype(np.int32)

    pred_ids.flush()
    return pred_ids


def load_reference(proportion_csv, celltypes, dtype=np.float32):
    ref_matrix = pd.read_csv(proportion_csv, index_col=0)
    genes = ref_matrix.columns.to_list()

    missing = [ct for ct in celltypes if ct not in ref_matrix.index]
    if missing:
        raise ValueError(f"These celltypes do not exist in reference matrix: {missing}")

    ref_matrix_np = ref_matrix.loc[celltypes].to_numpy(dtype=dtype)
    return ref_matrix_np, genes


def normalize_reference_by_celltype(ref_matrix_np):
    row_sums = ref_matrix_np.sum(axis=1, keepdims=True) + 1e-8
    return ref_matrix_np / row_sums * 1e4


def fit_reference_nmf(ref_matrix_np, dim, target_variance=None, random_state=42):
    x = np.asarray(ref_matrix_np, dtype=np.float32)
    if np.any(x < 0):
        raise ValueError("NMF requires nonnegative input. Use --normalize none or --normalize celltype.")

    max_components = min(int(dim), x.shape[0], x.shape[1])
    if max_components < 1:
        raise ValueError("No valid NMF components available.")

    total_ss = float(np.square(x).sum()) + 1e-12
    records = []
    chosen_model = None
    chosen_w = None
    chosen_k = None

    k_seq = [max_components]
    if target_variance is not None:
        k_seq = list(range(1, max_components + 1))

    for k in k_seq:
        model = NMF(
            n_components=int(k),
            init="nndsvda",
            random_state=random_state,
            max_iter=500,
        )
        w_ref = model.fit_transform(x)
        recon = w_ref @ model.components_
        recon_ratio = 1.0 - float(np.square(x - recon).sum()) / total_ss

        records.append(
            {
                "k": int(k),
                "reconstruction_ratio": float(recon_ratio),
            }
        )

        chosen_model = model
        chosen_w = w_ref.astype(np.float32, copy=False)
        chosen_k = int(k)

        if target_variance is not None and recon_ratio >= float(target_variance):
            break

    return chosen_model, chosen_w, chosen_k, records


def select_components_by_explained_variance(
    explained_variance_ratio,
    requested_dim,
    target_variance=None,
):
    evr = np.asarray(explained_variance_ratio, dtype=np.float64)

    if evr.ndim != 1 or evr.size == 0:
        raise ValueError("explained_variance_ratio must be a non-empty 1D array.")

    max_components = int(min(int(requested_dim), evr.size))
    if max_components < 1:
        raise ValueError("No valid components available.")

    if target_variance is None:
        chosen = max_components
    else:
        threshold = float(target_variance)
        if not (0.0 < threshold <= 1.0):
            raise ValueError("--expr-target-variance must be within (0, 1].")

        cum = np.cumsum(evr[:max_components])
        chosen = int(np.searchsorted(cum, threshold, side="left") + 1)
        chosen = min(chosen, max_components)

    records = []
    cum_full = np.cumsum(evr[:max_components])
    for i in range(max_components):
        records.append(
            {
                "k": int(i + 1),
                "explained_variance_ratio": float(evr[i]),
                "cumulative_explained_variance_ratio": float(cum_full[i]),
            }
        )

    return chosen, records


def transform_nmf_streaming(
    mode,
    prob_maps,
    ref_latent,
    h,
    w,
    block_h,
    pred_ids,
    output_pickle,
):
    n_components = ref_latent.shape[1]
    out = np.zeros((n_components, h, w), dtype=np.float32)

    for h0, h1 in tqdm(iter_block_ranges(h, block_h), desc="Building NMF latent map"):
        bh = h1 - h0

        if mode == "combined":
            n_celltype = ref_latent.shape[0]
            prob_block = np.asarray(prob_maps[:, h0:h1, :], dtype=np.float32)
            block_flat = prob_block.reshape(n_celltype, -1)
            latent_flat = (ref_latent.T @ block_flat).T
            del prob_block, block_flat
        else:
            ids_block = np.asarray(pred_ids[h0:h1, :], dtype=np.int32).reshape(-1)
            latent_flat = ref_latent[ids_block]
            del ids_block

        out[:, h0:h1, :] = latent_flat.reshape(bh, w, n_components).transpose(2, 0, 1)

        del latent_flat
        gc.collect()

    save_pickle(out, output_pickle)
    return out


def resolve_mask(prefix, mask_file, h, w, no_mask=False):
    if no_mask:
        mask = np.ones((h, w), dtype=bool)
        print("Mask: forced to use all pixels")
        return mask

    if mask_file is not None:
        if str(mask_file).strip().lower() in {"", "none", "null"}:
            mask = np.ones((h, w), dtype=bool)
            print("Mask: using all pixels")
            return mask
        candidate = mask_file
    else:
        auto_candidate = os.path.join(prefix, "mask-small.png")
        candidate = auto_candidate if os.path.exists(auto_candidate) else None

    if candidate is None:
        mask = np.ones((h, w), dtype=bool)
        print("Mask: using all pixels")
        return mask

    mask_img = np.array(Image.open(candidate))
    if mask_img.ndim == 3:
        mask = np.any(mask_img > 0, axis=2)
    else:
        mask = mask_img > 0

    if mask.shape != (h, w):
        raise ValueError(f"Mask shape mismatch: got {mask.shape}, expected {(h, w)}")

    print(f"Mask loaded from {candidate}, valid pixels: {int(mask.sum())}/{h * w}")
    return mask.astype(bool)


def iter_block_ranges(h, block_h):
    for h0 in range(0, h, block_h):
        yield h0, min(h0 + block_h, h)


def get_flat_block(
    mode,
    prob_maps,
    ref_matrix_np,
    h0,
    h1,
    pred_ids=None,
    out_dtype=np.float32,
):
    if mode == "combined":
        n_celltype = ref_matrix_np.shape[0]
        prob_block = np.asarray(prob_maps[:, h0:h1, :], dtype=np.float32)
        block_flat = prob_block.reshape(n_celltype, -1)
        gene_flat = (ref_matrix_np.T @ block_flat).T
        del prob_block, block_flat
        return gene_flat.astype(out_dtype, copy=False)

    ids_block = np.asarray(pred_ids[h0:h1, :], dtype=np.int32).reshape(-1)
    gene_flat = ref_matrix_np[ids_block]
    del ids_block
    return gene_flat.astype(out_dtype, copy=False)


def compute_gene_stats_streaming(
    mode,
    prob_maps,
    ref_matrix_np,
    h,
    w,
    block_h,
    valid_mask,
    normalize_mode,
    pred_ids=None,
):
    c_gene = ref_matrix_np.shape[1]
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        raise ValueError("Mask has zero valid pixels.")

    if normalize_mode == "gene-zscore":
        gene_sum = np.zeros((c_gene,), dtype=np.float64)
        gene_sq_sum = np.zeros((c_gene,), dtype=np.float64)

        for h0, h1 in tqdm(iter_block_ranges(h, block_h), desc="Pass 1/3: compute mean/std"):
            block = get_flat_block(
                mode,
                prob_maps,
                ref_matrix_np,
                h0,
                h1,
                pred_ids=pred_ids,
                out_dtype=np.float32,
            )
            block_mask = valid_mask[h0:h1, :].reshape(-1)

            if np.any(block_mask):
                block = np.asarray(block[block_mask], dtype=np.float64)
                gene_sum += block.sum(axis=0)
                gene_sq_sum += np.square(block).sum(axis=0)

            del block, block_mask
            gc.collect()

        mean = gene_sum / n_valid
        var = gene_sq_sum / n_valid - np.square(mean)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var) + 1e-8

        return mean.astype(np.float32), std.astype(np.float32)

    if normalize_mode == "gene-sum":
        gene_sum = np.zeros((c_gene,), dtype=np.float64)

        for h0, h1 in tqdm(iter_block_ranges(h, block_h), desc="Pass 1/3: compute gene sums"):
            block = get_flat_block(
                mode,
                prob_maps,
                ref_matrix_np,
                h0,
                h1,
                pred_ids=pred_ids,
                out_dtype=np.float32,
            )
            block_mask = valid_mask[h0:h1, :].reshape(-1)

            if np.any(block_mask):
                block = np.asarray(block[block_mask], dtype=np.float64)
                gene_sum += block.sum(axis=0)

            del block, block_mask
            gc.collect()

        return (gene_sum + 1e-8).astype(np.float32), None

    return None, None


def normalize_block(block, normalize_mode, stat1, stat2):
    if normalize_mode == "gene-zscore":
        return (block - stat1[None, :]) / stat2[None, :]

    if normalize_mode == "gene-sum":
        return block / stat1[None, :] * 1e4

    return block


def fit_incremental_pca_streaming(
    mode,
    prob_maps,
    ref_matrix_np,
    h,
    w,
    block_h,
    valid_mask,
    pred_ids,
    normalize_mode,
    stat1,
    stat2,
    n_components,
):
    ipca = IncrementalPCA(n_components=n_components)

    first_buffer = []
    first_rows = 0
    fitted = False

    for h0, h1 in tqdm(iter_block_ranges(h, block_h), desc="Pass 2/3: fit IncrementalPCA"):
        block = get_flat_block(
            mode,
            prob_maps,
            ref_matrix_np,
            h0,
            h1,
            pred_ids=pred_ids,
            out_dtype=np.float32,
        )
        block_mask = valid_mask[h0:h1, :].reshape(-1)

        if np.any(block_mask):
            block = np.asarray(block[block_mask], dtype=np.float32)
            block = normalize_block(block, normalize_mode, stat1, stat2)

            if not fitted:
                first_buffer.append(block)
                first_rows += block.shape[0]

                if first_rows >= n_components:
                    init_block = np.concatenate(first_buffer, axis=0)
                    ipca.partial_fit(init_block)
                    fitted = True

                    del init_block
                    first_buffer = []
            else:
                ipca.partial_fit(block)

        del block, block_mask
        gc.collect()

    if not fitted:
        if first_rows == 0:
            raise ValueError("No valid pixels available for PCA fitting.")

        init_block = np.concatenate(first_buffer, axis=0)

        if init_block.shape[0] < n_components:
            raise ValueError(
                f"Not enough valid pixels ({init_block.shape[0]}) for dim={n_components}. "
                "Lower --expr-dim / --max-expr-dim or use a less restrictive mask."
            )

        ipca.partial_fit(init_block)
        del init_block

    return ipca


def build_masked_training_matrix(
    mode,
    prob_maps,
    ref_matrix_np,
    h,
    w,
    block_h,
    valid_mask,
    pred_ids,
    normalize_mode,
    stat1,
    stat2,
    workdir,
    dtype=np.float32,
):
    c_gene = ref_matrix_np.shape[1]
    n_valid = int(valid_mask.sum())

    x_path = os.path.join(workdir, "X_masked_pixel_major.dat")
    x = np.memmap(x_path, mode="w+", dtype=dtype, shape=(n_valid, c_gene))

    row_start = 0

    for h0, h1 in tqdm(iter_block_ranges(h, block_h), desc="Pass 2/3: build masked matrix for TruncatedSVD"):
        block = get_flat_block(
            mode,
            prob_maps,
            ref_matrix_np,
            h0,
            h1,
            pred_ids=pred_ids,
            out_dtype=np.float32,
        )
        block_mask = valid_mask[h0:h1, :].reshape(-1)

        if np.any(block_mask):
            block = np.asarray(block[block_mask], dtype=np.float32)
            block = normalize_block(block, normalize_mode, stat1, stat2)

            n_rows = block.shape[0]
            x[row_start : row_start + n_rows, :] = block.astype(dtype, copy=False)
            row_start += n_rows

        del block, block_mask
        gc.collect()

    x.flush()
    return x


def transform_and_save_streaming(
    model,
    mode,
    prob_maps,
    ref_matrix_np,
    h,
    w,
    block_h,
    valid_mask,
    pred_ids,
    normalize_mode,
    stat1,
    stat2,
    n_components,
    workdir,
    output_pickle,
):
    out_path = os.path.join(workdir, "pixel_gene_array_svd.dat")
    out = np.memmap(out_path, mode="w+", dtype=np.float32, shape=(n_components, h, w))
    out[:] = 0.0

    for h0, h1 in tqdm(iter_block_ranges(h, block_h), desc="Pass 3/3: transform blocks"):
        bh = h1 - h0

        block = get_flat_block(
            mode,
            prob_maps,
            ref_matrix_np,
            h0,
            h1,
            pred_ids=pred_ids,
            out_dtype=np.float32,
        )
        block_mask = valid_mask[h0:h1, :].reshape(-1)

        out_block = np.zeros((bh * w, n_components), dtype=np.float32)

        if np.any(block_mask):
            fit_block = np.asarray(block[block_mask], dtype=np.float32)
            fit_block = normalize_block(fit_block, normalize_mode, stat1, stat2)

            transformed = model.transform(fit_block).astype(np.float32, copy=False)
            out_block[block_mask] = transformed[:, :n_components]

            del transformed
            del fit_block

        out[:, h0:h1, :] = out_block.reshape(bh, w, n_components).transpose(2, 0, 1)

        del block, block_mask, out_block
        gc.collect()

    out.flush()
    save_pickle(np.asarray(out), output_pickle)

    del out
    gc.collect()


def merge_embeddings(prefix):
    embs_feat = load_pickle(prefix + "embeddings-hist-merged.pickle")
    combined = load_pickle(prefix + "embeddings-gene.pickle")

    embs = {
        "vit": embs_feat["vit"],
        "uni": embs_feat["uni"],
        "combined": combined,
    }

    with open(prefix + "embeddings-combined.pickle", "wb") as f:
        pickle.dump(embs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved", prefix + "embeddings-combined.pickle")


def resolve_expression_embedding_selection(args):
    """
    Decide expression embedding dimension selection mode.

    Cases:
    1. --expr-dim K:
       fixed dimension = K
       target_variance = None

    2. --expr-target-variance V:
       fit up to --max-expr-dim components,
       then choose the smallest K reaching V.

    3. Neither is provided:
       fixed dimension = --max-expr-dim
       target_variance = None
    """
    if args.expr_dim is not None:
        if args.expr_dim < 1:
            raise ValueError("--expr-dim must be >= 1.")

        svd_dim = int(args.expr_dim)
        target_variance = None
        selection_mode = "fixed_dim"

    else:
        if args.max_expr_dim < 1:
            raise ValueError("--max-expr-dim must be >= 1.")

        svd_dim = int(args.max_expr_dim)

        if args.expr_target_variance is not None:
            target_variance = float(args.expr_target_variance)

            if not (0.0 < target_variance <= 1.0):
                raise ValueError("--expr-target-variance must be within (0, 1].")

            selection_mode = "target_variance"
        else:
            target_variance = None
            selection_mode = "fixed_dim"

    return svd_dim, target_variance, selection_mode


def main():
    args = get_args()

    prefix = args.prefix
    mode = args.mode
    normalize = None if args.normalize == "none" else args.normalize
    block_h = max(int(args.block_h), 1)
    dtype = np.float32 if args.dtype == "float32" else np.float64

    svd_dim, target_variance, selection_mode = resolve_expression_embedding_selection(args)

    print("Expression embedding selection mode:", selection_mode)
    print("Maximum/fixed expression embedding dim:", svd_dim)
    print("Target variance/reconstruction ratio:", target_variance)

    pickle_dir = prefix + "Cell_proportion/cnts-super"
    proportion_csv = prefix + "reference.csv"
    output_pickle = prefix + "embeddings-gene.pickle"

    created_temp = False

    if args.workdir is None:
        workdir = tempfile.mkdtemp(prefix="memmap_gene_", dir=".")
        created_temp = True
    else:
        os.makedirs(args.workdir, exist_ok=True)
        workdir = args.workdir

    print("Temporary workdir:", workdir)

    files = list_pickle_files(pickle_dir)

    prob_maps, celltypes, h, w = prepare_prob_memmap(
        pickle_dir,
        files,
        workdir,
        dtype=dtype,
    )
    print("Loaded probability maps:", prob_maps.shape)

    valid_mask = resolve_mask(
        prefix,
        args.mask_file,
        h,
        w,
        no_mask=args.no_mask,
    )
    n_valid = int(valid_mask.sum())

    ref_matrix_np, genes = load_reference(
        proportion_csv,
        celltypes,
        dtype=dtype,
    )
    print("Reference matrix:", ref_matrix_np.shape)

    if normalize == "celltype":
        ref_matrix_np = normalize_reference_by_celltype(ref_matrix_np)

    if mode == "uncombined":
        pred_ids = build_pred_ids_streaming(
            prob_maps,
            h,
            w,
            block_h,
        )
    else:
        pred_ids = None

    # ============================================================
    # NMF branch
    # ============================================================
    if args.svd_method == "nmf":
        if normalize in {"gene-zscore", "gene-sum"}:
            raise ValueError(
                "NMF only supports nonnegative normalization. "
                "Use --normalize none or --normalize celltype."
            )

        model, ref_latent, n_components, nmf_records = fit_reference_nmf(
            ref_matrix_np=ref_matrix_np,
            dim=svd_dim,
            target_variance=target_variance,
        )

        print(f"NMF components: {n_components}")

        if nmf_records:
            print(f"NMF reconstruction ratio: {nmf_records[-1]['reconstruction_ratio']:.4f}")

        transform_nmf_streaming(
            mode=mode,
            prob_maps=prob_maps,
            ref_latent=ref_latent,
            h=h,
            w=w,
            block_h=block_h,
            pred_ids=pred_ids,
            output_pickle=output_pickle,
        )

        report = {
            "method": "nmf",
            "selection_mode": selection_mode,
            "requested_dim": int(svd_dim),
            "max_expr_dim": int(svd_dim),
            "chosen_dim": int(n_components),
            "target_variance_or_reconstruction_ratio": None
            if target_variance is None
            else float(target_variance),
            "note": (
                "For NMF, target value refers to reconstruction ratio: "
                "1 - sum((x - recon)^2) / sum(x^2)."
            ),
            "records": nmf_records,
        }

        with open(prefix + "embeddings-gene-meta.json", "w") as f:
            import json

            json.dump(report, f, indent=2)

        print("Saved", prefix + "embeddings-gene-meta.json")

        del prob_maps, ref_matrix_np, pred_ids, valid_mask, genes, model, ref_latent
        gc.collect()

        merge_embeddings(prefix)

        if created_temp and not args.keep_temp:
            shutil.rmtree(workdir, ignore_errors=True)
            print("Removed temporary workdir:", workdir)

        return

    # ============================================================
    # PCA / SVD branch
    # ============================================================
    stat1, stat2 = compute_gene_stats_streaming(
        mode=mode,
        prob_maps=prob_maps,
        ref_matrix_np=ref_matrix_np,
        h=h,
        w=w,
        block_h=block_h,
        valid_mask=valid_mask,
        normalize_mode=normalize,
        pred_ids=pred_ids,
    )

    c_gene = ref_matrix_np.shape[1]

    n_components = min(int(svd_dim), c_gene, n_valid)

    if n_components < 1:
        raise ValueError("No valid PCA/SVD components available.")

    if n_components != svd_dim:
        print(f"Adjusted max/fixed dim from {svd_dim} to {n_components}")

    if args.svd_method == "incremental-pca":
        model = fit_incremental_pca_streaming(
            mode=mode,
            prob_maps=prob_maps,
            ref_matrix_np=ref_matrix_np,
            h=h,
            w=w,
            block_h=block_h,
            valid_mask=valid_mask,
            pred_ids=pred_ids,
            normalize_mode=normalize,
            stat1=stat1,
            stat2=stat2,
            n_components=n_components,
        )

        n_components, pca_records = select_components_by_explained_variance(
            explained_variance_ratio=model.explained_variance_ratio_,
            requested_dim=n_components,
            target_variance=target_variance,
        )

        explained = float(np.sum(model.explained_variance_ratio_[:n_components]))

        print(f"IncrementalPCA fitted components: {model.n_components_}")
        print(f"IncrementalPCA chosen components: {n_components}")
        print(f"Explained variance ratio sum: {explained:.4f}")

        report = {
            "method": "incremental-pca",
            "selection_mode": selection_mode,
            "requested_dim": int(svd_dim),
            "max_expr_dim": int(svd_dim),
            "fitted_dim": int(model.n_components_),
            "chosen_dim": int(n_components),
            "target_variance": None if target_variance is None else float(target_variance),
            "explained_variance_ratio_sum": explained,
            "records": pca_records,
        }

    else:
        x = build_masked_training_matrix(
            mode=mode,
            prob_maps=prob_maps,
            ref_matrix_np=ref_matrix_np,
            h=h,
            w=w,
            block_h=block_h,
            valid_mask=valid_mask,
            pred_ids=pred_ids,
            normalize_mode=normalize,
            stat1=stat1,
            stat2=stat2,
            workdir=workdir,
            dtype=dtype,
        )

        print("Running TruncatedSVD on:", x.shape)

        model = TruncatedSVD(
            n_components=n_components,
            random_state=42,
        )
        model.fit(x)

        n_components, svd_records = select_components_by_explained_variance(
            explained_variance_ratio=model.explained_variance_ratio_,
            requested_dim=n_components,
            target_variance=target_variance,
        )

        explained = float(np.sum(model.explained_variance_ratio_[:n_components]))

        print(f"TruncatedSVD fitted components: {model.n_components}")
        print(f"TruncatedSVD chosen components: {n_components}")
        print(f"Explained variance ratio sum: {explained:.4f}")

        report = {
            "method": "truncated-svd",
            "selection_mode": selection_mode,
            "requested_dim": int(svd_dim),
            "max_expr_dim": int(svd_dim),
            "fitted_dim": int(model.n_components),
            "chosen_dim": int(n_components),
            "target_variance": None if target_variance is None else float(target_variance),
            "explained_variance_ratio_sum": explained,
            "records": svd_records,
        }

        del x
        gc.collect()

    transform_and_save_streaming(
        model=model,
        mode=mode,
        prob_maps=prob_maps,
        ref_matrix_np=ref_matrix_np,
        h=h,
        w=w,
        block_h=block_h,
        valid_mask=valid_mask,
        pred_ids=pred_ids,
        normalize_mode=normalize,
        stat1=stat1,
        stat2=stat2,
        n_components=n_components,
        workdir=workdir,
        output_pickle=output_pickle,
    )

    del prob_maps, ref_matrix_np, pred_ids, stat1, stat2, model, valid_mask, genes
    gc.collect()

    with open(prefix + "embeddings-gene-meta.json", "w") as f:
        import json

        json.dump(report, f, indent=2)

    print("Saved", prefix + "embeddings-gene-meta.json")

    merge_embeddings(prefix)

    if created_temp and not args.keep_temp:
        shutil.rmtree(workdir, ignore_errors=True)
        print("Removed temporary workdir:", workdir)


if __name__ == "__main__":
    main()
