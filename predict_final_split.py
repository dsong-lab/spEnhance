import argparse
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from impute_final_split import flatten, group_sizes, predict_single, preprocess_cnts
from impute_slide import pad_sliding
from model_val_final import scstGCN
from utils import get_disk_mask, load_csv, load_pickle, read_string, save_pickle


def load_gene_groups(prefix):
    import json

    group_file = prefix / "gene-names-groups.txt"
    if not group_file.exists():
        group_file = prefix / "gene-names-group.txt"

    with open(group_file, "r") as f:
        groups = json.load(f)

    return flatten(groups), group_sizes(groups)


def resolve_checkpoint(state_path):
    state_path = Path(state_path).expanduser()
    if state_path.is_file():
        return state_path

    if not state_path.is_dir():
        raise FileNotFoundError(f"State path does not exist: {state_path}")

    best_ckpts = list((state_path / "checkpoints").glob("best-*.ckpt"))
    if best_ckpts:
        return min(best_ckpts, key=checkpoint_loss)

    model_ckpt = state_path / "model.ckpt"
    if model_ckpt.exists():
        return model_ckpt

    last_ckpts = sorted((state_path / "checkpoints").glob("last*.ckpt"))
    if last_ckpts:
        return last_ckpts[-1]

    raise FileNotFoundError(f"No checkpoint found under {state_path}")


def checkpoint_loss(path):
    match = re.search(r"best-([0-9.]+)-", path.name)
    if match:
        return float(match.group(1))
    return float("inf")


def load_models(state_paths, device):
    models = []
    for state_path in state_paths:
        ckpt = resolve_checkpoint(state_path)
        model = scstGCN.load_from_checkpoint(str(ckpt))
        model = model.to(device).eval()
        models.append(model)
        print(f"Loaded model from {ckpt}")
    return models


def get_prediction_range(prefix, cnts_name, gene_names, mask_size):
    cnts = preprocess_cnts(load_csv(prefix / cnts_name), gene_names)
    cnts = cnts.to_numpy().astype(np.float32)

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts_range = np.stack([cnts_min, cnts_max], -1)
    cnts_range /= mask_size
    return cnts_range


def normalize_feature_block(block):
    block = np.asarray(block, dtype=np.float32)
    mean = np.nanmean(block, axis=(1, 2), keepdims=True)
    std = np.nanstd(block, axis=(1, 2), keepdims=True)
    return np.nan_to_num((block - mean) / (std + 1e-6), nan=0.0, posinf=0.0, neginf=0.0)


def filter_low_variance_channels(block, min_std):
    if min_std <= 0:
        return block
    std = np.nanstd(block, axis=(1, 2))
    keep = std > min_std
    if not np.any(keep):
        raise ValueError(f"No combined embedding channels passed min std threshold {min_std}")
    print(f"Keeping {int(keep.sum())}/{len(keep)} combined channels with std > {min_std}")
    return block[keep]


def load_embedding(prefix, embedding_name, normalize_embeddings=False, combined_weight=1.0,
                   combined_std_threshold=0.0):
    embs = load_pickle(prefix / embedding_name)
    uni = np.asarray(embs["uni"], dtype=np.float32)
    vit = np.asarray(embs["vit"], dtype=np.float32)
    combined = np.asarray(embs["combined"], dtype=np.float32)
    combined = filter_low_variance_channels(combined, combined_std_threshold)

    if normalize_embeddings:
        uni = normalize_feature_block(uni)
        vit = normalize_feature_block(vit)
        combined = normalize_feature_block(combined)

    combined = combined * np.float32(combined_weight)
    return np.concatenate([uni, vit, combined]).transpose(1, 2, 0)


def iter_patch_batches(img_feature, patch_size, stride, batch_size):
    H, W, C = img_feature.shape
    batch_coords = []
    batch_patches = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch_feat = img_feature[i:i + patch_size, j:j + patch_size, :]
            batch_coords.append((i, j))
            batch_patches.append(patch_feat.reshape(-1, C))

            if len(batch_patches) == batch_size:
                yield batch_coords, np.stack(batch_patches, axis=0)
                batch_coords = []
                batch_patches = []

    if batch_patches:
        yield batch_coords, np.stack(batch_patches, axis=0)


def predict_streaming(h, w, img_feature, model_list, names_list, prefix, y_range,
                      patch_size=7, stride=1, batch_size=8, device="cuda",
                      seed_name=None):
    H, W, _ = img_feature.shape
    G = len(names_list)
    n_rows = len(range(0, H - patch_size + 1, stride))
    n_cols = len(range(0, W - patch_size + 1, stride))
    total_batches = int(np.ceil((n_rows * n_cols) / batch_size))

    all_model_preds = []
    for model_idx, model in enumerate(model_list):
        print(f"\nUsing model {model_idx + 1}/{len(model_list)}")

        gene_expr_sum = np.zeros((H, W, G), dtype=np.float32)
        gene_expr_weight = np.zeros((H, W), dtype=np.float32)

        model = model.to(device).eval()
        with torch.no_grad():
            batches = iter_patch_batches(img_feature, patch_size, stride, batch_size)
            pbar = tqdm(batches, total=total_batches, desc=f"Model {model_idx + 1}")
            for coords, patches in pbar:
                batch = torch.tensor(patches, dtype=torch.float32, device=device)
                pred = predict_single(model=model, x=batch, y_range=y_range)
                pred = pred.reshape(pred.shape[0], patch_size, patch_size, G)

                for k, (i, j) in enumerate(coords):
                    gene_expr_sum[i:i + patch_size, j:j + patch_size, :] += pred[k]
                    gene_expr_weight[i:i + patch_size, j:j + patch_size] += 1

        gene_expr_weight[gene_expr_weight == 0] = 1
        gene_expr_sum /= gene_expr_weight[:, :, None]
        gene_expr_sum = gene_expr_sum[:h, :w, :]

        if len(model_list) == 1:
            save_predictions(gene_expr_sum, names_list, prefix, seed_name)
            return

        all_model_preds.append(gene_expr_sum)

    median_pred = np.median(np.stack(all_model_preds, axis=0), axis=0)
    save_predictions(median_pred, names_list, prefix, seed_name)


def save_predictions(prediction, names_list, prefix, seed_name):
    for i, gene in enumerate(names_list):
        out_file = f'{prefix}/Prediction_val/cnts-super{seed_name or ""}/{gene}.pickle'
        save_pickle(prediction[:, :, i], out_file)
        print(f'{gene}.pickle saved to {out_file}')


def get_args():
    parser = argparse.ArgumentParser(
        description="Run only the final-split prediction step from impute_final_split.py."
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path("../benchmark/Xenium/breast-cancer"),
        help="Dataset directory. Defaults to the Xenium breast-cancer example.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        nargs="+",
        default=[
            Path("../benchmark/Xenium/breast-cancer/states/00-val-embs-val-bt64-epoch_60_nstates_5")
        ],
        help="One or more states/00-val-* directories or checkpoint files.",
    )
    parser.add_argument(
        "--cnts-train-name",
        default="cnts_train_seed_1.csv",
        help="Training counts file used to recover prediction gene order.",
    )
    parser.add_argument(
        "--cnts-name",
        default="cnts.csv",
        help="Full counts file used to compute the output count range.",
    )
    parser.add_argument(
        "--embs_all_name",
        "--embs-all-name",
        "--embedding-name",
        dest="embs_all_name",
        default="embeddings-combined-all.pickle",
        help="Embedding pickle to predict over.",
    )
    parser.add_argument(
        "--output-suffix",
        default="-predict-only",
        help="Suffix appended to Prediction_val/cnts-super to avoid overwriting existing outputs.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="Z-score each embedding channel before prediction.",
    )
    parser.add_argument(
        "--combined-weight",
        type=float,
        default=1.0,
        help="Multiplier for the cell-type fused combined block after optional normalization.",
    )
    parser.add_argument(
        "--combined-std-threshold",
        type=float,
        default=0.0,
        help="Drop cell-type fused channels with spatial std at or below this value.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    prefix = args.prefix.expanduser().resolve()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.device == "cpu":
        raise RuntimeError("model_val_final.scstGCN.forward currently hard-codes CUDA tensors.")

    gene_names, _ = load_gene_groups(prefix)
    cnts_train = preprocess_cnts(load_csv(prefix / args.cnts_train_name), gene_names)
    names = cnts_train.columns

    models = load_models(args.state_path, args.device)
    ori_radius = int(getattr(models[0].hparams, "ori_radius", read_string(prefix / "radius.txt")))
    mask_size = int(np.sum(get_disk_mask(ori_radius / 16)))

    cnts_range = get_prediction_range(prefix, args.cnts_name, names, mask_size)
    embs = load_embedding(
        prefix,
        args.embs_all_name,
        normalize_embeddings=args.normalize_embeddings,
        combined_weight=args.combined_weight,
        combined_std_threshold=args.combined_std_threshold,
    )

    h, w = embs.shape[:2]
    tile_size = min(h, w) // 20
    step_size = tile_size // 2
    embs_padded = pad_sliding(embs, kernel_size=tile_size, stride=step_size)

    print(f"Predicting {len(names)} genes on {h}x{w} embeddings")
    print(f"tile_size={tile_size}, step_size={step_size}, models={len(models)}")
    predict_streaming(
        h,
        w,
        img_feature=embs_padded,
        model_list=models,
        names_list=names,
        prefix=str(prefix),
        y_range=cnts_range,
        patch_size=tile_size,
        stride=step_size,
        batch_size=args.batch_size,
        device=args.device,
        seed_name=args.output_suffix,
    )


if __name__ == "__main__":
    main()
