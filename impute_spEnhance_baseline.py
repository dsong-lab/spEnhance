import argparse
import json
import multiprocessing
import os
from itertools import chain

import numpy as np
import torch
from tqdm import tqdm

from impute_slide import SpotDataset, get_locs, pad_sliding
from model_spEnhance_baseline import scstGCN
from train_and_val_spEnhance_baseline import get_model as train_load_model
from utils import load_csv, load_pickle, read_string, save_pickle


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


def group_sizes(list_of_lists):
    return [len(group) for group in list_of_lists]


def preprocess_cnts(cnts, gene_names):
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]
    return cnts


def parse_feature_names(feature_names_raw):
    feature_names = [name.strip() for name in feature_names_raw.split(",") if name.strip()]
    valid = {"vit", "uni", "combined"}
    invalid = [name for name in feature_names if name not in valid]
    if invalid:
        raise ValueError(f"Unsupported features: {invalid}. Valid options are: {sorted(valid)}")
    if not feature_names:
        raise ValueError("At least one feature must be selected.")
    return feature_names


def load_group_info(prefix):
    group_file = f"{prefix}gene-names-group.txt"
    with open(group_file, "r") as f:
        loaded_list = json.load(f)
    gene_names = flatten(loaded_list)
    cluster_size = group_sizes(loaded_list)
    return gene_names, cluster_size


def get_data(prefix, cnts_train_name, cnts_val_name, feature_names):
    gene_names, cluster_size = load_group_info(prefix)

    cnts_train = preprocess_cnts(load_csv(f"{prefix}{cnts_train_name}"), gene_names)
    cnts_val = preprocess_cnts(load_csv(f"{prefix}{cnts_val_name}"), gene_names)

    embs = load_pickle(f"{prefix}embeddings-combined.pickle")
    embs = np.concatenate([embs[name] for name in feature_names]).transpose(1, 2, 0)
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    return embs, cnts_train, cnts_val, locs, cluster_size


def normalize(embs, cnts_train, cnts_val):
    embs = embs.copy()
    cnts_train = cnts_train.copy()
    cnts_val = cnts_val.copy()

    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    def normalize_cnts(cnts):
        cnts_min = cnts.min(0)
        cnts_max = cnts.max(0)
        cnts -= cnts_min
        cnts /= (cnts_max - cnts_min) + 1e-12
        return cnts, cnts_min, cnts_max

    cnts_train, cnts_train_min, cnts_train_max = normalize_cnts(cnts_train)
    cnts_val, cnts_val_min, cnts_val_max = normalize_cnts(cnts_val)
    return cnts_train, (cnts_train_min, cnts_train_max), cnts_val, (cnts_val_min, cnts_val_max)


def predict_single(model, x, y_range):
    x = torch.tensor(x, device=model.device)
    y = model.forward(x)
    y = y.cpu().detach().numpy()
    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y


def predict(h, w, img_feature, model_list, names_list, prefix, y_range, patch_size=7, stride=1, batch_size=8, device="cuda"):
    height, width, channels = img_feature.shape
    num_genes = len(names_list)

    coords, patches = [], []
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patch_feat = img_feature[i : i + patch_size, j : j + patch_size, :].reshape(-1, channels)
            coords.append((i, j))
            patches.append(patch_feat)
    patches = np.stack(patches, axis=0)

    model_maps = []
    for model_idx, model in enumerate(model_list):
        print(f"\nUsing model {model_idx + 1}/{len(model_list)}")
        gene_expr_sum = np.zeros((height, width, num_genes), dtype=np.float32)
        gene_expr_weight = np.zeros((height, width, num_genes), dtype=np.float32)

        model = model.to(device).eval()
        with torch.no_grad():
            pbar = tqdm(range(0, len(patches), batch_size), desc=f"Model {model_idx + 1}")
            for start in pbar:
                end = min(start + batch_size, len(patches))
                batch = torch.tensor(patches[start:end], dtype=torch.float32, device=device)
                pred = predict_single(model=model, x=batch, y_range=y_range)
                pred = pred.reshape(pred.shape[0], patch_size, patch_size, num_genes)

                for local_idx, (i, j) in enumerate(coords[start:end]):
                    gene_expr_sum[i : i + patch_size, j : j + patch_size, :] += pred[local_idx]
                    gene_expr_weight[i : i + patch_size, j : j + patch_size, :] += 1

        gene_expr_weight[gene_expr_weight == 0] = 1
        gene_expr_sum /= gene_expr_weight
        gene_expr_sum = gene_expr_sum[:h, :w, :]
        model_maps.append(gene_expr_sum)
        print("Finished")

    out_dir = f"{prefix}/Prediction_val_baseline/cnts-super"
    os.makedirs(out_dir, exist_ok=True)
    # Keep the original (2) behavior: save the first state's output.
    first_map = model_maps[0]
    for gene_idx, gene in enumerate(names_list):
        gene_array = first_map[:, :, gene_idx]
        save_pickle(gene_array, f"{out_dir}/{gene}.pickle")
        print(f"{gene}.pickle saved to {out_dir}/{gene}.pickle")


def get_model_kwargs(kwargs):
    return get_model(**kwargs)


def get_model(
    x_train,
    y_train,
    x_val,
    y_val,
    locs,
    radius,
    ori_radius,
    cluster_size,
    prefix,
    batch_size,
    epochs,
    lr,
    graph_model,
    gat_heads,
    load_saved=False,
    device="cuda",
):
    x_train = x_train.copy()
    x_val = x_val.copy()

    dataset = SpotDataset(x_train, y_train, locs, radius)
    val_dataset = SpotDataset(x_val, y_val, locs, radius)

    model = train_load_model(
        model_class=scstGCN,
        model_kwargs=dict(
            num_features=x_train.shape[-1],
            num_genes=cluster_size,
            ori_radius=ori_radius,
            lr=lr,
            graph_model=graph_model,
            gat_heads=gat_heads,
        ),
        dataset=dataset,
        val_dataset=val_dataset,
        prefix=prefix,
        batch_size=batch_size,
        epochs=epochs,
        load_saved=load_saved,
        device=device,
    )
    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
    return model, dataset


def impute(
    embs,
    cnts_train,
    cnts_val,
    locs,
    radius,
    cluster_size,
    ori_radius,
    epochs,
    batch_size,
    prefix,
    graph_model="gcn",
    gat_heads=1,
    n_states=1,
    load_saved=False,
    device="cuda",
    n_jobs=1,
):
    names = cnts_train.columns
    cnts_train = cnts_train.to_numpy().astype(np.float32)
    cnts_val = cnts_val.to_numpy().astype(np.float32)

    cnts_train, (cnts_train_min, cnts_train_max), cnts_val, (_, _) = normalize(embs, cnts_train, cnts_val)

    kwargs_list = [
        dict(
            x_train=embs,
            y_train=cnts_train,
            x_val=embs,
            y_val=cnts_val,
            locs=locs,
            radius=radius,
            ori_radius=ori_radius,
            cluster_size=cluster_size,
            batch_size=batch_size,
            epochs=epochs,
            lr=1e-4,
            graph_model=graph_model,
            gat_heads=gat_heads,
            prefix=f"{prefix}states_baseline/{idx:02d}-val/",
            load_saved=load_saved,
            device=device,
        )
        for idx in range(n_states)
    ]

    if n_jobs is None or n_jobs < 1:
        n_jobs = n_states
    if n_jobs == 1:
        out_list = [get_model_kwargs(kwargs) for kwargs in kwargs_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            out_list = pool.map(get_model_kwargs, kwargs_list)

    model_list = [out[0] for out in out_list]
    dataset_list = [out[1] for out in out_list]
    mask_size = dataset_list[0].mask.sum()

    cnts_train_range = np.stack([cnts_train_min, cnts_train_max], -1)
    cnts_train_range /= mask_size

    h, w = embs.shape[0], embs.shape[1]
    tile_size = min(h, w) // 5
    step_size = tile_size // 2
    embs_padded = pad_sliding(embs, kernel_size=tile_size, stride=step_size)

    del embs
    predict(
        h=h,
        w=w,
        img_feature=embs_padded,
        model_list=model_list,
        names_list=names,
        prefix=prefix,
        y_range=cnts_train_range,
        patch_size=tile_size,
        stride=step_size,
        batch_size=batch_size,
        device=device,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str)
    parser.add_argument("--cnts_train_name", type=str, required=True)
    parser.add_argument("--cnts_val_name", type=str, required=True)
    parser.add_argument("--features", type=str, default="combined")
    parser.add_argument("--graph-model", type=str, choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--gat-heads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_states", type=int, default=5)
    parser.add_argument("--load_saved", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    feature_names = parse_feature_names(args.features)

    embs, cnts_train, cnts_val, locs, cluster_size = get_data(
        args.prefix,
        args.cnts_train_name,
        args.cnts_val_name,
        feature_names,
    )

    ori_radius = int(read_string(f"{args.prefix}radius.txt"))
    radius = ori_radius / 16
    n_train = cnts_train.shape[0]
    batch_size = min(16, n_train // 8)

    impute(
        embs=embs,
        cnts_train=cnts_train,
        cnts_val=cnts_val,
        locs=locs,
        radius=radius,
        cluster_size=cluster_size,
        ori_radius=ori_radius,
        epochs=args.epochs,
        batch_size=batch_size,
        prefix=args.prefix,
        graph_model=args.graph_model,
        gat_heads=args.gat_heads,
        n_states=args.n_states,
        load_saved=args.load_saved,
        device=args.device,
        n_jobs=1,
    )


if __name__ == "__main__":
    main()
