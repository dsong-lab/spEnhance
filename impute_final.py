import argparse
import multiprocessing
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.distance import cdist
from utils import read_lines, read_string, save_pickle, load_pickle, load_image, get_disk_mask, load_csv
from train_and_val import get_model as train_load_model
from model_val_final import scstGCN
from impute_slide import pad_sliding, pad, SpotDataset, get_patches_flat, get_locs
from itertools import chain
import json
import ast

def flatten(list_of_lists):
    
    return list(chain.from_iterable(list_of_lists))

def preprocess_cnts(cnts, gene_names):
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]
    return cnts

def group_sizes(list_of_lists):
    
    return [len(group) for group in list_of_lists]

def predict_single(model, x, y_range, indices=None):
    x = torch.tensor(x, device=model.device)
    y = model.forward(x)
    y = y.cpu().detach().numpy()

    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y
    
def get_data(prefix, cnts_train_name, cnts_val_name):

    with open(f'{prefix}gene-names-groups.txt', "r") as f:
        loaded_list = json.load(f)
        
    gene_names = flatten(loaded_list)
    cluster_size = group_sizes(loaded_list)

    cnts_train = preprocess_cnts(load_csv(f'{prefix}{cnts_train_name}'), gene_names)
    cnts_val = preprocess_cnts(load_csv(f'{prefix}{cnts_val_name}'), gene_names)

    embs = load_pickle(f'{prefix}embeddings-combined.pickle')
    # embs = np.concatenate([embs['pos'], embs['his'], embs['rgb']]).transpose(1, 2, 0)
    embs = np.concatenate([embs['uni'], embs['vit'], embs['combined']]).transpose(1,2,0)
    
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    return embs, cnts_train, cnts_val, locs, cluster_size

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
    x_train, y_train, x_val, y_val,
    locs, radius, ori_radius, cluster_size, prefix, batch_size, epochs, lr,
    load_saved=False, device='cuda'):

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
                lr=lr),
            dataset=dataset, val_dataset = val_dataset,
            prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            load_saved=load_saved, device=device)
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model, dataset

def normalize(embs, cnts_train, cnts_val):

    embs = embs.copy()
    cnts_train = cnts_train.copy()
    cnts_val = cnts_val.copy()

    # TODO: check if adjsut_weights in extract_features can be skipped
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

import numpy as np
import torch
from tqdm import tqdm

def predict(h, w, img_feature, model_list, names_list, prefix, y_range,
            patch_size=7, stride=1, batch_size=8, device='cuda'):

    H, W, C = img_feature.shape
    G = len(names_list)

    # Generating tiles
    coords, patches = [], []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch_feat = img_feature[i:i+patch_size, j:j+patch_size, :]
            patch_feat = patch_feat.reshape(-1, C)  # [N, C]
            coords.append((i, j))
            patches.append(patch_feat)
    patches = np.stack(patches, axis=0)  # [num_tiles, N, C]

    all_model_preds = []

    # Prediction with all models in the list
    for model_idx, model in enumerate(model_list):
        print(f"\nUsing model {model_idx+1}/{len(model_list)}")

        gene_expr_sum = np.zeros((H, W, G), dtype=np.float32)
        gene_expr_weight = np.zeros((H, W, G), dtype=np.float32)

        model = model.to(device).eval()
        with torch.no_grad():
            pbar = tqdm(range(0, len(patches), batch_size), desc=f"Model {model_idx+1}")
            for start in pbar:
                end = min(start + batch_size, len(patches))
                batch = torch.tensor(patches[start:end], dtype=torch.float32, device=device)  # [B, N, C]

                pred = predict_single(model=model, x=batch, y_range=y_range)  # [B, N, G]
                B = pred.shape[0]
                pred = pred.reshape(B, patch_size, patch_size, G)

                # Aggregate back to images
                for k, (i, j) in enumerate(coords[start:end]):
                    gene_expr_sum[i:i+patch_size, j:j+patch_size, :] += pred[k]
                    gene_expr_weight[i:i+patch_size, j:j+patch_size, :] += 1

        
        gene_expr_weight[gene_expr_weight == 0] = 1
        gene_expr_sum /= gene_expr_weight
        gene_expr_sum = gene_expr_sum[:h, :w, :]  
        all_model_preds.append(gene_expr_sum)

    # Calculate median values
    all_model_preds = np.stack(all_model_preds, axis=0)  # [num_models, H, W, G]
    median_pred = np.median(all_model_preds, axis=0)     # [H, W, G]

    # Save prediction for each gene
    for i, gene in enumerate(names_list):
        gene_array = median_pred[:, :, i]
        save_pickle(gene_array, f'{prefix}/Prediction_val/cnts-super/{gene}.pickle')
        print(f'{gene}.pickle saved to {prefix}/Prediction_val/cnts-super/{gene}.pickle')

def impute(
        embs, cnts_train, cnts_val,
        locs, radius, cluster_size, ori_radius, epochs, batch_size, prefix,
        n_states=1, load_saved=False, device='cuda', n_jobs=1):

    names = cnts_train.columns
    cnts_train = cnts_train.to_numpy().astype(np.float32)
    cnts_val = cnts_val.to_numpy().astype(np.float32)

    (cnts_train, (cnts_train_min, cnts_train_max),
     cnts_val, (_, _)) = normalize(embs, cnts_train, cnts_val)

    dataset = SpotDataset(embs, cnts_train, locs, radius)
    val_dataset = SpotDataset(embs, cnts_val, locs, radius)

    kwargs_list = [
        dict(
            x_train=embs, y_train=cnts_train, x_val=embs, y_val = cnts_val,
            locs=locs, radius=radius, ori_radius=ori_radius, cluster_size = cluster_size,
            batch_size=batch_size, epochs=epochs, lr=1e-4,
            prefix=f'{prefix}states/{i:02d}-val/',
            load_saved=load_saved, device=device)
        for i in range(n_states)]

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
    tile_size = min(h, w) // 20
    step_size = tile_size // 2
    embs_1 = pad_sliding(embs, kernel_size=tile_size, stride=step_size)

    del embs
    predict(h, w, img_feature = embs_1, model_list=model_list, names_list=names, prefix=prefix, y_range = cnts_train_range,
            patch_size=tile_size, stride=step_size,
            device='cuda')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--cnts_train_name', type=str)
    parser.add_argument('--cnts_val_name', type=str)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_states', type=int, default=5)
    parser.add_argument('--load_saved', action='store_true')
    args = parser.parse_args()
    return args

def main():
    
    args = get_args()
    
    prefix = args.prefix
    cnts_train_name = args.cnts_train_name
    cnts_val_name = args.cnts_val_name
    epochs = args.epochs
    device = args.device
    n_states = args.n_states
    load_saved = args.load_saved

    embs, cnts_train, cnts_val, locs, cluster_size = get_data(prefix, cnts_train_name, cnts_val_name)
    
    factor = 16
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    radius = ori_radius / factor

    n_train = cnts_train.shape[0]

    batch_size = min(256, n_train//16)

    impute(
            embs=embs, cnts_train=cnts_train, cnts_val=cnts_val, 
            locs=locs, radius=radius, cluster_size = cluster_size,
            ori_radius=ori_radius, epochs=epochs,
            batch_size=batch_size, n_states=n_states,
            prefix=prefix, load_saved=load_saved,
            device=device, n_jobs=1)
    
if __name__ == '__main__':
    main()
