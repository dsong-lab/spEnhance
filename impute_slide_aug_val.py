import multiprocessing
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import read_lines, read_string, save_pickle, load_pickle, load_tsv, load_image, get_disk_mask, load_csv
from train_and_val import get_model as train_load_model
from scipy.spatial.distance import cdist
from model_val import scstGCN
from impute_slide_aug import pad_sliding, pad, SpotDataset, predict_single
from impute_slide_val import normalize
import os
import argparse
from tqdm import tqdm

def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    mask = np.ones_like(mask, dtype=bool)
    # 2-D  shape=[201,201]  r=[[-101, 101], [-101, 101]]
    center = shape // 2  #
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]

        x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list

def get_locs(prefix, locs_filename, target_shape=None):

    locs = load_csv(f'{prefix}{locs_filename}') 

    locs = np.stack([locs['y'], locs['x']], -1)

    if target_shape is not None:
        wsi = load_image(f'{prefix}he.jpg')
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        locs = locs.astype(float)
        locs /= rescale_factor

    locs = locs.round().astype(int)

    return locs

def preprocess_cnts(cnts, gene_names):
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]
    return cnts

def get_data(prefix, cnts_train_name, cnts_val_name, locs_train_name):
    gene_names = read_lines(f'{prefix}gene-names.txt')

    cnts_train = preprocess_cnts(load_csv(f'{prefix}{cnts_train_name}'), gene_names)
    cnts_val = preprocess_cnts(load_csv(f'{prefix}{cnts_val_name}'), gene_names)

    embs = load_pickle(f'{prefix}embeddings-combined.pickle')
    embs = np.concatenate([embs['uni'], embs['vit'], embs['combined']]).transpose(1, 2, 0)
    # embs = np.concatenate([embs['pos'], embs['his'], embs['rgb']]).transpose(1, 2, 0)
    # embs = np.concatenate([embs['transformer']]).transpose(1,2,0)
    # embs = load_pickle(f'{prefix}embeddings-hist-uni.pickle')
    # embs = np.concatenate([embs['his'], embs['rgb'], embs['pos']]).transpose(1, 2, 0)

    locs_train = get_locs(prefix, locs_train_name, target_shape=embs.shape[:2])
    locs_val = get_locs(prefix, "locs.csv",target_shape=embs.shape[:2])
    
    return embs, cnts_train, cnts_val, locs_train, locs_val

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
        x_train, y_train, x_val, y_val,
        locs_train, locs_val,
        radius, ori_radius, prefix, batch_size, epochs, lr,
        load_saved=False, device='cuda'):
    x_train = x_train.copy()
    x_val = x_val.copy()

    dataset = SpotDataset(x_train, y_train, locs_train, radius)
    val_dataset = SpotDataset(x_val, y_val, locs_val, radius)
    model = train_load_model(
            model_class=scstGCN,
            model_kwargs=dict(
                num_features=x_train.shape[-1],
                num_genes=y_train.shape[-1],
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

import torch
import torch.nn.functional as F

def predict(h, w, img_feature, model, names_list, prefix, y_range, patch_size=7, stride=1, device='cuda'):
    H, W, C = img_feature.shape
    G = len(names_list) 

    gene_expr_sum = np.zeros((H, W, G))  
    gene_expr_weight = np.zeros((H, W, G))

    model = model.to(device)
    model.eval()
    
    for i in tqdm(range(0, H - patch_size + 1, stride), desc="Rows"):
        for j in range(0, W - patch_size + 1, stride):
            patch_feat = img_feature[i:i+patch_size, j:j+patch_size, :]  
    
            patch_feat_reshaped = patch_feat.reshape(1, -1, patch_feat.shape[-1])  # [1, N, C]
            patch_feat_tensor = torch.tensor(patch_feat_reshaped, dtype=torch.float32).to(device)
    
            patch_pred = predict_single(model=model, x=patch_feat_tensor, y_range=y_range)  # [1, N, G]
            patch_pred = patch_pred.reshape(patch_size, patch_size, G)
    
            gene_expr_sum[i:i+patch_size, j:j+patch_size, :] += patch_pred
            gene_expr_weight[i:i+patch_size, j:j+patch_size, :] += 1
    # 避免除以0
    gene_expr_weight[gene_expr_weight == 0] = 1
    
    # 就地除法，结果直接存回 gene_expr_sum
    gene_expr_sum /= gene_expr_weight

    gene_expr_sum = gene_expr_sum[:h,:w,:]
    
    for i, gene in enumerate(names_list):
        gene_array = gene_expr_sum[:, :, i]
        save_pickle(gene_array, f'{prefix}cnts-super-val/{gene}.pickle')
        print(f'{gene}.pickle saved to {prefix}cnts-super-val/{gene}.pickle')


def impute(
        embs, cnts_train, cnts_val,
        locs_train, locs_val, 
        radius, ori_radius, epochs, batch_size, prefix,
        n_states=1, load_saved=False, device='cuda', n_jobs=1):

    names = cnts_train.columns
    cnts_train = cnts_train.to_numpy().astype(np.float32)
    cnts_val = cnts_val.to_numpy().astype(np.float32)

    (cnts_train, (cnts_train_min, cnts_train_max),
     cnts_val, (cnts_val_min, cnts_val_max)) = normalize(embs, cnts_train, cnts_val)

    dataset = SpotDataset(embs, cnts_train, locs_train, radius)
    val_dataset = SpotDataset(embs, cnts_val, locs_val, radius)

    kwargs_list = [
        dict(
            x_train=embs, y_train=cnts_train, x_val=embs, y_val = cnts_val,
            locs_train=locs_train, locs_val = locs_val,
            radius=radius, ori_radius=ori_radius,
            batch_size=64, epochs=epochs, lr=1e-4,
            prefix=f'{prefix}states/{i:02d}-aug-val/',
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

    ####################################--######---------
    gra_size = 7
    h, w = embs.shape[0], embs.shape[1]
    print("Padding image...")
    embs_1 = pad_sliding(embs, kernel_size=7, stride=2)

    del embs
    
    batch_size_row = gra_size
    n_batches_row = embs_1.shape[0] // batch_size_row

    batch_size_col = gra_size
    n_batches_col = embs_1.shape[1] // batch_size_col

    print("Splitting image into batches")
    embs_batches = np.array_split(embs_1, n_batches_row, axis=0)

    embs_batches = [np.array_split(i, n_batches_col, axis=1) for i in embs_batches]
    print("Prediction started!")
    predict(h, w, img_feature = embs_1, model=model_list[0], names_list=names, prefix=prefix, y_range = cnts_train_range,
            patch_size=7, stride=2,
            device='cuda')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--cnts_train_name', type=str, default='cnts_aug_val.csv')
    parser.add_argument('--cnts_val_name', type=str)
    parser.add_argument('--locs_train_name', type=str, default='locs_aug_val.csv')
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
    locs_train_name = args.locs_train_name
    epochs = args.epochs
    device = args.device
    n_states = args.n_states
    load_saved = args.load_saved

    (embs, cnts_train, cnts_val, 
    locs_train, locs_val) = get_data(prefix, cnts_train_name, cnts_val_name, locs_train_name)
    
    
    factor = 16
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    radius = ori_radius / factor

    n_train = cnts_train.shape[0]
    batch_size = min(1024, n_train//16)

    impute(
            embs=embs, cnts_train=cnts_train, cnts_val=cnts_val, 
            locs_train=locs_train, locs_val = locs_val,
            radius=radius,
            ori_radius=ori_radius, epochs=epochs,
            batch_size=batch_size, n_states=n_states,
            prefix=prefix, load_saved=load_saved,
            device=device, n_jobs=1)
    
if __name__ == '__main__':
    main()
