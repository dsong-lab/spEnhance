import argparse
import multiprocessing
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.distance import cdist
from utils import read_lines, read_string, save_pickle, load_pickle, load_image, get_disk_mask, load_csv
from train_and_val import get_model as train_load_model
from model_val import scstGCN
from impute_slide import pad_sliding, pad, SpotDataset, get_patches_flat, get_locs, predict_single

def preprocess_cnts(cnts, gene_names):
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]
    return cnts

def get_data(prefix, cnts_train_name, cnts_val_name):
    gene_names = read_lines(f'{prefix}gene-names.txt')

    cnts_train = preprocess_cnts(load_csv(f'{prefix}{cnts_train_name}'), gene_names)
    cnts_val = preprocess_cnts(load_csv(f'{prefix}{cnts_val_name}'), gene_names)

    embs = load_pickle(f'{prefix}embeddings-combined.pickle')
    embs = np.concatenate([embs['vit'], embs['uni'], embs['combined']]).transpose(1,2,0)
    # embs = load_pickle(f'{prefix}embeddings-hist-uni.pickle')
    # embs = np.concatenate([embs['his'], embs['rgb'], embs['pos']]).transpose(1, 2, 0)
    # # embs = np.concatenate([embs['pos'], embs['his'], embs['rgb']]).transpose(1, 2, 0)
    # embs = np.concatenate([embs['hist'], embs['gene']]).transpose(1, 2, 0)
    
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    return embs, cnts_train, cnts_val, locs

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
    x_train, y_train, x_val, y_val,
    locs, radius, ori_radius, prefix, batch_size, epochs, lr,
    load_saved=False, device='cuda'):

    x_train = x_train.copy()
    x_val = x_val.copy()

    dataset = SpotDataset(x_train, y_train, locs, radius)
    val_dataset = SpotDataset(x_val, y_val, locs, radius)
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

import torch
import torch.nn.functional as F

def predict(h, w, img_feature, model, names_list, prefix, y_range, patch_size=7, stride=1, device='cuda'):

    H, W, C = img_feature.shape
    G = len(names_list)  

    gene_expr_sum = np.zeros((H, W, G))  
    gene_expr_weight = np.zeros((H, W, G))

    model = model.to(device)
    model.eval()
    
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch_feat = img_feature[i:i+patch_size, j:j+patch_size, :]  
    
            patch_feat_reshaped = patch_feat.reshape(1, -1, patch_feat.shape[-1])  # [1, N, C]
            patch_feat_tensor = torch.tensor(patch_feat_reshaped, dtype=torch.float32).to(device)
    
            patch_pred = predict_single(model=model, x=patch_feat_tensor, y_range=y_range)  # [1, N, G]
            
            patch_pred = patch_pred.reshape(patch_size, patch_size, G)
            gene_expr_sum[i:i+patch_size, j:j+patch_size, :] += patch_pred
            gene_expr_weight[i:i+patch_size, j:j+patch_size, :] += 1

    gene_expr_weight[gene_expr_weight == 0] = 1

    gene_expr_map = gene_expr_sum / gene_expr_weight  # [H, W, G]

    gene_expr_map = gene_expr_map[:h,:w,:]
    
    for i, gene in enumerate(names_list):
        gene_array = gene_expr_map[:, :, i]
        save_pickle(gene_array, f'{prefix}/Primary_prediction_val/cnts-super/{gene}.pickle')
        print(f'{gene}.pickle saved to {prefix}/Primary_prediction_val/cnts-super/{gene}.pickle')


def impute(
        embs, cnts_train, cnts_val,
        locs, radius, ori_radius, epochs, batch_size, prefix,
        n_states=1, load_saved=False, device='cuda', n_jobs=1):

    names = cnts_train.columns
    cnts_train = cnts_train.to_numpy().astype(np.float32)
    cnts_val = cnts_val.to_numpy().astype(np.float32)

    (cnts_train, (cnts_train_min, cnts_train_max),
     cnts_val, (_, _)) = normalize(embs, cnts_train, cnts_val)

    # dataset = SpotDataset(embs, cnts_train, locs, radius)
    # val_dataset = SpotDataset(embs, cnts_val, locs, radius)

    kwargs_list = [
        dict(
            x_train=embs, y_train=cnts_train, x_val=embs, y_val = cnts_val,
            locs=locs, radius=radius, ori_radius=ori_radius,
            batch_size=64, epochs=epochs, lr=1e-4,
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

    ####################################--######---------
    gra_size = 7
    h, w = embs.shape[0], embs.shape[1]
    embs_1 = pad_sliding(embs, kernel_size=7, stride=2)

    batch_size_row = gra_size
    n_batches_row = embs_1.shape[0] // batch_size_row

    batch_size_col = gra_size
    n_batches_col = embs_1.shape[1] // batch_size_col

    embs_batches = np.array_split(embs_1, n_batches_row, axis=0)

    embs_batches = [np.array_split(i, n_batches_col, axis=1) for i in embs_batches]
    
    predict(h, w, img_feature = embs_1, model=model_list[0], names_list=names, prefix=prefix, y_range = cnts_train_range,
            patch_size=7, stride=2,
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

    embs, cnts_train, cnts_val, locs = get_data(prefix, cnts_train_name, cnts_val_name)
    
    factor = 16
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    radius = ori_radius / factor

    n_train = cnts_train.shape[0]

    batch_size = min(1024, n_train//16)

    impute(
            embs=embs, cnts_train=cnts_train, cnts_val=cnts_val, 
            locs=locs, radius=radius,
            ori_radius=ori_radius, epochs=epochs,
            batch_size=batch_size, n_states=n_states,
            prefix=prefix, load_saved=load_saved,
            device=device, n_jobs=1)
    
if __name__ == '__main__':
    main()
