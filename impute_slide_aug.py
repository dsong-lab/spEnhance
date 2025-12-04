import multiprocessing
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import read_lines, read_string, save_pickle, load_pickle, load_tsv, load_image, get_disk_mask, load_csv
from train import get_model as train_load_model
from scipy.spatial.distance import cdist
from model_loss import scstGCN
import os

import argparse

def pad_sliding(arr, kernel_size=7, stride=4):
    """
    Pads the input array (H, W, C) only at the bottom and right,
    so that it can be processed using sliding windows with given kernel and stride sizes.

    Parameters:
        arr (np.ndarray): Input array of shape (H, W, C)
        kernel_size (int): Size of the sliding window
        stride (int): Stride of the sliding window

    Returns:
        padded_arr (np.ndarray): Padded array
        (pad_bottom, pad_right): Tuple of padding applied at bottom and right
    """
    H, W, C = arr.shape

    # Compute required padded size
    def get_pad(size):
        return (np.ceil((size - kernel_size) / stride) + 1) * stride + kernel_size - stride

    new_H = int(get_pad(H))
    new_W = int(get_pad(W))

    pad_bottom = new_H - H
    pad_right = new_W - W

    # Perform padding: only bottom and right
    padded_arr = np.pad(arr,
                        ((0, pad_bottom), (0, pad_right), (0, 0)),
                        mode='constant')

    return padded_arr

def pad(emd, num):
    h, w = emd.shape[0], emd.shape[1]

    pad_h = (num - h % num) % num
    pad_w = (num - w % num) % num

    padded_matrix = np.pad(emd,
                           ((0, pad_h), (0, pad_w), (0, 0)),
                           'constant', constant_values=0)

    new_h, new_w = padded_matrix.shape[:2]
    assert new_h % num == 0 and new_w % num == 0
    return padded_matrix

class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius):
        super().__init__()
        mask = get_disk_mask(radius)
        x = get_patches_flat(x_all, locs, mask)

        self.x = x
        self.y = y
        self.locs = locs
        self.size = x_all.shape[:2]
        self.radius = radius
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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

def get_locs(prefix, target_shape=None):

    locs = load_csv(f'{prefix}locs_aug.csv')

    locs = np.stack([locs['y'], locs['x']], -1)

    if target_shape is not None:
        wsi = load_image(f'{prefix}he.jpg')
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        locs = locs.astype(float)
        locs /= rescale_factor

    locs = locs.round().astype(int)

    return locs

def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')

    cnts = load_csv(f'{prefix}cnts_aug.csv')
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]

    embs = load_pickle(f'{prefix}embeddings-hist-merged.pickle')
    # embs = np.concatenate([embs['pos'], embs['his'], embs['rgb']]).transpose(1, 2, 0)
    embs = np.concatenate([embs['uni'], embs['vit']]).transpose(1, 2, 0)
    # embs = np.concatenate([embs['transformer']]).transpose(1,2,0)
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    return embs, cnts, locs

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
        x, y, locs, radius, ori_radius, prefix, batch_size, epochs, lr,
        load_saved=False, device='cuda'):

    print('x:', x.shape, ', y:', y.shape)

    x = x.copy()

    dataset = SpotDataset(x, y, locs, radius)
    model = train_load_model(
            model_class=scstGCN,
            model_kwargs=dict(
                num_features=x.shape[-1],
                num_genes=y.shape[-1],
                ori_radius=ori_radius,
                lr=lr),
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            load_saved=load_saved, device=device)
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model, dataset

def normalize(embs, cnts):

    embs = embs.copy()
    cnts = cnts.copy()

    # TODO: check if adjsut_weights in extract_features can be skipped
    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)

def predict_single_out(model, z, indices, names, y_range):
    z = torch.tensor(z, device=model.device)
    y = model.get_gene(z, indices=indices)
    y = y.cpu().detach().numpy()

    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y


def predict_single_lat(model, x):
    x = torch.tensor(x, device=model.device)
    z = model.get_hidden(x)

    z = z.cpu().detach().numpy()
    return z
    
def predict_single(model, x, y_range, indices=None):
    x = torch.tensor(x, device=model.device)
    x = model.get_hidden(x)
    y = model.get_gene(x, indices=indices)
    y = y.cpu().detach().numpy()

    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y
    
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
        save_pickle(gene_array, f'{prefix}cnts-super/{gene}.pickle')
        print(f'{gene}.pickle saved to {prefix}cnts-super/{gene}.pickle')


def impute(
        embs, cnts, locs, radius, ori_radius, epochs, batch_size, prefix,
        n_states=1, load_saved=False, device='cuda', n_jobs=1):

    names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)


    kwargs_list = [
            dict(
                x=embs, y=cnts, locs=locs, radius=radius, ori_radius=ori_radius,
                batch_size=batch_size, epochs=epochs, lr=1e-4,
                prefix=f'{prefix}states/{i:02d}/',
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

    cnts_range = np.stack([cnts_min, cnts_max], -1)
    cnts_range /= mask_size

    ####################################--######---------
    gra_size = 7
    h, w = embs.shape[0], embs.shape[1]
    embs_1 = pad_sliding(embs, kernel_size=7, stride=3)

    batch_size_row = gra_size
    n_batches_row = embs_1.shape[0] // batch_size_row

    batch_size_col = gra_size
    n_batches_col = embs_1.shape[1] // batch_size_col

    embs_batches = np.array_split(embs_1, n_batches_row, axis=0)

    embs_batches = [np.array_split(i, n_batches_col, axis=1) for i in embs_batches]
    
    predict(h, w, img_feature = embs_1, model=model_list[0], names_list=names, prefix=prefix, y_range = cnts_range,
            patch_size=7, stride=3,
            device='cuda')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_states', type=int, default=5)
    parser.add_argument('--load_saved', action='store_true')
    args = parser.parse_args()
    return args

def main():
    
    args = get_args()
    
    prefix = args.prefix
    epochs = args.epochs
    device = args.device
    n_states = args.n_states
    load_saved = args.load_saved

    embs, cnts, locs = get_data(prefix)
    
    factor = 16
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    radius = ori_radius / factor

    n_train = cnts.shape[0]
    batch_size = min(2048, n_train//16)

    impute(
            embs=embs, cnts=cnts, locs=locs, radius=radius,
            ori_radius=ori_radius, epochs=epochs,
            batch_size=batch_size, n_states=n_states,
            prefix=prefix, load_saved=load_saved,
            device=device, n_jobs=1)
    
if __name__ == '__main__':
    main()
