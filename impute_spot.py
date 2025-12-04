import numpy as np
import pandas as pd
import pickle
import os
from scipy.spatial import Delaunay
from tqdm import tqdm

import multiprocessing
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import read_lines, read_string, save_pickle, load_pickle, load_tsv, load_image, get_disk_mask
from train import get_model as train_load_model
from scipy.spatial.distance import cdist

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--folder_val_name', type=str, default=None)
    args = parser.parse_args()
    return args
    
def main():

    args = get_args()
    prefix = args.prefix
    folder_val_name = args.folder_val_name
    
    mask = get_disk_mask(3.434) # Normally, you don't need to change this radius
    
    def load_pickle(filename, verbose=True):
        with open(filename, 'rb') as file:
            x = pickle.load(file)
        return x
    
    def normalization(data):
        _range = (np.max(data) - np.min(data) ) + 1e-6
        return (data - np.min(data)) / _range
    
    def read_lines(filename):
        with open(filename, 'r') as file:
            lines = [line.rstrip() for line in file]
        return lines
    
    locs = pd.read_csv(prefix + "locs.csv", index_col=0)  # columns: x, y
    locs = np.ceil(locs/16) - 1
    with open(prefix+"gene-names.txt") as f:
        gene_names = [line.strip() for line in f]
    
    cnts_stack = []
    if folder_val_name:
        print("Results based on validation.")

    for gene in tqdm(gene_names, desc="Loading expression matrices"):
        if not folder_val_name:
            with open(f"{prefix}Primary_prediction_val/cnts-super/{gene}.pickle", "rb") as f:
                cnts_stack.append(pickle.load(f))  # assumed to be 2D np.array
            continue
        with open(f"{prefix}{folder_val_name}/cnts-super/{gene}.pickle", "rb") as f:
            cnts_stack.append(pickle.load(f))  # assumed to be 2D np.array
        
    cnts_stack = np.stack(cnts_stack, axis=-1)  # shape: (H, W, G)
    
    mask = get_disk_mask(3.434)
    
    coords = locs[['x', 'y']].values
    tri = Delaunay(coords)
    barycenters = coords[tri.simplices].mean(axis=1)  # shape: (T, 2)
    
    # Convert barycenter positions to integer pixel coordinates (for mask indexing)
    barycenters_int = np.round(barycenters).astype(int)
    
    new_spot_expr = []
    new_spot_pos = []
    
    h, w, g = cnts_stack.shape
    mh, mw = mask.shape
    
    # Spot mask assumed centered — we get offsets
    spot_pixels = np.argwhere(mask)  # e.g. [[-2, 0], [-1, -1], ...]
    
    offsets = spot_pixels - np.array(mask.shape) // 2  # offset relative to center
    
    for center in tqdm(barycenters_int, desc="Computing new spot expressions"):
        cx, cy = center
        expr = np.zeros(g)
        valid = True
        for dy, dx in offsets:
            y, x = cy + dy, cx + dx
            if 0 <= y < h and 0 <= x < w:
                expr += cnts_stack[y, x]
            else:
                valid = False
                break
        if valid:
            new_spot_expr.append(expr)
            new_spot_pos.append([cx, cy])
    
    new_spot_expr = np.array(new_spot_expr)  # shape: (N_new, G)
    new_spot_pos = np.array(new_spot_pos)    # shape: (N_new, 2)
    
    locs_old = locs[['x', 'y']].values
    cnts_old = pd.read_csv(prefix+"cnts_train_seed_1.csv", index_col=0)[gene_names]
 
    cnts_all = np.vstack([cnts_old, new_spot_expr])  # shape: (N_total, G)
    locs_all = np.vstack([locs_old, new_spot_pos])   # shape: (N_total, 2)
    locs_all *= 16
    new_ids = [f"new_spot_{i}" for i in range(len(new_spot_expr))]
    
    spot_ids = list(locs.index) + new_ids
    
    combined_mask = cnts_all.sum(axis=1) > 50
    
    cnts_df_filtered = cnts_all[np.array(combined_mask)]
    locs_df_filtered = locs_all[np.array(combined_mask)]
    
    cnts_df = pd.DataFrame(cnts_df_filtered, columns=gene_names)
    locs_df = pd.DataFrame(locs_df_filtered, columns=["x", "y"])
    if not folder_val_name:
        cnts_df.to_csv(prefix+"cnts_aug.csv")
        locs_df.to_csv(prefix+"locs_aug.csv")
    else:
        cnts_df.to_csv(prefix+"cnts_aug_val.csv")
        locs_df.to_csv(prefix+"locs_aug_val.csv")
    
    print("Done! Augmented matrices saved")

if __name__ == '__main__':
    main()