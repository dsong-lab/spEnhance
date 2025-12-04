import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD
import pickle

from PIL import Image
from sklearn.decomposition import NMF
from scipy.interpolate import griddata
import math
import matplotlib.image
import cv2 as cv2
from sklearn.metrics import roc_auc_score

import argparse

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--mode', type=str, default='combined') 
    parser.add_argument('--normalize', type=str, default='gene-zscore')
    parser.add_argument('--dim', type=int, default=256)
    args = parser.parse_args()
    return args

def main():
    # ======= Directory =======
    args = get_args()
    
    prefix = args.prefix
    mode = args.mode
    normalize = args.normalize
    svd_dim = args.dim
    
    pickle_dir = prefix + "Cell_proportion/cnts-super"      # Path to all the celltype.pickle files
    proportion_csv = prefix + "reference.csv"  # Path to the single cell reference matrix
    output_pickle = prefix + "embeddings-gene.pickle"  # Path to save the result
    
    # ======= Parameters =======
    # mode: "uncombined" -> assign celltype with biggest probability
    #       "combined" -> linear combination of gene expression features of all cell types according to their probability
    # normalize: None / "celltype" / "gene-zscore" / "gene-sum"
    # ======= Step 1. Load probability matrix of all cell types =======
    celltype_prob_maps = []
    celltypes = []
    
    for fname in os.listdir(pickle_dir):
        if fname.endswith(".pickle"):
            celltype = os.path.splitext(fname)[0]
            with open(os.path.join(pickle_dir, fname), "rb") as f:
                prob_map = pickle.load(f)  # shape [H, W]
            celltype_prob_maps.append(prob_map)
            celltypes.append(celltype)
    
    celltype_prob_maps = np.stack(celltype_prob_maps, axis=0)  # [N_celltype, H, W]
    celltypes = np.array(celltypes)
    
    print("Loaded probability maps:", celltype_prob_maps.shape)
    
    # ======= Step 2. Predict most probable cell type for each pixel=======
    pred_ids = np.argmax(celltype_prob_maps, axis=0)  # [H, W]
    H, W = pred_ids.shape
    # ======= Step 3. Load reference matrix =======
    ref_matrix = pd.read_csv(proportion_csv, index_col=0)  # 行: celltype, 列: gene
    genes = ref_matrix.columns.to_list()
    print("Reference matrix:", ref_matrix.shape)
    
    missing = [ct for ct in celltypes if ct not in ref_matrix.index]
    if missing:
        raise ValueError(f"These celltypes do not exist in reference matrix: {missing}")
    
    ref_matrix_np = ref_matrix.loc[celltypes].to_numpy(dtype=np.float32)  # [N_celltype, C_gene]
    C_gene = ref_matrix_np.shape[1]
    
    # ======= Step 3.5 Normalization by celltype =======
    if normalize == "celltype":
        row_sums = ref_matrix_np.sum(axis=1, keepdims=True) + 1e-8
        ref_matrix_np = ref_matrix_np / row_sums * 1e4
    
    # ======= Step 4. Generate pixel-level gene expression =======
    if mode == "uncombined":
        # Obtain cell type with highest probability for each pixel
        pixel_gene_array = ref_matrix_np[pred_ids]  # [H, W, C_gene]
        pixel_gene_array = pixel_gene_array.transpose(2, 0, 1)  # [C_gene, H, W]
    elif mode == "combined":
        # Initialize output
        pixel_gene_array = np.zeros((C_gene, H, W), dtype=np.float32)
    
        block_size = 100   # Adjust according to memory
        n_blocks = (H + block_size - 1) // block_size
    
        for b in tqdm(range(n_blocks), desc="Combining blocks"):
            h0 = b * block_size
            h1 = min((b + 1) * block_size, H)
    
            # [N_celltype, h1-h0, W]
            prob_block = celltype_prob_maps[:, h0:h1, :]  
    
            # [N_celltype, C_gene]T × [N_celltype, (h1-h0)*W]
            # => [C_gene, (h1-h0)*W]
            block_flat = prob_block.reshape(len(celltypes), -1)  # [N_celltype, n_pix]
            block_exp = ref_matrix_np.T @ block_flat             # [C_gene, n_pix]
    
            # reshape to [C_gene, h1-h0, W]
            block_exp = block_exp.reshape(C_gene, h1-h0, W)
    
            pixel_gene_array[:, h0:h1, :] = block_exp
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # ======= Step 4.5 Normalization by gene =======
    if normalize == "gene-zscore":
        mean = pixel_gene_array.mean(axis=(1, 2), keepdims=True)
        std = pixel_gene_array.std(axis=(1, 2), keepdims=True) + 1e-8
        pixel_gene_array = (pixel_gene_array - mean) / std
    
    elif normalize == "gene-sum":
        sums = pixel_gene_array.sum(axis=(1, 2), keepdims=True) + 1e-8
        pixel_gene_array = pixel_gene_array / sums * 1e4
    
    print("Pixel-level gene expression array:", pixel_gene_array.shape)
    
    # ======= Step 5 SVD ========
    C_gene, H, W = pixel_gene_array.shape
    X = pixel_gene_array.reshape(C_gene, -1).T  # [N_pixel, C_gene]
    
    print("Running TruncatedSVD on:", X.shape)
    
    # TruncatedSVD
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    
    X_svd = svd.fit_transform(X)
    
    print(f"SVD chosen components: {svd.n_components}")
    print(f"Explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.4f}")
    
    # Transform back to [C_svd, H, W]
    pixel_gene_array_svd = X_svd.T.reshape(svd.n_components, H, W)
    
    print("Pixel-level SVD gene expression array:", pixel_gene_array_svd.shape)
    # ======= Step 5. Save the results =======
    with open(output_pickle, "wb") as f:
        pickle.dump(
           pixel_gene_array_svd,
            f
        )
    print("Saved to", output_pickle)
    
    embs_feat = load_pickle(prefix + "embeddings-hist-merged.pickle")
    
    combined = load_pickle(prefix + "embeddings-gene.pickle")
    
    embs = {}
    embs['vit'] = embs_feat['vit']
    embs['uni'] = embs_feat['uni']
    embs['combined'] = combined
    
    with open(prefix + "embeddings-combined.pickle", "wb") as f:
        pickle.dump(
           embs,
            f
        )
    print("Saved")

if __name__ == '__main__':
    main()