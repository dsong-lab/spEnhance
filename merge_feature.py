import pickle
import numpy as np
from PIL import Image
import pandas as pd
from scipy.interpolate import griddata
import math
import matplotlib.image
import cv2 as cv2
from PIL import Image
from sklearn.metrics import roc_auc_score
import os
import argparse

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, NMF, FastICA
import numpy as np
from scipy.stats import kurtosis

def pca_select_n_components(feature_C_H_W, threshold=0.95, name ='pca'):
    C, H, W = feature_C_H_W.shape
    X = feature_C_H_W.reshape(C, -1).T  # [H*W, C]
    
    pca = PCA().fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    n_components = np.searchsorted(cum_var, threshold) + 1
    
    plt.plot(cum_var, label="Cumulative Variance")
    plt.axhline(threshold, color='r', linestyle='--', label=f"{int(threshold*100)}% threshold")
    plt.axvline(n_components, color='g', linestyle='--', label=f"{n_components} components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Variance Analysis"+ '-' + name + '[Lung-1]')
    plt.legend()
    plt.grid()
    plt.savefig('Cumulative Variance' +"-PCA-" + name +'.png')

    return n_components
    
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

def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)

def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--method', type=str)
    args = parser.parse_args()
    return args

def ica_select_n_components(data, max_k=30):
    scores = []
    for k in range(2, max_k+1):
        ica = FastICA(n_components=k, random_state=0)
        X_trans = ica.fit_transform(data)
        kurt = np.mean(np.abs(kurtosis(X_trans)))  
        scores.append(kurt)

    best_k = np.argmax(scores) + 2
    return best_k

def nmf_select_n_components(data, max_k=30):
    errors = []
    for k in range(2, max_k+1):
        model = NMF(n_components=k, init='nndsvda', random_state=0)
        W = model.fit_transform(data)
        H = model.components_
        recon = np.dot(W, H)
        error = np.linalg.norm(data - recon)
        errors.append(error)

    optimal_k = np.argmin(np.gradient(np.gradient(errors))) + 2

    return optimal_k

def auto_select_n_components(data, method='pca', threshold=0.95, max_k=30, name=None):
    if method == 'pca':
        pca = PCA()
        pca.fit(data)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(cumulative, threshold) + 1
    elif method == 'nmf':
        n_components = nmf_select_n_components(data, max_k=max_k)
    elif method == 'ica':
        n_components = ica_select_n_components(data, max_k=max_k)
    else:
        raise ValueError(f"No auto selection available for method: {method}")

    print(f"{name}: Selected {n_components} components using {method.upper()}")
    return n_components

from sklearn.decomposition import IncrementalPCA

def pca_with_incremental(X, threshold=0.95, batch_size=1000):
    ipca = IncrementalPCA(batch_size=batch_size)
    ipca.fit(X)

    cum_var = np.cumsum(ipca.explained_variance_ratio_)
    k = np.searchsorted(cum_var, threshold) + 1
    print(f"PCA with Incremental: selected {k} components to reach {threshold:.2f} cumulative variance.")

    # transform only the first k components
    X_reduced = ipca.transform(X)[:, :k]
    return X_reduced, k


def dim_reduction(embs, name, threshold=0.95, max_k=30, method='pca', n_components=None, random_state=42, batch_size=1000):
    """
    Performs dimensionality reduction on (C, H, W) tensor using specified method.
    Uses IncrementalPCA for large matrices to reduce memory usage.
    """
    C, H, W = embs.shape
    data_reshaped = embs.reshape(C, -1)  # (C, H*W)
    data = data_reshaped.T  # shape: (H*W, C)

    print(f"{name}: Running {method.upper()} on data with shape {data.shape}")

    if method.lower() == 'pca':
        try:
            data_reduced, n_components = pca_with_incremental(data, threshold=threshold, batch_size=batch_size)
        except Exception as e:
            raise RuntimeError(f"Incremental PCA failed: {e}")

    elif method.lower() == 'nmf':
        if np.any(data < 0):
            raise ValueError("NMF requires non-negative data.")
        if n_components is None:
            n_components = nmf_select_n_components(data, max_k=max_k)
        model = NMF(n_components=n_components, init='nndsvda', random_state=random_state)
        data_reduced = model.fit_transform(data)

    elif method.lower() == 'ica':
        if n_components is None:
            n_components = ica_select_n_components(data, max_k=max_k)
        model = FastICA(n_components=n_components, random_state=random_state)
        data_reduced = model.fit_transform(data)

    else:
        raise ValueError(f"Unsupported method: {method}")

    try:
        embs_out = data_reduced.T.reshape(n_components, H, W)
    except Exception as e:
        raise RuntimeError(f"Reshape failed: {e}")

    print(f"{name}: Reduced using {method.upper()} to shape {embs_out.shape}")
    return embs_out



def main():
    args = get_args()

    embs_uni = load_pickle(args.prefix + "embeddings-hist-uni.pickle")
    embs_vit = load_pickle(args.prefix + "embeddings-hist-vit.pickle")

    uni = np.concatenate([embs_uni['his'], embs_uni['rgb'], embs_uni['pos']])
    vit = np.concatenate([embs_vit['sub'], embs_vit['rgb'], embs_vit['cls']])

    uni_reduced = dim_reduction(embs = uni, name = 'uni', threshold = 0.99, max_k=30, method=args.method)
    vit_reduced = dim_reduction(embs = vit, name = 'vit', threshold = 0.99, max_k=30, method=args.method)

    embs_merged = {}
    embs_merged['vit'] = vit_reduced
    embs_merged['uni'] = uni_reduced

    save_pickle(embs_merged, args.prefix + 'embeddings-hist-merged.pickle')

if __name__ == '__main__':
    main()