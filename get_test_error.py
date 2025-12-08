import os, sys
import numpy as np
from model_val_final import scstGCN
from impute_final_split import get_data_test, normalize
import torch
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from impute_slide import pad_sliding, SpotDataset
from utils import read_string, save_pickle
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import pickle
import argparse

def normalize_cnts(cnts):
        cnts_min = cnts.min(0)
        cnts_max = cnts.max(0)
        cnts -= cnts_min
        cnts /= (cnts_max - cnts_min) + 1e-12
        return cnts, cnts_min, cnts_max
def max_min_norm(arr, percentile=0.9999):
    """Max-min normalization with cutoff at percentile."""
    cutoff = np.percentile(arr, percentile*100)
    arr_clipped = np.clip(arr, 0, cutoff)
    norm_arr = (arr_clipped - arr_clipped.min()) / (arr_clipped.max() - arr_clipped.min() + 1e-8)
    return norm_arr

def compute_metrics(truth, pred):
    truth = truth.astype(np.float32)
    pred = pred.astype(np.float32)
    
    truth_flat = truth.flatten()
    pred_flat = pred.flatten()
    
    ssim_val = ssim(truth, pred, data_range=truth.max() - truth.min())
    rmse_val = np.sqrt(np.mean((truth - pred)**2))
    pcc_val, _ = pearsonr(truth_flat, pred_flat)
    spearman_val, _ = spearmanr(truth_flat, pred_flat)
    # auc_val = average_precision_score((truth > 0).astype(int), pred)
    
    return ssim_val, rmse_val, pcc_val, spearman_val

def apply_mask(arr, mask):
    return arr * mask

def read_pickle_array(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
        
def process_gene(index, gene, truth_path, pred_arr_all, mask):
    truth_file = os.path.join(truth_path, f"{gene}.pickle")    
    if not os.path.exists(truth_file):
        print(f"Skipping {gene}: file missing.")
        return None
    truth_arr = read_pickle_array(truth_file)
    pred_arr = pred_arr_all[:,:,index]
    h= min(truth_arr.shape[0], pred_arr.shape[0], mask.shape[0])
    w= min(truth_arr.shape[1], pred_arr.shape[1], mask.shape[1])

    truth_arr = truth_arr[:h,:w]
    pred_arr = pred_arr[:h,:w]
    mask = mask[:h,:w]
    # mask
    truth_arr = apply_mask(truth_arr, mask)
    pred_arr = apply_mask(pred_arr, mask)
    
    # normalization
    truth_norm = max_min_norm(truth_arr)
    pred_norm = max_min_norm(pred_arr)
    
    # compute metrics
    ssim_val, rmse_val, pcc_val, spearman_val = compute_metrics(truth_norm, pred_norm)
    
    return {
        'gene': gene,
        'SSIM': ssim_val,
        'RMSE': rmse_val,
        'PCC': pcc_val,
        'Spearman': spearman_val
        # 'AUC': np.round(auc_val, 4)
    }

def evaluate_gene_parallel(genes, truth_path, pred_arr_all, mask, max_workers=32):
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_gene, index, gene, truth_path, pred_arr_all, mask): (index, gene) for index, gene in enumerate(genes)}
        
        for future in tqdm(futures, total=len(futures), desc="Processing genes"):
            res = future.result()
            if res is not None:
                results.append(res)
    
    df = pd.DataFrame(results)
    df.set_index('gene', inplace=True)
    return df

def plot_pixel_diff(prefix, genes, truth_path, impute_path):
    mask_img = Image.open(prefix + "mask-small.png").convert('L')
    mask = np.array(mask_img)
    mask = (mask > 128).astype(np.float32) 
    for gene in genes:
        truth_file = os.path.join(truth_path, f"{gene}.pickle")    
        if not os.path.exists(truth_file):
            print(f"Skipping {gene}: file missing.")
            return None
        truth_arr = read_pickle_array(truth_file)
        pred_arr = read_pickle_array(os.path.join(impute_path, f"{gene}.pickle") )
        h= min(truth_arr.shape[0], pred_arr.shape[0], mask.shape[0])
        w= min(truth_arr.shape[1], pred_arr.shape[1], mask.shape[1])

        truth_arr = truth_arr[:h,:w]
        pred_arr = pred_arr[:h,:w]
        mask = mask[:h,:w]
        # mask
        truth_arr = apply_mask(truth_arr, mask)
        pred_arr = apply_mask(pred_arr, mask)

def predict_single(model, x, y_range, indices=None):
    x = torch.tensor(x, device=model.device)
    y = model.forward(x)
    y = y.cpu().detach().numpy()

    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y

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
    return median_pred

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--index', type=int)
    parser.add_argument('--save_cnts_super', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dataset_name = args.dataset_name
    index = int(args.index)
    save_cnts_super = bool(args.save_cnts_super)
    prefix=f'/home/FCAM/dosong/xutao/stEnhance_work/benchmark/{dataset_name}'
    device='cuda'
    factor = 16
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    radius = ori_radius / factor
    embs, cnts, locs, cluster_size = get_data_test(prefix)

    names = cnts.columns
    cnts = cnts.to_numpy().astype(np.float32)

    cnts, cnts_min, cnts_max = normalize_cnts(cnts)

    dataset = SpotDataset(embs, cnts, locs, radius)

    mask_size = dataset.mask.sum()

    cnts_range = np.stack([cnts_min, cnts_max], -1)
    cnts_range /= mask_size

    h, w = embs.shape[0], embs.shape[1]
    tile_size = min(h, w) // 20
    step_size = tile_size // 2
    embs_1 = pad_sliding(embs, kernel_size=tile_size, stride=step_size)
    del embs

    path_to_model = f'{prefix}/states/00-val/'
    path_to_load_model = f'{path_to_model}/checkpoints'
    ckpt_files = [f for f in os.listdir(path_to_load_model) if f.startswith('epoch-')]
    ckpt_files.sort()
    n = len(ckpt_files)
    print(n)

    # model = scstGCN.load_from_checkpoint(f'{path_to_load_model}/{ckpt_files[0]}')

    checkpoint_file = ckpt_files[index]
    print(checkpoint_file)
    # # Load checkpoint weights
    # state_dict = torch.load(f'{path_to_load_model}/{checkpoint_file}', map_location=device)['state_dict']
    # model.load_state_dict(state_dict)
    model = scstGCN.load_from_checkpoint(f'{path_to_load_model}/{checkpoint_file}')
    model.eval() # Disable dropout
    model_list = [model]
    print("Starting prediction")

    pred_arr_all = predict(h, w, img_feature = embs_1, model_list=model_list, 
                        names_list=names, prefix=prefix, y_range = cnts_range,
                        patch_size=tile_size, stride=step_size,
                        device='cuda')
    if save_cnts_super:
        path_to_lab = f'/labs/Song/stEnhance/benchmark_results/{dataset_name}'
        for i, gene in enumerate(names):
            gene_array = pred_arr_all[:, :, i]
            save_pickle(gene_array, f'{path_to_lab}/checkpoints-cnts-super/{checkpoint_file}/{gene}.pickle')
            print(f'{gene}.pickle saved to {path_to_lab}/checkpoints-cnts-super/{checkpoint_file}/{gene}.pickle')

        sys.exit()

    path_to_truth = f'/labs/Song/stEnhance/data/benchmark/Truth/{dataset_name}/cnts-super/'
    mask_img = Image.open(prefix + "mask-small.png").convert('L')
    mask = np.array(mask_img)
    mask = (mask > 128).astype(np.float32) 

    df = evaluate_gene_parallel(names, path_to_truth, pred_arr_all, mask)
    if not os.path.exists(f'{prefix}/checkpoints/'):
        os.makedirs(f'{prefix}/checkpoints/')
    df.to_csv(f'{prefix}/checkpoints/df_{checkpoint_file}.csv')
    print(f"file saved to {prefix}/checkpoints/df_{checkpoint_file}.csv")

if __name__ == '__main__':
    main()
        