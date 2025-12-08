import os
import argparse
import pandas as pd
import json
import numpy as np
from impute_final_split import preprocess_cnts, flatten
from scipy.ndimage import gaussian_filter
from utils import load_image, load_csv, read_string, get_disk_mask, load_pickle, plot_with_colorbar, plot_without_colorbar
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from get_test_error import predict, normalize_cnts, max_min_norm, compute_metrics
from skimage.metrics import structural_similarity as ssim
from model_val_final import scstGCN
from impute_slide import pad_sliding, SpotDataset

def get_locs(prefix, locs, target_shape=None):

    locs = np.stack([locs['y'], locs['x']], -1)

    if target_shape is not None:
        wsi = load_image(f'{prefix}he.jpg')
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        locs = locs.astype(float)
        locs /= rescale_factor

    locs = locs.round().astype(int)

    return locs

def scale_locs(locs):
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs /= 16
    locs = locs.round().astype(int)
    return locs

def predict_custom(prefix, cnts, locs, embs, gene_names, radius):
    embs = np.concatenate([embs['uni'], embs['vit'], embs['combined']]).transpose(1,2,0)
    cnts = preprocess_cnts(cnts, gene_names)
    locs = get_locs(prefix, locs, target_shape=embs.shape[:2])

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
    
    model_list = []
    path_to_states = f'{prefix}/states'
    path_to_vals = [os.path.join(path_to_states, f, 'checkpoints') 
                   for f in os.listdir(path_to_states) if f.endswith("-val")]
    path_to_models = []
    for path_to_val in path_to_vals:
        tmp = None
        for mod in os.listdir(path_to_val):
            if mod.startswith('best-'):
                if not tmp:
                    tmp = mod
                    error = float(tmp.split('-')[1])
                else:
                    new_error = float(mod.split('-')[1])
                    if new_error < error:
                        error = new_error
                        tmp = mod
        path_to_models.append(f'{path_to_val}/{tmp}')
    print(f'path_to_models is {path_to_models}')
    for path_to_model in path_to_models:
        model = scstGCN.load_from_checkpoint(f'{path_to_model}')
        model.eval()
        model_list.append(model)
    
    pred_arr_all = predict(h, w, img_feature = embs_1, model_list=model_list, 
                        names_list=gene_names, prefix=prefix, y_range = cnts_range,
                        patch_size=tile_size, stride=step_size,
                        device='cuda')
    return pred_arr_all

def pixel_to_spot_parallel(prefix, pred_arr_all, locs, mask, gene_names, max_workers=32):
    results = []
    locs_edit = get_locs(prefix, locs, target_shape=pred_arr_all.shape[:2])
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(pixel_to_spot, index, gene, pred_arr_all, locs_edit, mask, locs.index): 
                        (index, gene) for index, gene in enumerate(gene_names)}
        
        for future in tqdm(futures, total=len(futures), desc="Transform value from pixel to spot"):
            res = future.result()
            if res is not None:
                results.append(res)
    
    return pd.concat(results, axis = 1)


def pixel_to_spot(index, gene, pred_arr_all, locs, mask, locs_index):
    pred_arr = pred_arr_all[:,:,index]
    
    shape = np.array(mask.shape) # mask shape is 7 X 7
    mask = np.ones_like(mask, dtype=bool)
    center = shape // 2 
    r = np.stack([-center, shape-center], -1)  # offset
    impute_spot_list = []
    for s in locs:
        patch = pred_arr[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        x = np.nansum(patch[mask])
        impute_spot_list.append(x)
    df_impute_spot = pd.DataFrame({gene: impute_spot_list})
    df_impute_spot.index = locs_index
    
    return df_impute_spot

def get_spot_diff(df_impute_spot, cnts_val, gene_name):
    diff = np.sqrt(cnts_val[gene_name]) - np.sqrt(df_impute_spot[gene_name])
    return {gene_name: diff}

def get_spot_diff_parallel(df_impute_spot, cnts_val, gene_names, max_workers = 32):
    spot_diff_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_spot_diff, df_impute_spot, cnts_val, gene): gene for gene in gene_names}
        
        for future in tqdm(futures, total=len(futures), desc="Compute spot diffeence for each gene"):
            res = future.result()
            if res is not None:
                spot_diff_dict.update(res)
    
    return spot_diff_dict

def get_pixel_map(gene, spot_diff_dict, 
               locs, img, radius, smoothing_sigma, 
               disk_mask, mask_put):
    cnts = spot_diff_dict[gene]
    if img.ndim == 2:
        H, W = img.shape
    elif img.ndim == 3:
        H, W, _ = img.shape
    pixel_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)  # for normalization
    radius = int(np.round(radius))
    # Create mask
    if disk_mask:
        mask = get_disk_mask(radius)
    else:
        mask = np.ones((2 * radius, 2 * radius), dtype=bool)

    patch_offset = np.stack(np.where(mask), axis=-1) - radius

    # Place values into pixel map
    for (y, x), val in zip(locs, cnts):
        for dy, dx in patch_offset:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                pixel_map[ny, nx] += val
                count_map[ny, nx] += 1

    # Normalize overlapping areas
    with np.errstate(invalid='ignore'):
        pixel_map = np.divide(pixel_map, count_map, out=np.zeros_like(pixel_map), where=count_map > 0)

    # Apply Gaussian smoothing
    if smoothing_sigma > 0:
        pixel_map = gaussian_filter(pixel_map, sigma=smoothing_sigma)

    h, w = min(mask_put.shape[0], pixel_map.shape[0]), min(mask_put.shape[1], pixel_map.shape[1])
    mask_put = mask_put[:h, :w]
    pixel_map = pixel_map[:h, :w]
    pixel_map[~mask_put] = np.nan

    return pixel_map

def plot_error_single(gene, spot_diff_dict, 
               prefix, locs, img, radius, smoothing_sigma, 
               disk_mask, mask_put, colorbar):
    
    pixel_map = get_pixel_map(gene, spot_diff_dict, 
               locs, img, radius, smoothing_sigma, 
               disk_mask, mask_put)
    outfolder = f'{prefix}/local_uncertainty_plots/'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    outfile = f'{outfolder}/{gene}.png'
    
    if colorbar:
        plot_with_colorbar(pixel_map, outfile)
    else:
        plot_without_colorbar(pixel_map, outfile)
    print(f'Figure saved to {outfile}')

def plot_error(prefix, locs, spot_diff_dict, smoothing_sigma, colorbar = True):
    
    locs_edit = scale_locs(locs)
    img = load_image(f'{prefix}he.jpg')
    mask_put = load_image(f'{prefix}mask-small.png') > 0
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    radius = ori_radius / 16
    for gene in spot_diff_dict.keys():
        plot_error_single(gene, spot_diff_dict, prefix = prefix, locs=locs_edit, img = img, 
                          radius=radius, smoothing_sigma=smoothing_sigma, 
                          disk_mask=True, mask_put = mask_put, colorbar = colorbar)
        
def get_spot_metric(locs, df_impute_spot, cnts_truth, gene, mask, radius):
    # normalization
    truth_norm = max_min_norm(cnts_truth[gene].to_numpy())
    pred_norm = max_min_norm(df_impute_spot[gene].to_numpy())
    # compute metrics
    _, rmse_val, pcc_val, spearman_val = compute_metrics(truth_norm, pred_norm)
    
    # Compute ssim
    cnts_truth_pixel_map = get_pixel_map(gene, cnts_truth, locs, img = mask, 
                                         radius = radius, smoothing_sigma = 0, 
                                         disk_mask = True, mask_put = mask)
    
    cnts_impute_pixel_map = get_pixel_map(gene, df_impute_spot, locs, img = mask, 
                                          radius = radius, smoothing_sigma = 0, 
                                          disk_mask = True, mask_put = mask)
    
    ssim_val = ssim(cnts_truth_pixel_map, cnts_impute_pixel_map,
                    data_range = cnts_truth_pixel_map.max() - cnts_truth_pixel_map.min())
    return {
        'gene': gene,
        'SSIM': np.round(ssim_val, 4),
        'RMSE': np.round(rmse_val, 4),
        'PCC': np.round(pcc_val, 4),
        'Spearman': np.round(spearman_val, 4)
    }

def get_spot_metric_parallel(prefix, locs, df_impute_spot, cnts_truth, gene_names, max_workers = 64):
    result = []
    mask_put = load_image(prefix + "mask-small.png") > 0 
    locs_edit = scale_locs(locs)
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    radius = ori_radius / 16
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_spot_metric, locs_edit, df_impute_spot, cnts_truth, gene, mask_put, radius): gene for gene in gene_names}
        
        for future in tqdm(futures, total=len(futures), desc="Compute metrics at spot level for each genes"):
            res = future.result()
            if res is not None:
                result.append(res)
    
    return pd.DataFrame(result)


def plot_pixel_diff(prefix, genes, truth_path, impute_path, colorbar = True):
    mask = load_image(prefix + "mask-small.png") > 0 

    for gene in genes:
        truth_file = os.path.join(truth_path, f"{gene}.pickle")    
        if not os.path.exists(truth_file):
            print(f"Skipping {gene}: file missing.")
            return None
        truth_arr = load_pickle(truth_file)
        pred_arr = load_pickle(os.path.join(impute_path, f"{gene}.pickle") )
        h = min(truth_arr.shape[0], pred_arr.shape[0], mask.shape[0])
        w = min(truth_arr.shape[1], pred_arr.shape[1], mask.shape[1])

        truth_arr = truth_arr[:h,:w]
        pred_arr = pred_arr[:h,:w]
        mask = mask[:h,:w]
        # mask
        truth_arr[~mask] = np.nan
        pred_arr[~mask] = np.nan
    
        diff_arr = np.sqrt(truth_arr) - np.sqrt(pred_arr)
        outfolder = f'{prefix}/local_uncertainty_plots/'
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        outfile = f'{outfolder}/{gene}_pixel_diff.png'
        if colorbar:
            plot_with_colorbar(diff_arr, outfile)
        else:
            plot_without_colorbar(diff_arr, outfile)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--cnts_val_name', type=str)
    parser.add_argument('--locs_val_name', type=str)
    parser.add_argument('--smoothing_sigma', type=int, default=3)
    args = parser.parse_args()
    return args

def main():
    
    args = get_args()
    
    prefix = args.prefix
    cnts_val_name = args.cnts_val_name
    locs_val_name = args.locs_val_name
    smoothing_sigma = args.smoothing_sigma
    
    cnts_val = load_csv(f'{prefix}{cnts_val_name}')
    locs_val = load_csv(f'{prefix}{locs_val_name}')
    embs_val = load_pickle(f'{prefix}embeddings-combined-val.pickle')
    
    with open(f'{prefix}gene-names-groups.txt', "r") as f:
        loaded_list = json.load(f)
        gene_names = flatten(loaded_list)
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    radius = ori_radius / 16
    mask = get_disk_mask(radius)
    # Impute based on validation embedding 
    pred_arr_val = predict_custom(prefix, cnts_val, locs_val, embs_val, gene_names, radius)
    df_impute_spot = pixel_to_spot_parallel(prefix, pred_arr_val, locs_val, mask, gene_names)
    print("Finish converting validation prediction from pixel to spot")
    
    # spot_diff_dict = get_spot_diff_parallel(df_impute_spot, cnts_val, gene_names)
    # plot_error(prefix, locs_val, spot_diff_dict, smoothing_sigma)
    # print(f"Finish plotting local error quantification")
    
    spot_rank = get_spot_metric_parallel(prefix, locs_val, df_impute_spot, cnts_val, gene_names)
    spot_rank.to_csv(f'{prefix}/df_spot_rank.csv')
    print('Finish compute spot rank')

if __name__ == '__main__':
    main()
