import argparse
import os
import pandas as pd
import numpy as np
from einops import reduce
from impute_slide import get_data
from scipy.ndimage import gaussian_filter
from utils import load_image, load_csv, read_lines, read_string, save_image, get_disk_mask, load_pickle

import matplotlib.pyplot as plt

# Convert prediction from pixel to spot
def pixel_to_spot(prefix, impute_folder_name, locs, gene_names):

    df_impute_spot = pd.DataFrame()
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    mask = get_disk_mask(ori_radius/16)
    for gn in gene_names:
        path_to_cnts_supr = f'{prefix}{impute_folder_name}/{gn}.pickle'
        if not os.path.exists(path_to_cnts_supr):
            print(f'Results not found for {gn}')
            continue
        impute = np.round(load_pickle(f'{prefix}{impute_folder_name}/{gn}.pickle')) # path_to_pixel
        shape = np.array(mask.shape) # mask shape is 7 X 7
        mask = np.ones_like(mask, dtype=bool)
        center = shape // 2  #
        r = np.stack([-center, shape-center], -1)  # offset
        impute_spot_list = []
        for s in locs:
            patch = impute[
                    s[0]+r[0][0]:s[0]+r[0][1],
                    s[1]+r[1][0]:s[1]+r[1][1]]
            x = np.nansum(patch[mask])
            #x = np.mean(patch[mask]) # Take means or sums, should have simialr pattern
            impute_spot_list.append(x)
        df_tmp = pd.DataFrame({gn: impute_spot_list})
        df_impute_spot = pd.concat([df_impute_spot, df_tmp], axis = 1)
    locs_ori = load_csv(f'{prefix}locs.csv')
    df_impute_spot.index = locs_ori.index
    
    return df_impute_spot

class update_exprs_neighbor:
    def __init__(self, cnts, locs, gene_names):
        '''
        locs: (N,2) array consist of location info
        gene_names: list of gene names,
        cnts: matrix, with gene names as columns, locs as rows
        distance: numeric, distance from center to sorrounding
        dist_method: spatial distance methods

        Output: Pandas dataframe with updated expreesion considering spatial distance (gene names as columns, locs as rows)

        '''
        self.cnts = cnts
        self.locs = locs
        self.gene_names = gene_names

    def update_exprs(self, dist_method, **kwargs):
        available_dist_method = ['average', 'exp_decay', 'gaussian_kernel']
        if dist_method not in available_dist_method:
            raise ValueError('Available methods are: ' + ','.join(available_dist_method))
        df_out = pd.DataFrame()
        if dist_method == 'average':
            distance = kwargs.get('distance', 30)
            return self.update_exprs_average(df_out, distance)

        elif dist_method == 'exp_decay':
            lambda_param = kwargs.get('lambda_param', 0.1)
            return self.update_exprs_exp_decay(df_out, lambda_param)

        elif dist_method == 'gaussian_kernel':
            sigma = kwargs.get('sigma', 1)
            return self.update_exprs_gaussain_kernal(df_out, sigma)

    def update_exprs_average(self, df_out, distance):
        locs_view = self.locs.view([('', self.locs.dtype)] * self.locs.shape[1])
        for gn in self.gene_names:
            cnts_spot_round = []

            for center in self.locs:
                # Compute L2 norm (Euclidean distance) along axis=1, i.e., for each row (each vector):
                dists = np.linalg.norm(self.locs - center, axis=1)
                locs_sub = self.locs[dists <= distance]
                # Convert to sets of rows for matching
                locs_sub_view = locs_sub.view([('', locs_sub.dtype)] * locs_sub.shape[1])
                # Find matches
                mask = np.isin(locs_view, locs_sub_view)
                indices = np.where(mask.ravel())[0]
                new_exprs = np.mean(self.cnts[gn].iloc[indices])
                cnts_spot_round.append(new_exprs)
            df_tmp = pd.DataFrame({gn: cnts_spot_round})
            df_out = pd.concat([df_out, df_tmp], axis = 1)
        df_out.index = self.cnts.index
        return df_out

    def update_exprs_exp_decay(self, df_out, lambda_param):
        for gn in self.gene_names:
            cnts_spot_round = []
            for center in self.locs:
                dists = np.linalg.norm(self.locs - center, axis=1)
                weights = np.exp(-lambda_param * dists)
                values = self.cnts[gn].iloc[:]
                new_exprs = np.sum(weights * values) / np.sum(weights)
                cnts_spot_round.append(new_exprs)
            df_tmp = pd.DataFrame({gn: cnts_spot_round})
            df_out = pd.concat([df_out, df_tmp], axis = 1)
        df_out.index = self.cnts.index
        return df_out

    def update_exprs_gaussain_kernal(self, df_out, sigma):
        for gn in self.gene_names:
            cnts_spot_round = []

            for center in self.locs:
                dists = np.linalg.norm(self.locs - center, axis=1)
                # Gaussian kernel weights
                weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))
                values = self.cnts[gn].iloc[:]
                # Weighted mean using Gaussian kernel
                new_exprs = np.sum(weights * values) / np.sum(weights)
                cnts_spot_round.append(new_exprs)

            df_tmp = pd.DataFrame({gn: cnts_spot_round})
            df_out = pd.concat([df_out, df_tmp], axis = 1)
        df_out.index = self.cnts.index
        return df_out

def normalization(data):
    _range = np.nanmax(data) - np.nanmin(data)
    return (data - np.nanmin(data)) / _range

def get_spot_diff(df_impute_cat, df_truth_cat, df_neighbor_gene, gene_name):

    df_spot_diff = pd.DataFrame(
            {gene_name: np.abs(df_impute_cat - df_truth_cat)/
                    (np.array(df_neighbor_gene[gene_name]) + 1e-3)})

    second_largest = df_spot_diff[gene_name].drop_duplicates().nlargest(2).iloc[-1]
    df_spot_diff[gene_name] = df_spot_diff[gene_name].replace(1000, second_largest + 0.01)
    return df_spot_diff

def get_spot_diff_all(gene_names, cnts_val, df_impute_spot, df_smooth_spot):
    spot_diff_dict = {}
    for gn in gene_names:
        df_impute_norm = normalization(df_impute_spot[gn])
        df_truth_norm = normalization(cnts_val[gn])
        spot_diff_dict[gn] = get_spot_diff(df_impute_norm,
                                           df_truth_norm,
                                           df_smooth_spot, gn) 
    return spot_diff_dict
        
def smooth_spots_to_pixel_map(locs, cnts, img, radius, mask_put, outfile = None,
                              smoothing_sigma=2.0, disk_mask=True,
                              normalize=True):
    """
    Smooths spot-level data into a dense pixel-level heatmap.

    Args:
        locs: (N, 2) array of (y, x) coordinates.
        cnts: (N,) array of spot-level values.
        image_shape: shape of output image (H, W)
        radius: size of disk or patch to apply around each spot.
        smoothing_sigma: Gaussian smoothing parameter.
        disk_mask: if True, apply values in disk; otherwise square.
        normalize: if True, normalize the output map to [0, 1].

    Returns:
        pixel_map: 2D numpy array of smoothed spot signal.
    """
    H, W, _ = img.shape
    pixel_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)  # for normalization

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
    # Normalize to [0, 1] if requested
    if normalize:
        pixel_map -= np.nanmin(pixel_map)
        pixel_map /= np.nanmax(pixel_map) + 1e-12
    cmap = plt.get_cmap('turbo')

    img = cmap(pixel_map)[..., :3]
    img[~mask_put] = 1.0
    img = (img * 255).astype(np.uint8)
    save_image(img, outfile)

def plot_smoothed_spot_diff(prefix, spot_diff_dict):
    locs = load_csv(f'{prefix}locs.csv')
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs /= 16
    locs = locs.round().astype(int)
    img = load_image(f'{prefix}he.jpg')
    mask_put = load_image(f'{prefix}mask-small.png') > 0

    for gene_name, cnts_diff in spot_diff_dict.items():
        smooth_spots_to_pixel_map(locs=locs,
                cnts=cnts_diff.to_numpy(),
                img=img,
                radius=5,
                smoothing_sigma=5,
                disk_mask=True, normalize = True,
                mask_put = mask_put,
                outfile = f'{prefix}local_uncertainty_plots/{gene_name}.png')
        
# Get global rank
from sklearn.metrics import average_precision_score, mean_squared_error
from get_SSIM import structural_similarity
def plot_spots_ssim(
        img, cnts, locs, radius, cmap='magma',
        weight=0.8, disk_mask=True, standardize_img=False):
    cnts = cnts.astype(np.float32)

    img = img.astype(np.float32)
    img /= 255.0

    if standardize_img:
        if np.isclose(0.0, np.nanstd(img, (0, 1))).all():
            img[:] = 1.0
        else:
            img -= np.nanmin(img)
            img /= np.nanmax(img) + 1e-12

    cnts -= np.nanmin(cnts)
    cnts /= np.nanmax(cnts) + 1e-12

    cmap = plt.get_cmap(cmap)
    if disk_mask:
        mask_patch = get_disk_mask(radius)
    else:
        mask_patch = np.ones((radius*2, radius*2)).astype(bool)
    indices_patch = np.stack(np.where(mask_patch), -1)
    indices_patch -= radius
    for ij, ct in zip(locs, cnts):
        color = np.array(cmap(ct)[:3])
        indices = indices_patch + ij
        img[indices[:, 0], indices[:, 1]] *= 1 - weight
        img[indices[:, 0], indices[:, 1]] += color * weight
    img = (img * 255).astype(np.uint8)
    return img

class spot_rank:
    def __init__(self, cnts, df_impute_spot, gene_names):
        self.cnts = cnts
        self.df_impute_spot = df_impute_spot
        self.gene_names = gene_names

    def get_auc_rmse(self):
        auc = []
        rmse = []
        for gene in self.gene_names:
            spot_truth = self.cnts[gene]
            impute_val = self.df_impute_spot[gene]
            assert np.all(spot_truth.index == impute_val.index)
            spot_truth_norm = spot_truth if np.all(spot_truth == spot_truth.iloc[0]) else normalization(spot_truth)
            impute_val_norm = impute_val if np.all(impute_val == impute_val.iloc[0]) else normalization(impute_val)            
            # RMSE
            rmse.append(mean_squared_error(spot_truth_norm, impute_val_norm))
            # AUC
            y_true = (spot_truth_norm > 0).astype(int)
            auc.append(average_precision_score(y_true, impute_val_norm))
        return pd.DataFrame({'Gene': self.gene_names, 'RMSE_spot': rmse, 'AUC_spot': auc})

    def get_ssim(self, prefix):
        factor = 16
        locs = load_csv(f'{prefix}locs.csv')
        locs = locs.astype(float)
        locs = np.stack([locs['y'], locs['x']], -1)
        locs /= factor
        locs = locs.round().astype(int)
        img = load_image(f'{prefix}he.jpg')
        img = reduce(
        img.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factor, w=factor).astype(np.uint8)
        white_img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
        
        radius = int(read_string(f'{prefix}radius.txt'))
        radius = np.round(radius / factor).astype(int)
        ssim = []
        for gene in self.gene_names:
            img_truth = plot_spots_ssim(white_img, self.cnts[gene], locs = locs, radius = radius, cmap = 'turbo')
            img_impute = plot_spots_ssim(white_img, self.df_impute_spot[gene], locs = locs, radius = radius, cmap = 'turbo')
            ssim.append(structural_similarity(img_truth, img_impute, channel_axis=-1))
        return pd.DataFrame({'Gene': self.gene_names, 'SSIM_spot': ssim})

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--impute_folder_name', type=str, default='cnts-super')
    parser.add_argument('--cnts_val_name', type=str)
    parser.add_argument("--smooth_method", type=str, 
                        choices=['average', 'exp_decay', 'gaussian_kernel'],   # allowed values
                        default='exp_decay')
    args = parser.parse_args()
    return args

def main():
    
    args = get_args()
    
    prefix = args.prefix
    impute_folder_name = args.impute_folder_name
    cnts_val_name = args.cnts_val_name
    smooth_method = args.smooth_method
    print(f"Neighborhood smooth method is: {smooth_method}")
    cnts_val = load_csv(f'{prefix}{cnts_val_name}')
    
    _, _, locs = get_data(prefix)
    gene_names = read_lines(f'{prefix}gene-names.txt')
    
    df_impute_spot = pixel_to_spot(prefix, impute_folder_name, locs, gene_names)
    print("Finish converting prediction from pixel to spot")
    
    print("Updating expression based on neghiborhood values and starting local uncertainty computation")
    spot_exprs = update_exprs_neighbor(cnts_val, locs, gene_names)

    df_smooth_spot = spot_exprs.update_exprs(smooth_method)
    spot_diff_dict = get_spot_diff_all(gene_names, cnts_val, df_impute_spot, df_smooth_spot)
    # Get local uncertainty plot
    print("Finishing local uncertainty computation")
    plot_smoothed_spot_diff(prefix, spot_diff_dict)
    print(f"Finish plotting local uncertainty. Results in {prefix}local_uncertain_plots/")

    # Get global rank
    print('Start computing global ranks.')
    global_metrics = spot_rank(cnts_val, df_impute_spot, gene_names)
    df_global_auc_rmse = global_metrics.get_auc_rmse()
    df_global_ssim = global_metrics.get_ssim(prefix)
    df_global = pd.merge(df_global_auc_rmse, df_global_ssim, on = "Gene")
    df_global.to_csv(f'{prefix}global_rank_spots.csv', index = False)
    print(f"Finish computing global ranks. Results in {prefix}global_rank_spots.csv")

if __name__ == '__main__':
    main()
