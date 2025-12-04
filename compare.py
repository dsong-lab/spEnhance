import os
import numpy as np
import cv2
import pandas as pd
import math
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.metrics import mutual_info_score
from tqdm import tqdm  # ✅ 用于进度条显示

# Moran's I support
try:
    from esda.moran import Moran
    from libpysal.weights import lat2W
    moran_available = True
except ImportError:
    moran_available = False

# === CONFIG ===
truth_dir = 'brain_cancer/Truth/cnts-super-plots'
pred_dir = 'brain_cancer/Merged-aug-s3-loss/cnts-super-plots'
output_csv = 'brain_cancer/Results/Merged-aug-s3-loss.csv'

# === UTILITIES ===

def normalization(data):
    _range = (np.max(data) - np.min(data) ) + 1e-6
    return (data - np.min(data)) / _range

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return np.nan_to_num(img)

def jensen_shannon_divergence(p, q):
    p = p.flatten() + 1e-8
    q = q.flatten() + 1e-8
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

# === MAIN METRIC LOOP ===

results = []
gene_names = [f[:-4] for f in os.listdir(truth_dir) if f.endswith('.png')]

# ✅ 使用 tqdm 包裹循环
for gene in tqdm(gene_names, desc="🔍 Processing genes"):
    truth_path = os.path.join(truth_dir, f'{gene}.png')
    pred_path = os.path.join(pred_dir, f'{gene}.png')

    if not os.path.exists(pred_path):
        print(f"❌ Missing prediction for {gene}, skipping.")
        continue

    try:
        truth = load_image(truth_path)
        pred = load_image(pred_path)

        # preprocess
        truth_processed = truth[:1463,:]
        pred_processed = pred[:1463,:]
        
        # Normalize for some metrics
        truth = normalization(truth_processed)
        pred = normalization(pred_processed)

        # Metric calculations
        rmse_val = math.sqrt((np.square(pred - truth)).mean(axis=None))
        mse_val = mean_squared_error(truth, pred)
        r2_val = r2_score(truth.flatten(), pred.flatten())
        ssim_val = ssim(truth*255, pred*255, data_range=255)
        psnr_val = psnr(truth*255, pred*255, data_range=255)
        pcc_val = pearsonr(truth.flatten(), pred.flatten())[0]
        spearman_val = spearmanr(truth.flatten(), pred.flatten())[0]
        mi_val = mutual_info_score(None, None, contingency=np.histogram2d(truth.flatten(), pred.flatten(), bins=64)[0])
        js_val = jensen_shannon_divergence(truth, pred)
        exp = np.sum(truth)

        # AUC using Otsu binarization
        _, truth_bin = cv2.threshold(truth_processed, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pred_flat = MinMaxScaler().fit_transform(pred_processed.reshape(-1, 1)).flatten()
        if len(np.unique(truth_bin)) == 2:
            auc_val = roc_auc_score(truth_bin.flatten(), pred_flat)
        else:
            auc_val = np.nan

        # Moran's I (optional)
        if moran_available:
            w = lat2W(*truth.shape)
            moran_truth = Moran(truth.flatten(), w).I
            moran_pred = Moran(pred.flatten(), w).I
        else:
            moran_truth = moran_pred = np.nan

        results.append([
            gene, rmse_val, mse_val, r2_val, ssim_val, psnr_val, pcc_val,
            spearman_val, mi_val, js_val, auc_val, moran_truth, moran_pred, exp
        ])

    except Exception as e:
        print(f"❌ Error processing {gene}: {e}")

# === SAVE CSV ===
df = pd.DataFrame(results, columns=[
    "Gene", "RMSE", "MSE", "R2", "SSIM", "PSNR", "PCC", "Spearman",
    "MI", "JS_Div", "AUC", "Moran_I_Truth", "Moran_I_Pred", "True_expression"
])
df.set_index("Gene", inplace=True)
df.to_csv(output_csv)
print(f"\n✅ Metrics saved to {output_csv}")
