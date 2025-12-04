import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    args = parser.parse_args()
    return args

def main():
    
    # ========== Parameters ==========
    args = get_args()
    prefix = args.prefix
    
    pickle_dir = prefix + "Cell_proportion/cnts-super"
    mask_path = prefix + "mask-small.png"
    output_png = prefix + "predicted_celltype.png"
    
    # ========== 1. Load all the .pickle files ==========
    file_list = sorted([f for f in os.listdir(pickle_dir) if f.endswith(".pickle")])
    
    celltypes = [os.path.splitext(f)[0] for f in file_list]  
    print("Celltypes detected:", celltypes)
    
    mats = []
    for f in file_list:
        with open(os.path.join(pickle_dir, f), "rb") as handle:
            mat = pickle.load(handle)  # H x W
            mats.append(mat)
    
    mats = np.stack(mats, axis=0)  # [num_celltypes, H, W]
    print("Stacked matrix shape:", mats.shape)
    
    import matplotlib.patches as mpatches
    # ========== 2. Predict cell type of pixel ==========
    pred_ids = np.argmax(mats, axis=0)
    
    # ========== 3. Load tissue mask ==========
    mask = np.array(Image.open(mask_path).convert("L"))
    mask_bin = mask > 127
    pred_ids_masked = np.where(mask_bin, pred_ids, -1)
    
    # ========== 4. Visualization ==========
    cmap = plt.cm.get_cmap("tab20", len(celltypes))
    colors = np.zeros((*pred_ids_masked.shape, 3), dtype=np.uint8)
    
    for idx in range(len(celltypes)):
        colors[pred_ids_masked == idx] = (np.array(cmap(idx)[:3]) * 255).astype(np.uint8)
    
    colors[pred_ids_masked == -1] = (0, 0, 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(colors)
    
    patches = [
        mpatches.Patch(color=cmap(i), label=ct)
        for i, ct in enumerate(celltypes)
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Result saved as:", output_png)
    
    celltype2id = {ct: i for i, ct in enumerate(celltypes)}
    print("celltype2id:", celltype2id)

if __name__ == '__main__':
    main()