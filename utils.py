import itertools
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
import yaml

import cv2 as cv
Image.MAX_IMAGE_PIXELS = None


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img


def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    return read_lines(filename)[0]


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def write_string(string, filename):
    return write_lines([string], filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df

def load_csv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df

def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)
    print(filename)

def save_csv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = ','
    x.to_csv(filename, **kwargs)
    print(filename)

def load_yaml(filename, verbose=False):
    with open(filename, 'r') as file:
        content = yaml.safe_load(file)
    if verbose:
        print(f'YAML loaded from {filename}')
    return content


def save_yaml(filename, content):
    with open(filename, 'w') as file:
        yaml.dump(content, file)
    print(file)


def join(x):
    return list(itertools.chain.from_iterable(x))


def get_most_frequent(x):
    # return the most frequent element in array
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]

def impute_missing(x, mask, radius=3, method='ns'):

    method_dict = {
            'telea': cv.INPAINT_TELEA,
            'ns': cv.INPAINT_NS}
    method = method_dict[method]

    x = x.copy()
    if x.dtype == np.float64:
        x = x.astype(np.float32)

    x[mask] = 0
    mask = mask.astype(np.uint8)

    expand_dim = np.ndim(x) == 2
    if expand_dim:
        x = x[..., np.newaxis]
    channels = [x[..., i] for i in range(x.shape[-1])]
    y = [cv.inpaint(c, mask, radius, method) for c in channels]
    y = np.stack(y, -1)
    if expand_dim:
        y = y[..., 0]

    return y


def smoothen(
        x, size, kernel='uniform', backend='cv', mode='mean',
        impute_missing_values=True, device='cuda'):

    if x.ndim == 3:
        expand_dim = False
    elif x.ndim == 2:
        expand_dim = True
        x = x[..., np.newaxis]
    else:
        raise ValueError('ndim must be 2 or 3')

    mask = np.isfinite(x).all(-1)
    if (~mask).any() and impute_missing_values:
        x = impute_missing(x, ~mask)

    if kernel == 'uniform':
        if backend == 'cv':
            kernel = np.ones((size, size), np.float32) / size**2
            y = cv.filter2D(
                    x, ddepth=-1, kernel=kernel,
                    borderType=cv.BORDER_REFLECT)
            if y.ndim == 2:
                y = y[..., np.newaxis]
        else:
            raise ValueError('backend must be cv')
    else:
        raise ValueError('kernel must be uniform')

    if not mask.all():
        y[~mask] = np.nan

    if expand_dim and y.ndim == 3:
        y = y[..., 0]
    return y

def get_disk_mask(radius, boundary_width=None):
    radius_ceil = np.array(radius).astype(int)
    locs = np.meshgrid(
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    locs = np.stack(locs, -1)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    return isin

def plot_with_colorbar(pixel_map, outfile):
    # Compute scaling based on 0.9999 quantile of absolute values
    q = np.nanquantile(np.abs(pixel_map), 0.9999)

    # Avoid division by zero
    if q == 0:
        q = 1e-6

    # Normalize into [-1, 1]
    norm_map = pixel_map / q
    norm_map = np.clip(norm_map, -1, 1)  # values outside -> clipped to [-1,1]

    # Custom colormap: blue → gray → red (center at 0)
    custom_cmap = LinearSegmentedColormap.from_list(
        "blue_gray_red",
        [
            (0.0, (0.0, 0.0, 1.0)),   # -1 -> blue
            (0.5, (0.6, 0.6, 0.6)),   #  0 -> gray
            (1.0, (1.0, 0.0, 0.0)),   # +1 -> red
        ]
    )
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    dpi = 200
    fig, ax = plt.subplots(figsize=(20, 10), dpi=dpi)

    im = ax.imshow(norm_map, cmap=custom_cmap, norm=norm)
    ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.10
    )

    ticks = [-1, -0.75, -0.5, -0.25, 
              0,
              0.25, 0.5, 0.75, 1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
    cbar.set_label("Scaled Value", fontsize=12)

    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_without_colorbar(pixel_map, outfile):
    # Compute scaling based on 0.9999 quantile of abs values
    q = np.nanquantile(np.abs(pixel_map), 0.9999)
    if q == 0:
        q = 1e-6

    # Normalize to [-1, 1]
    norm_map = pixel_map / q
    norm_map = np.clip(norm_map, -1, 1) # cap all values from -1 to 1

    # Prepare RGB array
    H, W = norm_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Map from [-1,1] to [0,1]
    t = (norm_map + 1) / 2.0  

    # Blue → Gray → Red interpolation
    blue = np.array([0, 0, 255], dtype=float)
    gray = np.array([153, 153, 153], dtype=float)
    red  = np.array([255, 0, 0], dtype=float)

    # For t < 0.5: interpolate blue to gray
    mask1 = t <= 0.5
    ratio1 = t[mask1] / 0.5  # 0: blue, 1: gray
    rgb[mask1] = (blue[None,:] * (1 - ratio1[:,None]) +
                  gray[None,:] * ratio1[:,None])

    # For t >= 0.5: interpolate gray to red
    mask2 = t > 0.5
    ratio2 = (t[mask2] - 0.5) / 0.5  # 0: gray, 1: red
    rgb[mask2] = (gray[None,:] * (1 - ratio2[:,None]) +
                  red[None,:] * ratio2[:,None])

    nan_mask = np.isnan(pixel_map)
    rgb[nan_mask] = [255, 255, 255] # keep NaN as white
    rgb = rgb.astype(np.uint8)
    Image.fromarray(rgb).save(outfile)
