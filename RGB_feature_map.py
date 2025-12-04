from einops import reduce
import numpy as np
from PIL import Image

def main(prefix):
    img = np.array(Image.open(f'{prefix}he.jpg').convert('RGB'))
    h, w, _ = img.shape
    assert h % 16 == 0 and w % 16 == 0, "Image height and width must be divisible by 16"

    rgb_emb = np.stack([
        reduce(
            img[..., i].astype(np.float32) / 255.0,
            '(h h_block) (w w_block) -> h w', 'mean',
            h_block=16, w_block=16)
        for i in range(3)
    ])  # shape: (3, H//16, W//16)

    return rgb_emb
