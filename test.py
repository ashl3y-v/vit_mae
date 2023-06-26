from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np
import torch as T
import transformers
from PIL import Image
import requests
from vitmae import ViTMAE

# T.manual_seed(1)
T.seed()

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

args = {"mask_ratio": 0.75}

url = "https://www.bbvaopenmind.com/wp-content/uploads/2018/05/feynmanfisica.jpeg"

images = Image.open(requests.get(url, stream=True).raw)

ckpt = "facebook/vit-mae-base"
vitmae = ViTMAE(ckpt=ckpt, dtype=dtype, device=device)

outputs, pixel_values = vitmae(images, return_pixel_values=True)

# test

im_mean = np.array(vitmae.image_processor.image_mean)
im_std = np.array(vitmae.image_processor.image_std)

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(T.clip((image * im_std + im_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

y = vitmae.transformer.unpatchify(outputs.logits)
y = T.einsum('nchw->nhwc', y).detach().cpu()

# visualize the mask
mask = outputs.mask.detach()
mask = mask.unsqueeze(-1).repeat(1, 1, vitmae.transformer.config.patch_size**2 *3)  # (N, H*W, p*p*3)
mask = vitmae.transformer.unpatchify(mask)  # 1 is removing, 0 is keeping
# here
mask = T.einsum('nchw->nhwc', mask).detach().cpu()

x = T.einsum('nchw->nhwc', pixel_values).to(dtype=T.float32, device="cpu")

# masked image
im_masked = x * (1 - mask)

# MAE reconstruction pasted with visible patches
im_paste = x * (1 - mask) + y * mask

# make the plt figure larger
plt.rcParams['figure.figsize'] = [24, 24]

selected_image = 0 # only 1 image (for now)
plt.subplot(1, 3, 1)
show_image(x[selected_image], "original")

plt.subplot(1, 3, 2)
show_image(im_masked[selected_image], "masked")

# plt.subplot(1, 4, 3)
# show_image(y[selected_image], "reconstruction")

plt.subplot(1, 3, 3) # 4
show_image(im_paste[selected_image], "reconstruction + visible")

plt.show()
