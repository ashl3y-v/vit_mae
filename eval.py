from datasets import load_dataset
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from vitmae import ViTMAE
import os
import random
import torch as T
import torchvision as tv
import math

T.manual_seed(0)

T.backends.cudnn.benchmark = True
T.autograd.set_detect_anomaly(True)
T.backends.cuda.matmul.allow_tf32 = True
T.backends.cudnn.allow_tf32 = True

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

T.cuda.empty_cache()

vitmae = ViTMAE(dtype=dtype, device=device)
if os.path.isfile(vitmae.file):
    print("Loading model", vitmae.file)
    vitmae.load()

proc = transforms.Compose(
    [
        transforms.Resize([256, 256], antialias=True),
        transforms.Lambda(lambda x: x if x.mode == "RGB" else x.convert("RGB")),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(T.uint8),
    ]
)

v_set = tv.datasets.ImageNet("./data/", split="val")

y_hat = vitmae(t_x, mask=True)
y_hat = vitmae.conv_t(y_hat)

loss = vitmae.loss(t_x, y_hat)

y_hat = (
    y_hat[0]
    .permute(2, 1, 0)
    .reshape(
        [
            3,
        ]
    )
)

plt.matshow(y_hat)
plt.show()
