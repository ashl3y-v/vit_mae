import os
import torch as T
from vit import ViT
from matplotlib import pyplot as plt
import torchvision as tv
from torchvision import transforms

T.manual_seed(0)

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

T.cuda.empty_cache()

vit = ViT(dtype=dtype, device=device)
vit.eval()
if os.path.isfile(vit.file):
    print("Loading model", vit.file)
    vit.load()

proc = transforms.Compose(
    [
        transforms.Resize([256, 256], antialias=True),
        transforms.Lambda(lambda x: x if x.mode == "RGB" else x.convert("RGB")),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(T.bfloat16),
        transforms.Normalize(0, 1),
    ]
)

dataset = tv.datasets.INaturalist(
    "./data/",
    version="2021_train_mini",
    target_type="class",
    transform=proc,
    # download=True,
)

x = dataset[1][0].to(dtype=dtype, device=device)

b, w_p, h_p, patch_size, c = 1, 16, 16, 16, 3

# x_0 = T.arange(0, 1, 1 / 256, dtype=dtype, device=device).reshape([1, 256])
# x_1 = (
#     T.arange(0, 1, 1 / 256, dtype=dtype, device=device)
#     .reshape([256, 1])
#     .expand([c, 256, 1])
# )
# f_0 = 4
# f_1 = 4
# x = x_0 * (x_0 * f_0).sin() + x_1 * (x_1 * f_1).cos()

# n = x.unsqueeze(0).reshape([b, w_p * h_p, patch_size * patch_size * c])
# n = x.unsqueeze(0).reshape([b, c, w_p, patch_size, h_p, patch_size])
# n = x.unsqueeze(0).reshape([b, c, w_p * patch_size, h_p * patch_size])
l = (
    x.unsqueeze(1)
    .unsqueeze(3)
    .reshape([b, c, w_p, patch_size, h_p, patch_size])
    .permute([0, 2, 4, 3, 5, 1])
    .flatten(3)
    .flatten(1, 2)
)

l[:, 32:64, 64:96] = 1
l[:, 200:208, :] = 1
l[:, :, 400:408] = 1

r = (
    l.reshape([b, w_p, h_p, patch_size, patch_size, c])
    .permute([0, 5, 1, 3, 2, 4])
    .flatten(4, 5)
    .flatten(2, 3)
)

r = r.squeeze(0)

plt.matshow(l.permute([1, 2, 0]).cpu().float())
plt.matshow(r.permute([1, 2, 0]).cpu().float())
plt.show()
