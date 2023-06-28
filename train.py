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

lr = 3e-3
clip = 24
epochs = 4096
save_interval = 128
t_batch_size = 62
v_batch_size = 16

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

dataset = tv.datasets.INaturalist(
    "./data/",
    version="2021_train_mini",
    target_type="class",
    transform=proc,
    # download=True,
)
t_set, v_set = T.utils.data.random_split(dataset, [0.8, 0.2])

t_loader = iter(
    T.utils.data.DataLoader(
        t_set,
        batch_size=t_batch_size,
        shuffle=True,
        num_workers=int(os.system("cat /proc/cpuinfo | grep processor | wc -l")),
    )
)

v_loader = iter(
    T.utils.data.DataLoader(
        v_set,
        batch_size=v_batch_size,
        shuffle=True,
        num_workers=int(os.system("cat /proc/cpuinfo | grep processor | wc -l")),
    )
)

print("Dataset loaded")

params = vitmae.parameters()

optim = T.optim.AdamW(params, lr=lr, fused=True)

u = 512
u_e = 4 * u
lr_sch = T.optim.lr_scheduler.SequentialLR(
    optim,
    schedulers=[
        T.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, u),
        T.optim.lr_scheduler.LinearLR(optim, 0.125, 0, total_iters=u_e),
    ],
    milestones=[epochs - u_e],
)

t_losses = T.tensor([])
v_losses = T.tensor([])
if os.path.isfile("t_losses.pt"):
    t_losses = T.load("stats/t_losses.pt").to(dtype=T.float32, device="cpu")
if os.path.isfile("v_losses.pt"):
    v_losses = T.load("stats/v_losses.pt").to(dtype=T.float32, device="cpu")

for i in range(epochs):
    T.cuda.empty_cache()

    # training
    vitmae.train()

    t_x = next(t_loader)[0].to(dtype=dtype, device=device)

    y_hat = vitmae(t_x, mask=True)
    y_hat = vitmae.conv_t(y_hat)

    optim.zero_grad(set_to_none=True)

    loss = vitmae.loss(t_x, y_hat)
    loss.backward()
    T.nn.utils.clip_grad_norm_(params, clip)
    optim.step()
    lr_sch.step()

    t_losses = T.cat(
        [t_losses, loss.detach().to(dtype=T.float32, device="cpu").reshape([1])]
    )

    # delete tensors to free memory
    del t_x, y_hat, loss
    T.cuda.empty_cache()

    # validation
    vitmae.eval()

    v_x = next(v_loader)[0].to(dtype=dtype, device=device)

    v_hat = vitmae(v_x, mask=True)
    v_hat = vitmae.conv_t(v_hat)

    v_loss = vitmae.loss(v_x, v_hat)

    v_losses = T.cat(
        [v_losses, v_loss.detach().to(dtype=T.float32, device="cpu").reshape([1])]
    )

    # delete tensors to free memory
    del v_x, v_hat, v_loss
    T.cuda.empty_cache()

    # print("Epoch loss", i, t_losses[-1], v_losses[-1])

    if i % save_interval == 0:
        vitmae.save()
        T.save(t_losses, "stats/t_losses.pt")
        T.save(v_losses, "stats/v_losses.pt")

vitmae.save()
T.save(t_losses, "stats/t_losses.pt")
T.save(v_losses, "stats/v_losses.pt")
print("Done!!!")
