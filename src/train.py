from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from vit import ViT
import multiprocessing
import os
import torch as T
import torchvision as tv

T.manual_seed(0)
# T.seed()

T.backends.cudnn.benchmark = True
T.autograd.set_detect_anomaly(True)
T.backends.cuda.matmul.allow_tf32 = True
T.backends.cudnn.allow_tf32 = True

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

lr = 3e-4
clip = 16
epochs = 8192
save_interval = epochs // 32
info_interval = epochs // 64
t_batch_size = 96
v_batch_size = 36

T.cuda.empty_cache()

vit = T.compile(ViT(dtype=dtype, device=device))
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

t_set, v_set = T.utils.data.random_split(dataset, [0.8, 0.2])

t_loader = iter(
    T.utils.data.DataLoader(
        t_set,
        batch_size=t_batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count() // 2,
    )
)

v_loader = iter(
    T.utils.data.DataLoader(
        v_set,
        batch_size=v_batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count() // 2,
    )
)

print("Dataset loaded")

params = vit.parameters()

optim = T.optim.AdamW(params, lr=lr, fused=True)

lr_sch = T.optim.lr_scheduler.OneCycleLR(
    optim, max_lr=lr, epochs=epochs, steps_per_epoch=1
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
    vit.train()

    t_x = next(t_loader)[0].to(dtype=dtype, device=device)

    t_x_hat = vit(t_x, noise=True)

    optim.zero_grad(set_to_none=True)

    loss = vit.loss(t_x, t_x_hat)
    loss.backward()
    T.nn.utils.clip_grad_norm_(params, clip)
    optim.step()
    lr_sch.step()

    t_losses = T.cat(
        [t_losses, loss.detach().to(dtype=T.float32, device="cpu").reshape([1])]
    )

    # delete tensors to free memory
    del t_x, t_x_hat, loss
    T.cuda.empty_cache()

    # validation
    vit.eval()

    v_x = next(v_loader)[0].to(dtype=dtype, device=device)

    v_x_hat = vit(v_x, noise=True)

    v_loss = vit.loss(v_x, v_x_hat)

    v_losses = T.cat(
        [v_losses, v_loss.detach().to(dtype=T.float32, device="cpu").reshape([1])]
    )

    # delete tensors to free memory
    del v_x, v_x_hat, v_loss
    T.cuda.empty_cache()

    if i % save_interval == 0:
        vit.save()
        T.save(t_losses, "stats/t_losses.pt")
        T.save(v_losses, "stats/v_losses.pt")

    if i % info_interval == 0:
        print(
            f"epoch: {i}, average loss from last epoch: {v_losses[-save_interval:-1].mean()}"
        )

vit.save()
T.save(t_losses, "stats/t_losses.pt")
T.save(v_losses, "stats/v_losses.pt")

print("Training complete")
