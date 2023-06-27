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

T.manual_seed(0)

T.backends.cudnn.benchmark = True
T.autograd.set_detect_anomaly(True)
T.backends.cuda.matmul.allow_tf32 = True
T.backends.cudnn.allow_tf32 = True

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

lr = 3e-3
clip = 16
epochs = 4096
save_interval = 64
batches = epochs
t_batch_size = 62
v_batch_size = 16

T.cuda.empty_cache()

vitmae = ViTMAE(dtype=dtype, device=device)
if os.path.isfile(vitmae.file):
    print("Loading model", vitmae.file)
    vitmae.load()

print("Loading dataset: " + str(batches * t_batch_size) + " images")

proc = transforms.Compose(
    [
        transforms.Resize([256, 256], antialias=True),
        transforms.Lambda(lambda x: x if x.mode == "RGB" else x.convert("RGB")),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(T.uint8),
    ]
)

t_set = tv.datasets.CelebA("./data/", split="train", transform=proc, download=True)
v_set = tv.datasets.CelebA("./data/", split="valid", transform=proc, download=True)

t_loader = T.utils.data.DataLoader(
    t_set,
    batch_size=t_batch_size,
    shuffle=True,
    num_workers=int(os.system("cat /proc/cpuinfo | grep processor | wc -l")),
)

v_loader = T.utils.data.DataLoader(
    v_set,
    batch_size=v_batch_size,
    shuffle=True,
    num_workers=int(os.system("cat /proc/cpuinfo | grep processor | wc -l")),
)

print("Dataset loaded")

params = vitmae.parameters()

optim = T.optim.AdamW(params, lr=lr, fused=True)

one_cycle = T.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, total_steps=epochs)
reduce_plat = T.optim.lr_scheduler.ReduceLROnPlateau(
    optim, factor=0.25, patience=32, threshold=1
)

t_losses = T.tensor([])
v_losses = T.tensor([])
if os.path.isfile("t_losses.pt"):
    t_losses = T.load("stats/t_losses.pt").to(dtype=T.float32, device="cpu")
if os.path.isfile("v_losses.pt"):
    v_losses = T.load("stats/v_losses.pt").to(dtype=T.float32, device="cpu")

for i, t_x in enumerate(t_loader):
    T.cuda.empty_cache()

    # training
    vitmae.train()
    rand_idx = random.randint(0, len(t_set) - t_batch_size)
    t_x = t_x[0].to(dtype=dtype, device=device)

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

    v_x = next(iter(v_loader))[0].to(dtype=dtype, device=device)

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
