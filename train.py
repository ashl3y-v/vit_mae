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

lr = 2e-2
clip = 16
epochs = 2048
save_interval = 64
batches = epochs
batch_size = 62
v_batch_size = 16

T.cuda.empty_cache()

vitmae = ViTMAE(dtype=dtype, device=device)
if os.path.isfile(vitmae.file):
    print("Loading model", vitmae.file)
    vitmae.load()

# use coco
print("Loading dataset: " + str(batches * batch_size) + " images")

proc_0 = transforms.Compose(
    [
        transforms.Resize([256, 256], antialias=True),
        transforms.Lambda(lambda x: x if x.mode == "RGB" else x.convert("RGB")),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(T.uint8),
    ]
)

t_set = tv.datasets.CelebA(
    "./data/", split="train", target_type=None, transform=proc_0, download=True
)
v_set = tv.datasets.CelebA(
    "./data/", split="valid", target_type=None, transform=proc_0, download=True
)

# t_set = load_dataset("Multimodal-Fatima/StanfordCars_train", split="train")["image"][
#     : batches * batch_size
# ]
# v_set = load_dataset("Multimodal-Fatima/StanfordCars_test", split="test")["image"][
#     0:batches
# ]
print("Dataset loaded")

print("Started transforms")
t_data, v_data = list(map(proc, t_data)), list(map(proc, v_data))
t_data, v_data = T.stack(t_data), T.stack(v_data)
print("Transforms completed", t_data.shape, v_data.shape)

params = vitmae.parameters()

optim = T.optim.AdamW(params, lr=lr, fused=True)

lr_sch = T.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, total_steps=epochs)

t_losses = T.tensor([])
v_losses = T.tensor([])
if os.path.isfile("t_losses.pt"):
    t_losses = T.load("t_losses.pt").to(dtype=T.float32, device="cpu")
if os.path.isfile("v_losses.pt"):
    v_losses = T.load("v_losses.pt").to(dtype=T.float32, device="cpu")

for i in range(epochs):
    T.cuda.empty_cache()

    # training
    vitmae.train()
    rand_idx = random.randint(0, len(t_data) - batch_size)
    x = t_data[rand_idx : rand_idx + batch_size].to(dtype=dtype, device=device)

    y_hat = vitmae(x, mask=True)
    y_hat = vitmae.conv_t(y_hat)

    optim.zero_grad(set_to_none=True)

    loss = vitmae.loss(x, y_hat)
    loss.backward()
    T.nn.utils.clip_grad_norm_(params, clip)
    optim.step()
    lr_sch.step()

    t_losses = T.cat(
        [t_losses, loss.detach().to(dtype=T.float32, device="cpu").reshape([1])]
    )

    # delete tensors to free memory
    del x, y_hat, loss
    T.cuda.empty_cache()

    # validation
    vitmae.eval()
    v_rand_idx = random.randint(0, len(v_data) - v_batch_size)
    v_x = t_data[v_rand_idx : v_rand_idx + v_batch_size].to(dtype=dtype, device=device)

    v_hat = vitmae(v_x, mask=True)
    v_hat = vitmae.conv_t(v_hat)

    v_loss = vitmae.loss(v_x, v_hat)

    v_losses = T.cat(
        [v_losses, v_loss.detach().to(dtype=T.float32, device="cpu").reshape([1])]
    )

    # delete tensors to free memory
    del v_x, v_hat, v_loss
    T.cuda.empty_cache()

    print("Epoch loss", i, t_losses[-1], v_losses[-1])

    if i % save_interval == 0:
        vitmae.save()
        T.save(t_losses, "t_losses.pt")
        T.save(v_losses, "v_losses.pt")

print("Done!!!")
vitmae.save()
T.save(t_losses, "t_losses.pt")
T.save(v_losses, "v_losses.pt")
