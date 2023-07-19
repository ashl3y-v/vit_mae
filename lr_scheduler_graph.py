import torch as T
from torch import nn
import matplotlib.pyplot as plt

epochs = 4096
steps = []
lrs = []
model = [nn.Parameter()]
optim = T.optim.AdamW(model, lr=1)

lr_sch = T.optim.lr_scheduler.OneCycleLR(optim, max_lr=3e-3, total_steps=epochs)

for epoch in range(epochs):
    lrs.append(lr_sch.get_last_lr()[0])
    steps.append(epoch)
    lr_sch.step()

plt.plot(steps, lrs)
plt.show()
