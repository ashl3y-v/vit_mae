import torch as T
from torch import nn
import matplotlib.pyplot as plt

EPOCHS = 4096
steps = []
lrs = []
model = [nn.Parameter()]
optim = T.optim.AdamW(model, lr=1)

u = 512
u_e = 4 * u
sch = T.optim.lr_scheduler.SequentialLR(
    optim,
    schedulers=[
        T.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, u),
        T.optim.lr_scheduler.LinearLR(optim, 0.125, 0, total_iters=u_e),
    ],
    milestones=[EPOCHS - u_e],
)

for epoch in range(EPOCHS):
    lrs.append(sch.get_last_lr()[0])
    steps.append(epoch)
    sch.step()

plt.plot(steps, lrs)
plt.show()
