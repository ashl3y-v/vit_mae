import torch as T
from torch import nn
import matplotlib.pyplot as plt

epochs = 4096
steps = []
lrs = []
model = [nn.Parameter()]
optim = T.optim.AdamW(model, lr=1)

u = epochs // 32
sch = T.optim.lr_scheduler.SequentialLR(
    optim,
    schedulers=[
        T.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, u),
        T.optim.lr_scheduler.CyclicLR(
            optim,
            base_lr=0,
            max_lr=1,
            step_size_up=u // 2,
            step_size_down=u // 2,
            gamma=1 - (10 / epochs),
            mode="exp_range",
            cycle_momentum=False,
        ),
    ],
    milestones=[epochs // 2],
)

for epoch in range(epochs):
    lrs.append(sch.get_last_lr()[0])
    steps.append(epoch)
    sch.step()

plt.plot(steps, lrs)
plt.show()
