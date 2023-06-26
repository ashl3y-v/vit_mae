import torch as T
from torch import nn
import matplotlib.pyplot as plt

EPOCHS = 1000
BATCHES = 10
steps = []
lrs = []
model = [nn.Parameter()]
optim = T.optim.AdamW(model, lr=1)

sch = T.optim.lr_scheduler.OneCycleLR(optim, max_lr=1, total_steps=EPOCHS * BATCHES)

for epoch in range(EPOCHS):
    for batch in range(BATCHES):
        lrs.append(sch.get_last_lr()[0])
        steps.append(epoch * BATCHES + batch)
        sch.step()

plt.figure()
plt.legend()
plt.plot(steps, lrs, label="OneCycle")
plt.show()
