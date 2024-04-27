import torch as T
import numpy
from matplotlib import pyplot as plt

t: T.Tensor = T.load("stats/t_losses.pt")
v: T.Tensor = T.load("stats/v_losses.pt")

x = T.arange(0, len(t))

plt.scatter(x, t, marker="+")
plt.scatter(x, v, marker="+")
plt.show()
