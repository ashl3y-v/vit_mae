import sys
import torch as T
from matplotlib import pyplot as plt
from torchvision import transforms
import torchvision as tv

images = sys.argv[1:]

for im in images:
    mat = T.load(im).detach().to(dtype=T.float32, device="cpu")

    if mat.dim() > 3:
        mat = mat[0]

    d = list(mat.shape)
    c = min(d)
    d.remove(c)

    mat = mat.reshape([*d, c])

    plt.matshow(mat)

plt.show()
