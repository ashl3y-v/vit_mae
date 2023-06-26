import sys
import torch as T
from matplotlib import pyplot as plt

mat = T.load(sys.argv[1]).detach().to(dtype=T.float32, device="cpu")

mat = (
    mat[0]
    .permute(2, 1, 0)
    .reshape(
        [
            3,
        ]
    )
)

print(mat.max())

plt.matshow(mat)
plt.show()
