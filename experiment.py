import torch as T
from vit import ViT
import time

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

T.manual_seed(0)

x = T.randn([100, 100, 100])

a = T.randn(x.size(), dtype=x.dtype, device=x.device)

print(a.mean(), a.std())
