import torch as T
from vit import ViT
import time

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

T.manual_seed(0)

a = T.randn([2, 2, 2])

print(a.dtype, a.device)
a.to(dtype=T.bfloat16, device="cuda")

print(a.dtype, a.device)
