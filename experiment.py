import torch as T
from vitmae import ViTMAE
import time
from transformers import ViTMAEConfig, ViTMAEModel
from pytorch_model_summary import summary

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

T.manual_seed(0)

# Initializing a ViT MAE vit-mae-base style configuration

configuration = ViTMAEConfig()

# Initializing a model (with random weights) from the vit-mae-base style configuration

vitmae = ViTMAEModel(configuration).to(dtype=dtype, device=device)

a = T.randn([1, 3, 224, 224], dtype=dtype, device=device)

print(summary(vitmae, a, show_input=True, show_hierarchical=True))
