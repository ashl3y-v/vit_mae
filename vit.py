from pytorch_model_summary import summary
from torch import nn
import math
import torch as T
import torch.nn.functional as F
from torchvision import transforms


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = T.arange(max_len).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = T.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = T.sin(position * div_term)
        pe[:, 0, 1::2] = T.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        print(self.pe.shape)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ViT(nn.Module):
    def __init__(
        self,
        w=256,
        h=256,
        c=3,
        n_layer=16,
        n_head=8,
        patch_size=16,
        d_feedforward=2048,
        noise=0.25,
        dropout=0.1,
        max_len=256,
        file="vit.pt",
        dtype=T.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device

        d_model = patch_size**2 * c

        self.w = w
        self.h = h
        self.c = c
        self.n_layer = n_layer
        self.n_head = n_head
        self.patch_size = patch_size
        self.w_p = w // patch_size
        self.h_p = h // patch_size
        self.d_model = d_model
        self.d_feedforward = d_feedforward

        self.file = file

        self.pos = PositionalEncoding(d_model, max_len=max_len)

        self.noise = transforms.Lambda(
            lambda x: x + T.randn(x.size(), dtype=x.dtype, device=x.device) * noise
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dropout=dropout,
            dim_feedforward=d_feedforward,
            batch_first=True,
            dtype=dtype,
            device=device,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)

        self.loss = nn.MSELoss()

        self.to(dtype=dtype, device=device)

    def forward(self, x: T.Tensor, noise=False):
        # [b, c, w, h] -> [b, c, w, h]
        if noise:
            x = self.noise(x)

        # [b, c, w, h] -> [b, l, e]
        x = (
            x.unsqueeze(1)
            .unsqueeze(3)
            .reshape(
                [
                    x.size(0),
                    self.c,
                    self.w_p,
                    self.patch_size,
                    self.h_p,
                    self.patch_size,
                ]
            )
            .permute([0, 2, 4, 3, 5, 1])
            .flatten(3)
            .flatten(1, 2)
        )

        # [b, l, e] -> [l, b, e]
        x = x.permute([1, 0, 2])

        # [l, b, e] -> [l, b, e]
        x = self.pos(x)

        # [l, b, e] -> [b, l, e]
        x = x.permute([1, 0, 2])

        # [b, l, e] -> [b, l, e]
        x = self.encoder(x)

        # [b, l, e] -> [b, c, w, h]
        x = (
            x.reshape(
                [
                    x.size(0),
                    self.w_p,
                    self.h_p,
                    self.patch_size,
                    self.patch_size,
                    self.c,
                ]
            )
            .permute([0, 5, 1, 3, 2, 4])
            .flatten(4, 5)
            .flatten(2, 3)
        )

        return x

    def save(self, file=None):
        T.save(self.state_dict(), file or self.file)

    def load(self, file=None):
        self.load_state_dict(T.load(file or self.file))


if __name__ == "__main__":
    dtype = T.bfloat16
    device = "cuda" if T.cuda.is_available() else "cpu"

    vit = ViT(dtype=dtype, device=device)
    a = T.rand([3, 3, 256, 256], dtype=dtype, device=device)

    # vit(a, mask=True)
    print(summary(vit, a, show_input=True, show_hierarchical=True))
    # b = vit.bn(a)
    # c = vit.reverse_batchnorm(b)

    # print(a.mean(), b.mean(), c.mean())

    # before = time.time()
    # r = vit(a, mask=True)
    # loss = vit.loss(
    #     a,
    #     vit.undo_batchnorm(vit.expand(r)),
    # )
    # print("b", time.time() - before, loss.item())
