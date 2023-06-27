from positional_encodings.torch_encodings import PositionalEncoding2D
from pytorch_model_summary import summary
from torch import nn
import time
import torch as T
import torch.nn.functional as F


class ViTMAE(nn.Module):
    def __init__(
        self,
        w=256,
        h=256,
        channels=3,
        n_layer=64,
        n_head=4,
        patch_size=16,
        d_model=128,
        d_feedforward=2048,
        mask_ratio=0.5,
        dropout=0.1,
        activation=F.mish,
        file="vitmae.pt",
        dtype=T.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.w = w
        self.h = h
        self.channels = channels
        self.n_layer = n_layer
        self.n_head = n_head
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_feedforward = d_feedforward

        self.mask_ratio = mask_ratio

        self.file = file

        self.positional_encodings = PositionalEncoding2D(channels)

        self.conv = nn.Conv2d(
            channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.t_conv = nn.ConvTranspose2d(
            d_model,
            channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dropout=dropout,
            activation=activation,
            dim_feedforward=d_feedforward,
            batch_first=True,
            dtype=dtype,
            device=device,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)

        self.loss = nn.MSELoss()

        self.to(dtype=dtype, device=device)

    def forward(self, x: T.Tensor, mask=None):
        if mask != None:
            mask_ratio = mask if isinstance(mask, float) else self.mask_ratio

            if isinstance(mask, float) or mask == True:
                mask = (
                    T.rand(
                        [
                            (x.shape[2] // self.patch_size),
                            (x.shape[3] // self.patch_size),
                        ],
                        dtype=self.dtype,
                        device=self.device,
                    )
                    - mask_ratio
                ).ceil()

            mask = (
                mask.unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(4)
                .expand([-1, -1, self.patch_size, -1, self.patch_size])
                .flatten(1, 2)
                .flatten(2, 3)
            )

        # [b, c, w, h] -> [b, w, h, c]
        x = x.permute([0, 2, 3, 1])
        x = x + self.positional_encodings(x)

        # [b, w, h, c] -> [b, c, w, h]
        x = x.permute([0, 3, 1, 2])

        if mask != None:
            x = x * mask.to(x.dtype)

        x = self.conv(x)

        # [b, e, w, h] -> [b, wh, e]
        x = x.flatten(2).permute([0, 2, 1])

        # [b, l, e] -> [l, b, e]
        x = x.permute([1, 0, 2])

        # [w, h] -> [wh, wh]
        # if mask != None:
        #     mask = (
        #         mask.flatten(0).unsqueeze(1).expand([-1, mask.shape[0] * mask.shape[1]])
        #     )
        #     print(mask.shape)
        #     T.save(mask, "mask.pt")

        # dont need to mask encoder, already done in conv
        x = self.encoder(x)

        x = x.permute([1, 2, 0])

        return x

    def conv_t(self, x):
        # [b, e, l] -> [b, c, l, e]
        x = x.reshape([x.shape[0], -1, self.patch_size, self.patch_size])
        x = self.t_conv(x)

        return x

    # def reverse_batchnorm(self, x):
    #     return x * self.bn.weight.unsqueeze(0).reshape(
    #         [1, -1, 1, 1]
    #     ) + self.bn.bias.reshape(1, -1, 1, 1)

    def save(self, file=None):
        T.save(self.state_dict(), file or self.file)

    def load(self, file=None):
        self.load_state_dict(T.load(file or self.file))


if __name__ == "__main__":
    dtype = T.bfloat16
    device = "cuda" if T.cuda.is_available() else "cpu"

    vitmae = ViTMAE(dtype=dtype, device=device)
    a = T.rand([3, 3, 256, 256], dtype=dtype, device=device)

    vitmae(a, mask=True)
    # print(summary(vitmae, a, show_input=True, show_hierarchical=True))
    # b = vitmae.bn(a)
    # c = vitmae.reverse_batchnorm(b)

    # print(a.mean(), b.mean(), c.mean())

    # before = time.time()
    # r = vitmae(a, mask=True)
    # loss = vitmae.loss(
    #     a,
    #     vitmae.undo_batchnorm(vitmae.expand(r)),
    # )
    # print("b", time.time() - before, loss.item())
