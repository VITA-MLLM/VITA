import torch
from typing import Tuple, Union


class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling_rate = 1
        self.right_context = 0

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.right_context = 6
        self.subsampling_rate = 4

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x, x_mask[:, :, 2::2][:, :, 2::2]


class Subsampling(torch.nn.Module):
    @staticmethod
    def add_arguments(group):
        """Add Subsampling common arguments."""
        group.add_argument("--subsampling-rate", default=4, type=int)
        group.add_argument("--subsampling-input-dim", default=256, type=int)
        group.add_argument("--subsampling-output-dim", default=256, type=int)
        group.add_argument("--subsampling-dropout-rate", default=0.1, type=float)

        return group

    def __init__(self, args):
        super().__init__()
        self.subsampling_rate = args.subsampling_rate
        self.subsampling_input_dim = args.subsampling_input_dim
        self.subsampling_output_dim = args.subsampling_output_dim
        self.subsampling_dropout_rate = args.subsampling_dropout_rate

        if self.subsampling_rate == 4:
            self.core = Conv2dSubsampling4(
                self.subsampling_input_dim,
                self.subsampling_output_dim,
                self.subsampling_dropout_rate,
            )

    def forward(self, xs, ilens, masks):
        xs, masks = self.core(xs, masks)
        ilens = masks.squeeze(1).sum(1)
        return xs, ilens, masks
