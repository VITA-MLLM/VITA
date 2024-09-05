"""Encoder self-attention layer definition."""

import math
import pdb
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vita.model.multimodal_encoder.whale.utils import IGNORE_ID, strtobool

try:
    from mamba_ssm.modules.mamba_simple import Mamba, Block
    from mamba_ssm.models.mixer_seq_simple import _init_weights
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    print("Please install mamba_ssm to use MambaSSM component.")


class MambaBlock(nn.Module):
    def __init__(self, in_channels, n_layer=1, d_state=16, d_conv=4, expand=4, bidirectional=False):
        super(MambaBlock, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        self.forward_norm_f = RMSNorm(in_channels, eps=1e-5)
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(
                        Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand
                    ),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=True,
                    residual_in_fp32=True,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        in_channels,
                        mixer_cls=partial(
                            Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand
                        ),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=True,
                        residual_in_fp32=True,
                    )
                )
            self.backward_norm_f = RMSNorm(in_channels, eps=1e-5)
        else:
            self.backward_blocks = None

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f
        residual = self.forward_norm_f(residual)

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (
                (backward_f + back_residual) if back_residual is not None else backward_f
            )

            back_residual = torch.flip(back_residual, [1])
            back_residual = self.backward_norm_f(back_residual)
            residual = torch.cat([residual, back_residual], -1)

        return residual


class MambaSSM(torch.nn.Module):
    @staticmethod
    def add_arguments(group):
        """Add TDNN common arguments."""
        group.add_argument(
            "--mamba-num-layers", default=4, type=int, help="Output dim of MambaSSM."
        )
        group.add_argument(
            "--mamba-input-dim", default=256, type=int, help="Input dim of MambaSSM."
        )
        group.add_argument(
            "--mamba-output-dim", default=256, type=int, help="Output dim of MambaSSM."
        )
        group.add_argument("--mamba-d-state", default=16, type=int, help="d-state of MambaSSM.")
        group.add_argument("--mamba-d-conv", default=4, type=int, help="d-conv of MambaSSM.")
        group.add_argument("--mamba-expand", default=4, type=int, help="expand of MambaSSM.")
        return group

    def __init__(self, args):
        """Construct an Encoder object."""
        super(MambaSSM, self).__init__()
        self.mamb_num_layers = args.mamba_num_layers
        self.mamba_input_dim = args.mamba_input_dim
        self.mamba_output_dim = args.mamba_output_dim
        self.mamba_d_state = args.mamba_d_state
        self.mamba_d_conv = args.mamba_d_conv
        self.mamba_expand = args.mamba_expand

        self.mamba = MambaBlock(
            self.mamba_input_dim,
            self.mamb_num_layers,
            self.mamba_d_state,
            self.mamba_d_conv,
            self.mamba_expand,
        )

    @torch.jit.unused
    def forward(self, xs, ilens=None, masks=None):
        """Embed positions in tensor.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """

        xs_out = self.mamba(xs)

        return xs_out.to(xs.dtype), ilens, masks
