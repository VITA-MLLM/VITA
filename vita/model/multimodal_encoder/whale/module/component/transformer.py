"""Encoder self-attention layer definition."""

import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vita.model.multimodal_encoder.whale.module.layer.attention import (
    Conv1dLinear,
    MultiHeadedAttention,
    MultiLayeredConv1d,
    PositionalEncoding,
    PositionwiseFeedForward,
    RelPositionalEncoding,
)

# from vita.model.multimodal_encoder.whale.module.component.utils import *
from vita.model.multimodal_encoder.whale.utils import IGNORE_ID, add_optional_chunk_mask, strtobool


def repeat(N, fn):
    """Repeat module N times.

    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn(n) for n in range(N)])


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, x, masks, pos_emb):

        """Repeat."""
        for m in self:
            x, masks, pos_emb = m(x, masks, pos_emb)
        return x, masks, pos_emb

    @torch.jit.export
    def infer(self, x, pos_emb, buffer, buffer_index, buffer_out):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        """Repeat."""
        for m in self:
            x, pos_emb, buffer, buffer_index, buffer_out = m.infer(
                x, pos_emb, buffer, buffer_index, buffer_out
            )
        return x, pos_emb, buffer, buffer_index, buffer_out

    @torch.jit.export
    def infer_hidden(self, x, pos_emb, buffer, buffer_index, buffer_out, hidden_out):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        """Repeat."""
        for m in self:
            x, pos_emb, buffer, buffer_index, buffer_out = m.infer(
                x, pos_emb, buffer, buffer_index, buffer_out
            )
            hidden_out.append(x)
        return x, pos_emb, buffer, buffer_index, buffer_out, hidden_out


class TransformerLayer(nn.Module):
    """Transformer layer module.

    :param int size: input dim
    :param self_attn: self attention module
    :param feed_forward: feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self, size, self_attn, feed_forward, dropout_rate, normalize_before=True, concat_after=False
    ):
        """Construct an TransformerLayer object."""
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = torch.nn.LayerNorm(size)
        self.norm2 = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    @torch.jit.unused
    def forward(self, x, mask, pos_emb):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask, pos_emb)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x, x, x, mask, pos_emb))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, pos_emb

    @torch.jit.export
    def infer(self, x, pos_emb, buffer, buffer_index, buffer_out):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        residual = x.clone()
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_att, buffer, buffer_index, buffer_out = self.self_attn.infer(
                x, x, x, pos_emb, buffer, buffer_index, buffer_out
            )
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x_att, buffer, buffer_index, buffer_out = self.self_attn.infer(
                x, x, x, pos_emb, buffer, buffer_index, buffer_out
            )
            x = residual + x_att
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x.clone()
        if self.normalize_before:
            x = self.norm2(x)
        x_feed, buffer, buffer_index, buffer_out = self.feed_forward.infer(
            x, buffer, buffer_index, buffer_out
        )
        x = residual + x_feed
        if not self.normalize_before:
            x = self.norm2(x)

        return x, pos_emb, buffer, buffer_index, buffer_out


class Transformer(torch.nn.Module):
    @staticmethod
    def add_arguments(group):
        """Add TDNN common arguments."""
        group.add_argument(
            "--transformer-input-dim", default=256, type=int, help="Input dim of Transformer."
        )
        group.add_argument(
            "--transformer-output-dim", default=4, type=int, help="Output dim of Transformer."
        )
        group.add_argument(
            "--transformer-attention-dim", default=256, type=int, help="Dimention of attention."
        )
        group.add_argument(
            "--transformer-attention-heads",
            default=4,
            type=int,
            help="The number of heads of multi head attention.",
        )
        group.add_argument(
            "--transformer-linear-units",
            default=1024,
            type=int,
            help="The number of units of position-wise feed forward.",
        )
        group.add_argument(
            "--transformer-num-blocks", default=6, type=int, help="The number of attention blocks."
        )
        group.add_argument(
            "--transformer-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate in Transformer.",
        )
        group.add_argument(
            "--transformer-attention-dropout-rate",
            default=0.0,
            type=float,
            help="Dropout rate in attention.",
        )
        group.add_argument(
            "--transformer-positional-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate after adding positional encoding.",
        )
        group.add_argument(
            "--transformer-input-layer", default="linear", type=str, help="Type of input layer"
        )
        group.add_argument("--transformer-pos-enc-class", default="abs-enc", type=str, help="")
        group.add_argument(
            "--transformer-normalize-before",
            default=True,
            type=strtobool,
            help="Whether to use layer-norm before the first block.",
        )
        group.add_argument(
            "--transformer-concat-after",
            default=False,
            type=strtobool,
            help="Whether to concat attention layer's input and output.",
        )
        group.add_argument(
            "--transformer-positionwise-layer-type",
            default="linear",
            type=str,
            help="Linear of conv1d.",
        )
        group.add_argument(
            "--transformer-positionwise-conv-kernel_size",
            default=1,
            type=int,
            help="Kernel size of positionwise conv1d layer.",
        )
        group.add_argument("--transformer-chunk_size", default=-1, type=int, help="")
        group.add_argument("--transformer-left_chunks", default=-1, type=int, help="")
        group.add_argument("--transformer-dynamic-chunks", default=True, type=strtobool, help="")
        return group

    def __init__(
        self,
        args,
        input_dim=None,
        output_dim=None,
        attention_dim=None,
        attention_heads=None,
        linear_units=None,
        num_blocks=None,
        dropout_rate=None,
        positional_dropout_rate=None,
        attention_dropout_rate=None,
        input_layer=None,
        pos_enc_class=None,
        normalize_before=None,
        concat_after=None,
        positionwise_layer_type=None,
        positionwise_conv_kernel_size=None,
        chunk_size=None,
        left_chunks=None,
    ):
        """Construct an Encoder object."""
        super(Transformer, self).__init__()
        if args is None:
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.attention_dim = attention_dim
            self.attention_heads = attention_heads
            self.linear_units = linear_units
            self.num_blocks = num_blocks
            self.dropout_rate = dropout_rate
            self.positional_dropout_rate = positional_dropout_rate
            self.attention_dropout_rate = attention_dropout_rate
            self.input_layer = input_layer
            self.pos_enc_class = pos_enc_class
            self.normalize_before = normalize_before
            self.concat_after = concat_after
            self.positionwise_layer_type = positionwise_layer_type
            self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
            self.chunk_size = chunk_size
            self.left_chunks = left_chunks
        else:
            self.input_dim = args.transformer_input_dim
            self.output_dim = args.transformer_output_dim
            self.attention_dim = args.transformer_attention_dim
            self.attention_heads = args.transformer_attention_heads
            self.linear_units = args.transformer_linear_units
            self.num_blocks = args.transformer_num_blocks
            self.dropout_rate = args.transformer_dropout_rate
            self.positional_dropout_rate = args.transformer_positional_dropout_rate
            self.attention_dropout_rate = args.transformer_attention_dropout_rate
            self.input_layer = args.transformer_input_layer
            self.pos_enc_class = args.transformer_pos_enc_class
            self.normalize_before = args.transformer_normalize_before
            self.concat_after = args.transformer_concat_after
            self.positionwise_layer_type = args.transformer_positionwise_layer_type
            self.positionwise_conv_kernel_size = args.transformer_positionwise_conv_kernel_size
            self.chunk_size = args.transformer_chunk_size
            self.left_chunks = args.transformer_left_chunks
            self.transformer_dynamic_chunks = args.transformer_dynamic_chunks

        if self.pos_enc_class == "abs-enc":
            pos_enc_args = (self.attention_dim, self.positional_dropout_rate)
            pos_enc_class = PositionalEncoding
        elif self.pos_enc_class == "rel-enc":
            pos_enc_args = (
                self.attention_dim,
                self.positional_dropout_rate,
                self.chunk_size,
                self.left_chunks,
            )
            pos_enc_class = RelPositionalEncoding

        if self.input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.attention_dim),
                torch.nn.LayerNorm(self.attention_dim),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.ReLU(),
            )
        elif self.input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(self.input_dim, self.attention_dim, padding_idx=IGNORE_ID)
            )
        elif self.input_layer == "none":
            self.embed = torch.nn.Sequential(torch.nn.Identity())
        else:
            raise ValueError("unknown input_layer: " + self.input_layer)
        self.pe = pos_enc_class(*pos_enc_args)
        self.embed_layer_num = len(self.embed)

        if self.positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (self.attention_dim, self.linear_units, self.dropout_rate)
        elif self.positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                self.attention_dim,
                self.linear_units,
                self.positionwise_conv_kernel_size,
                self.dropout_rate,
            )
        elif self.positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                self.attention_dim,
                self.linear_units,
                self.positionwise_conv_kernel_size,
                self.dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        self.encoders = repeat(
            self.num_blocks,
            lambda lnum: TransformerLayer(
                self.attention_dim,
                MultiHeadedAttention(
                    self.attention_heads,
                    self.attention_dim,
                    self.attention_dropout_rate,
                    self.chunk_size,
                    self.left_chunks,
                    self.pos_enc_class,
                ),
                positionwise_layer(*positionwise_layer_args),
                self.dropout_rate,
                self.normalize_before,
                self.concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(self.attention_dim)

    @torch.jit.unused
    def forward(self, xs, ilens=None, masks=None):
        """Embed positions in tensor.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """

        if self.transformer_dynamic_chunks == True:  # and self.training:
            chunk_masks = add_optional_chunk_mask(xs, masks, True, True, 0, 0, -1)
        else:
            chunk_masks = add_optional_chunk_mask(
                xs, masks, False, False, self.chunk_size, self.chunk_size, self.left_chunks
            ).to(xs.device)
        xs = self.embed(xs)
        xs, pos_emb = self.pe(xs)
        xs, chunk_masks, pos_emb = self.encoders(xs, chunk_masks, pos_emb)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, ilens, masks

    @torch.jit.export
    def infer(self, xs, buffer, buffer_index, buffer_out):
        xs = self.embed(xs)

        # pe_index = buffer[buffer_index: buffer_index + 1].reshape([1]).to(torch.int64)
        # xs, pos_emb, pe_index[0] = self.pe.infer(xs, pe_index[0])
        # buffer_out.append(pe_index.reshape(-1).to(torch.float32))
        # buffer_index = buffer_index + 1
        xs, pos_emb, _ = self.pe.infer(xs, 0)
        xs, pos_emb, buffer, buffer_index, buffer_out = self.encoders.infer(
            xs, pos_emb, buffer, buffer_index, buffer_out
        )

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, buffer, buffer_index, buffer_out

    @torch.jit.export
    def infer_hidden(self, xs, buffer, buffer_index, buffer_out, hidden_out):
        xs = self.embed(xs)

        # pe_index = buffer[buffer_index: buffer_index + 1].reshape([1]).to(torch.int64)
        # xs, pos_emb, pe_index[0] = self.pe.infer(xs, pe_index[0])
        # buffer_out.append(pe_index.reshape(-1).to(torch.float32))
        # buffer_index = buffer_index + 1
        xs, pos_emb, _ = self.pe.infer(xs, 0)
        xs, pos_emb, buffer, buffer_index, buffer_out, hidden_out = self.encoders.infer_hidden(
            xs, pos_emb, buffer, buffer_index, buffer_out, hidden_out
        )

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, buffer, buffer_index, buffer_out, hidden_out
