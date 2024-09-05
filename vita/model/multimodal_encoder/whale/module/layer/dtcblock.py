import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DTCBlock(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size, stride, causal_conv, dilation, dropout_rate
    ):
        super(DTCBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if causal_conv:
            self.padding = 0
            self.lorder = (kernel_size - 1) * self.dilation
            self.left_padding = nn.ConstantPad1d((self.lorder, 0), 0.0)
        else:
            assert (kernel_size - 1) % 2 == 0
            self.padding = ((kernel_size - 1) // 2) * self.dilation
            self.lorder = 0
        self.causal_conv = causal_conv
        self.depthwise_conv = nn.Conv1d(
            self.input_dim,
            self.input_dim,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            groups=self.input_dim,
        )
        self.point_conv_1 = nn.Conv1d(self.input_dim, self.input_dim, 1, 1, self.padding)
        self.point_conv_2 = nn.Conv1d(self.input_dim, self.input_dim, 1, 1, self.padding)
        self.bn_1 = nn.BatchNorm1d(self.input_dim)
        self.bn_2 = nn.BatchNorm1d(self.input_dim)
        self.bn_3 = nn.BatchNorm1d(self.input_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        # buffer = 1, self.input_dim, self.lorder
        self.lorder = (kernel_size - 1) * self.dilation - (self.stride - 1)
        self.buffer_size = 1 * self.input_dim * self.lorder

    @torch.jit.unused
    def forward(self, x):
        x_in = x
        x_data = x_in.transpose(1, 2)
        if self.causal_conv:
            x_data_pad = self.left_padding(x_data)
        else:
            x_data_pad = x_data
        x_depth = self.depthwise_conv(x_data_pad)
        x_bn_1 = self.bn_1(x_depth)
        x_point_1 = self.point_conv_1(x_bn_1)
        x_bn_2 = self.bn_2(x_point_1)
        x_relu_2 = torch.relu(x_bn_2)
        x_point_2 = self.point_conv_2(x_relu_2)
        x_bn_3 = self.bn_3(x_point_2)
        x_bn_3 = x_bn_3.transpose(1, 2)
        if self.stride == 1:
            x_relu_3 = torch.relu(x_bn_3 + x_in)
        else:
            x_relu_3 = torch.relu(x_bn_3)
        x_drop = self.dropout(x_relu_3)
        return x_drop

    @torch.jit.export
    def infer(self, x, buffer, buffer_index, buffer_out):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        x_in = x
        x = x_in.transpose(1, 2)
        cnn_buffer = buffer[buffer_index : buffer_index + self.buffer_size].reshape(
            [1, self.input_dim, self.lorder]
        )
        x = torch.cat([cnn_buffer, x], dim=2)
        buffer_out.append(x[:, :, -self.lorder :].reshape(-1))
        buffer_index = buffer_index + self.buffer_size
        x = self.depthwise_conv(x)
        x = self.bn_1(x)
        x = self.point_conv_1(x)
        x = self.bn_2(x)
        x = torch.relu(x)
        x = self.point_conv_2(x)
        x = self.bn_3(x)
        x = x.transpose(1, 2)
        if self.stride == 1:
            x = torch.relu(x + x_in)
        else:
            x = torch.relu(x)
        return x, buffer, buffer_index, buffer_out
