import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        causal_conv,
        dilation,
        dropout_rate,
        residual=True,
    ):
        super(Conv1dLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.causal_conv = causal_conv
        if causal_conv:
            self.lorder = (kernel_size - 1) * self.dilation
            self.left_padding = nn.ConstantPad1d((self.lorder, 0), 0.0)
        else:
            assert (kernel_size - 1) % 2 == 0
            self.lorder = ((kernel_size - 1) // 2) * self.dilation
            self.left_padding = nn.ConstantPad1d((self.lorder, self.lorder), 0.0)
        self.conv1d = nn.Conv1d(
            self.input_dim, self.output_dim, self.kernel_size, self.stride, 0, self.dilation
        )
        self.bn = nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.99)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.residual = residual
        if self.input_dim != self.output_dim:
            self.residual = False

        # buffer = 1, self.input_dim, self.lorder
        self.lorder = (kernel_size - 1) * self.dilation - (self.stride - 1)
        self.buffer_size = 1 * self.input_dim * self.lorder
        self.x_data_chache_size = self.lorder
        self.x_data_buffer_size = self.input_dim * self.x_data_chache_size

    @torch.jit.unused
    def forward(self, x):
        x_data = x
        x = self.left_padding(x)
        x = self.conv1d(x)
        x = self.bn(x)
        if self.stride == 1 and self.residual:
            x = self.relu(x + x_data)
        else:
            x = self.relu(x)
        x = self.dropout(x)
        return x

    @torch.jit.export
    def infer(self, x, buffer, buffer_index, buffer_out):
        # type: (Tensor) -> Tensor
        x_data = x.clone()

        cnn_buffer = buffer[buffer_index : buffer_index + self.buffer_size].reshape(
            [1, self.input_dim, self.lorder]
        )
        x = torch.cat([cnn_buffer, x], dim=2)
        buffer_out.append(x[:, :, -self.lorder :].reshape(-1))
        buffer_index = buffer_index + self.buffer_size

        x = self.conv1d(x)
        x = self.bn(x)

        if self.stride == 1 and self.residual:
            x_data_cnn_buffer = buffer[
                buffer_index : buffer_index + self.x_data_buffer_size
            ].reshape([1, self.input_dim, self.x_data_chache_size])
            x_data = torch.cat([x_data_cnn_buffer, x_data], dim=2)
            buffer_out.append(x_data[:, :, -self.x_data_chache_size :].reshape(-1))
            buffer_index = buffer_index + self.x_data_buffer_size
            x_data = x_data[:, :, : -self.x_data_chache_size]
            x = self.relu(x + x_data)
        else:
            x = self.relu(x)

        return x, buffer, buffer_index, buffer_out
