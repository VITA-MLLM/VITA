import torch
import torch.nn as nn
import torch.nn.functional as F


class FsmnLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        hidden_dim,
        left_frame=1,
        right_frame=1,
        left_dilation=1,
        right_dilation=1,
    ):
        super(FsmnLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.left_dilation = left_dilation
        self.right_dilation = right_dilation
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        if left_frame > 0:
            self.pad_left = nn.ConstantPad1d([left_dilation * left_frame, 0], 0.0)
            self.conv_left = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=left_frame + 1,
                dilation=left_dilation,
                bias=False,
                groups=hidden_dim,
            )
        if right_frame > 0:
            self.pad_right = nn.ConstantPad1d([-right_dilation, right_dilation * right_frame], 0.0)
            self.conv_right = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=right_frame,
                dilation=right_dilation,
                bias=False,
                groups=hidden_dim,
            )
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)

        # cache = 1, self.hidden_dim, left_frame * left_dilation + right_frame * right_dilation
        self.cache_size = left_frame * left_dilation + right_frame * right_dilation
        self.buffer_size = self.hidden_dim * self.cache_size
        self.p_in_raw_chache_size = self.right_frame * self.right_dilation
        self.p_in_raw_buffer_size = self.hidden_dim * self.p_in_raw_chache_size
        self.hidden_chache_size = self.right_frame * self.right_dilation
        self.hidden_buffer_size = self.hidden_dim * self.hidden_chache_size

    @torch.jit.unused
    def forward(self, x, hidden=None):
        x_data = x.transpose(1, 2)
        p_in = self.conv_in(x_data)
        if self.left_frame > 0:
            p_left = self.pad_left(p_in)
            p_left = self.conv_left(p_left)
        else:
            p_left = 0
        if self.right_frame > 0:
            p_right = self.pad_right(p_in)
            p_right = self.conv_right(p_right)
        else:
            p_right = 0
        p_out = p_in + p_right + p_left
        if hidden is not None:
            p_out = hidden + p_out
        out = F.relu(self.conv_out(p_out))
        out = out.transpose(1, 2)
        return out, p_out

    @torch.jit.export
    def infer(self, x, buffer, buffer_index, buffer_out, hidden=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        p_in_raw = self.conv_in(x)

        cnn_buffer = buffer[buffer_index : buffer_index + self.buffer_size].reshape(
            [1, self.hidden_dim, self.cache_size]
        )
        p_in = torch.cat([cnn_buffer, p_in_raw], dim=2)
        # buffer[buffer_index: buffer_index + self.buffer_size] =  p_in[:, :, -self.cache_size:].reshape(-1)
        buffer_out.append(p_in[:, :, -self.cache_size :].reshape(-1))
        buffer_index = buffer_index + self.buffer_size

        if self.left_frame > 0:
            if self.right_frame > 0:
                p_left = p_in[:, :, : -self.right_frame * self.right_dilation]
            else:
                p_left = p_in[:, :]
            p_left_out = self.conv_left(p_left)
        else:
            p_left_out = torch.tensor([0])
        if self.right_frame > 0:
            p_right = p_in[:, :, self.left_frame * self.left_dilation + 1 :]
            p_right_out = self.conv_right(p_right)
        else:
            p_right_out = torch.tensor([0])

        if self.right_frame > 0:
            p_in_raw_cnn_buffer = buffer[
                buffer_index : buffer_index + self.p_in_raw_buffer_size
            ].reshape([1, self.hidden_dim, self.p_in_raw_chache_size])
            p_in_raw = torch.cat([p_in_raw_cnn_buffer, p_in_raw], dim=2)
            # buffer[buffer_index: buffer_index + self.p_in_raw_buffer_size] =  p_in_raw[:, :, -self.p_in_raw_chache_size:].reshape(-1)
            buffer_out.append(p_in_raw[:, :, -self.p_in_raw_chache_size :].reshape(-1))
            buffer_index = buffer_index + self.p_in_raw_buffer_size
            p_in_raw = p_in_raw[:, :, : -self.p_in_raw_chache_size]
        p_out = p_in_raw + p_left_out + p_right_out

        if hidden is not None:
            if self.right_frame > 0:
                hidden_cnn_buffer = buffer[
                    buffer_index : buffer_index + self.hidden_buffer_size
                ].reshape([1, self.hidden_dim, self.hidden_chache_size])
                hidden = torch.cat([hidden_cnn_buffer, hidden], dim=2)
                # buffer[buffer_index: buffer_index + self.hidden_buffer_size] =  hidden[:, :, -self.hidden_chache_size:].reshape(-1)
                buffer_out.append(hidden[:, :, -self.hidden_chache_size :].reshape(-1))
                buffer_index = buffer_index + self.hidden_buffer_size
                hidden = hidden[:, :, : -self.hidden_chache_size]
            p_out = hidden + p_out

        out = F.relu(self.conv_out(p_out))

        return out, buffer, buffer_index, buffer_out, p_out
