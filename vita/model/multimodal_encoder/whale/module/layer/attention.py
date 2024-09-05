import math
import pdb

import numpy
import torch
import torch.nn as nn


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(
        self, d_model: int, dropout_rate: float, max_len: int = 1500, reverse: bool = False
    ):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: int = 0):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset : offset + x.size(1)]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int):
        """For getting encoding in a streaming fashion
        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.
        Args:
            offset (int): start offset
            size (int): requried size of position encoding
        Returns:
            torch.Tensor: Corresponding encoding
        """
        assert offset + size < self.max_len
        return self.dropout(self.pe[:, offset : offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        chunk_size: int,
        left_chunks: int,
        max_len: int = 5000,
    ):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)
        self.chunk_size = chunk_size
        self.left_chunks = left_chunks
        self.full_chunk_size = (self.left_chunks + 1) * self.chunk_size

        self.div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.max_len = self.chunk_size * (max_len // self.chunk_size) - self.full_chunk_size

    @torch.jit.export
    def forward(self, x: torch.Tensor, offset: int = 0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)

    @torch.jit.export
    def infer(self, xs, pe_index):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        pe_index = pe_index % self.max_len
        xs = xs * self.xscale

        pe = torch.zeros(self.full_chunk_size, self.d_model)
        position = torch.arange(
            pe_index, pe_index + self.full_chunk_size, dtype=torch.float32
        ).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        pos_emb = pe.unsqueeze(0)

        pe_index = pe_index + self.chunk_size
        return xs, pos_emb, pe_index


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.
    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

    @torch.jit.export
    def infer(self, xs, buffer, buffer_index, buffer_out):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        return self.w_2(torch.relu(self.w_1(xs))), buffer, buffer_index, buffer_out


class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed
    to replace positionwise feed-forward network
    in Transformer block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize MultiLayeredConv1d module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans,
            in_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    @torch.jit.unused
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Conv1dLinear(torch.nn.Module):
    """Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(Conv1dLinear, self).__init__()
        self.lorder = kernel_size - 1
        self.left_padding = nn.ConstantPad1d((self.lorder, 0), 0.0)
        self.w_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_chans, in_chans, kernel_size, stride=1, padding=0, groups=in_chans),
            torch.nn.Conv1d(in_chans, hidden_chans, 1, padding=0),
        )
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.in_chans = in_chans

        # cnn_buffer = 1, in_chans, self.lorder
        self.buffer_size = 1 * self.in_chans * self.lorder

    @torch.jit.unused
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = torch.relu(self.w_1(self.left_padding(x.transpose(-1, 1)))).transpose(-1, 1)
        return self.w_2(self.dropout(x))

    @torch.jit.export
    def infer(self, x, buffer, buffer_index, buffer_out):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        x = x.transpose(-1, 1)

        cnn_buffer = buffer[buffer_index : buffer_index + self.buffer_size].reshape(
            [1, self.in_chans, self.lorder]
        )
        x = torch.cat([cnn_buffer, x], dim=2)
        buffer_out.append(x[:, :, -self.lorder :].reshape(-1))
        buffer_index = buffer_index + self.buffer_size

        x = self.w_1(x)
        x = torch.relu(x).transpose(-1, 1)
        x = self.w_2(x)
        return x, buffer, buffer_index, buffer_out


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate, chunk_size, left_chunks, pos_enc_class):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)
        # self.min_value = float(numpy.finfo(torch.tensor(0, dtype=torch.float16).numpy().dtype).min)
        self.min_value = float(torch.finfo(torch.float16).min)
        # chunk par
        if chunk_size > 0 and left_chunks > 0:  # for streaming mode
            self.buffersize = chunk_size * (left_chunks)
            self.left_chunk_size = chunk_size * left_chunks
        else:  # for non-streaming mode
            self.buffersize = 1
            self.left_chunk_size = 1
        self.chunk_size = chunk_size

        # encoding setup
        if pos_enc_class == "rel-enc":
            self.rel_enc = True
            self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
            torch.nn.init.xavier_uniform_(self.pos_bias_u)
            torch.nn.init.xavier_uniform_(self.pos_bias_v)
        else:
            self.rel_enc = False
            self.linear_pos = nn.Identity()
            self.pos_bias_u = torch.tensor([0])
            self.pos_bias_v = torch.tensor([0])

        # buffer
        # key_buffer = 1, self.h, self.buffersize, self.d_k
        self.key_buffer_size = 1 * self.h * self.buffersize * self.d_k
        # value_buffer = 1, self.h, self.buffersize, self.d_k
        self.value_buffer_size = 1 * self.h * self.buffersize * self.d_k
        if self.chunk_size > 0:
            # buffer_mask_size = 1, self.h, self.chunk_size, self.buffersize
            self.buffer_mask_size = 1 * self.h * self.chunk_size * self.buffersize
            # self.buffer_mask = torch.ones([1, self.h, self.chunk_size, self.buffersize], dtype=torch.bool)
        else:
            self.buffer_mask = torch.ones([1, self.h, 1, 1], dtype=torch.bool)

    @torch.jit.unused
    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros(
            (x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    @torch.jit.export
    def forward(self, query, key, value, mask=None, pos_emb=torch.tensor(1.0)):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], Tensor) -> Tensor
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        if self.rel_enc:
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)
            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb.to(query.dtype)).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)
            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            # compute matrix b and matrix d
            # (batch, head, time1, time2)
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            # Remove rel_shift since it is useless in speech recognition,
            # and it requires special attention for streaming.
            # matrix_bd = self.rel_shift(matrix_bd)
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.d_k
            )  # (batch, head, time1, time2)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, self.min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)

        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    @torch.jit.export
    def infer(self, query, key, value, pos_emb, buffer, buffer_index, buffer_out):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        n_batch = query.size(0)

        q = (
            self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch, head, len_q, d_k)
        k = (
            self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch, head, len_k, d_k)
        v = (
            self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch, head, len_v, d_k)

        key_value_buffer = buffer[
            buffer_index : buffer_index + self.key_buffer_size + self.value_buffer_size
        ].reshape([1, self.h, self.buffersize * 2, self.d_k])
        key_buffer = torch.cat([key_value_buffer[:, :, : self.buffersize, :], k], dim=2)
        value_buffer = torch.cat([key_value_buffer[:, :, self.buffersize :, :], v], dim=2)
        buffer_out.append(
            torch.cat(
                [key_buffer[:, :, self.chunk_size :, :], value_buffer[:, :, self.chunk_size :, :]],
                dim=2,
            ).reshape(-1)
        )
        buffer_index = buffer_index + self.key_buffer_size + self.value_buffer_size

        if self.rel_enc:
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)
            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)
            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)
            matrix_ac = torch.matmul(q_with_bias_u, key_buffer.transpose(-2, -1))
            # compute matrix b and matrix d
            # (batch, head, time1, time2)
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            # Remove rel_shift since it is useless in speech recognition,
            # and it requires special attention for streaming.
            # matrix_bd = self.rel_shift(matrix_bd)
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        else:
            scores = torch.matmul(q, key_buffer.transpose(-2, -1)) / math.sqrt(
                self.d_k
            )  # (batch, head, len_q, buffersize)

        attn = torch.softmax(scores, dim=-1)

        x = torch.matmul(attn, value_buffer)  # (batch, head, len_q, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x), buffer, buffer_index, buffer_out  # (batch, time1, d_model)

    @torch.jit.export
    def infer_mask(self, query, key, value, mask, buffer, buffer_index, buffer_out, is_static):
        n_batch = query.size(0)

        q = (
            self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch, head, len_q, d_k)
        k = (
            self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch, head, len_k, d_k)
        v = (
            self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch, head, len_v, d_k)

        if is_static:
            key_buffer = k
            value_buffer = v
        else:
            key_value_buffer = buffer[
                buffer_index : buffer_index + self.key_buffer_size + self.value_buffer_size
            ].reshape([1, self.h, self.buffersize * 2, self.d_k])
            key_buffer = torch.cat([key_value_buffer[:, :, : self.buffersize, :], k], dim=2)
            value_buffer = torch.cat([key_value_buffer[:, :, self.buffersize :, :], v], dim=2)
            buffer_out.append(
                torch.cat(
                    [
                        key_buffer[:, :, self.chunk_size :, :],
                        value_buffer[:, :, self.chunk_size :, :],
                    ],
                    dim=2,
                ).reshape(-1)
            )
            buffer_index = buffer_index + self.key_buffer_size + self.value_buffer_size

        scores = torch.matmul(q, key_buffer.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (batch, head, len_q, buffersize)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, self.min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        x = torch.matmul(attn, value_buffer)  # (batch, head, len_q, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x), buffer_index, buffer_out  # (batch, time1, d_model)


class SoftAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SoftAttention, self).__init__()
        self.q = torch.nn.Parameter(torch.rand([hidden_dim]), requires_grad=True)
        self.wb = nn.Linear(in_dim, hidden_dim)
        self.min_value = float(numpy.finfo(torch.tensor(0, dtype=torch.float32).numpy().dtype).min)
        # buffer
        self.window_size = 50
        self.buffer_in = torch.zeros([1, self.window_size, in_dim], dtype=torch.float32)
        self.buffer = torch.zeros([1, self.window_size], dtype=torch.float32)
        self.buffer[:, :] = float(
            numpy.finfo(torch.tensor(0, dtype=torch.float32).numpy().dtype).min
        )

    @torch.jit.unused
    def forward(self, x, mask=None):
        hidden = torch.tanh(self.wb(x))  # B T D
        hidden = torch.einsum("btd,d->bt", hidden, self.q)
        score = torch.softmax(hidden, dim=-1)  # B T
        if mask is not None:
            score = score.masked_fill(mask, 0.0)
        output = torch.einsum("bt,btd->bd", score, x)
        return output

    @torch.jit.export
    def infer(self, x):
        # type: (Tensor) -> Tensor
        hidden = torch.tanh(self.wb(x))  # B T D
        hidden = torch.einsum("btd,d->bt", hidden, self.q)
        size = hidden.shape[1]
        output = torch.zeros([size, x.shape[-1]])
        for i in range(size):
            self.buffer = torch.cat([self.buffer, hidden[:, i : i + 1]], dim=-1)
            self.buffer = self.buffer[:, 1:]
            score = torch.softmax(self.buffer, dim=-1)  # B T
            self.buffer_in = torch.cat([self.buffer_in, x[:, i : i + 1, :]], dim=1)
            self.buffer_in = self.buffer_in[:, 1:]
            output[i : i + 1] = torch.einsum("bt,btd->bd", score, self.buffer_in)
        return output
