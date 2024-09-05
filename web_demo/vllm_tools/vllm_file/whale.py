# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
from typing import Union


from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)

from vllm.logger import init_logger



logger = init_logger(__name__)
_CONFIG_FOR_DOC = "WhaleConfig"


from transformers.configuration_utils import PretrainedConfig

class WhaleConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`WhaleModel`]. It is used to
    instantiate a vision encoder according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Args:
        input_dim (`int`, *optional*, defaults to 80):
            Dimensionality of the input features.
        num_channels (`int`, *optional*, defaults to 1):
            Number of color channels in the input images (e.g., 1 for grayscale).
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries and values in the self-attention layers.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 25):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 5000):
            The maximum number of position embeddings.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        qk_normalization (`bool`, *optional*, defaults to `True`):
            Whether to normalize the queries and keys in the self-attention layers.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        use_flash_attn (`bool`, *optional*, defaults to `True`):
            Whether to use flash attention.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        positional_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the positional encodings.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization before the attention and feed-forward operations.
        concat_after (`bool`, *optional*, defaults to `True`):
            Whether to concatenate the attention output with the input before the feed-forward layer.
        use_relative_pe (`bool`, *optional*, defaults to `True`):
            Whether to use relative position encodings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 0.1):
            A factor for layer scale.
    """

    model_type = 'whale'

    def __init__(
            self,
            input_dim=80,
            num_channels=1,
            qkv_bias=False,
            hidden_size=1024,
            num_attention_heads=25,
            max_position_embeddings=5000,
            intermediate_size=4096,
            qk_normalization=True,
            num_hidden_layers=48,
            use_flash_attn=True,
            hidden_act='relu',
            layer_norm_eps=1e-6,
            dropout=0.0,
            attention_dropout=0.0,
            positional_dropout=0.0,
            normalize_before=True,
            concat_after=True,
            use_relative_pe=True,
            initializer_range=0.02,
            initializer_factor=0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.positional_dropout = positional_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.use_flash_attn = use_flash_attn
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.max_position_embeddings = max_position_embeddings
        self.use_relative_pe = use_relative_pe

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'audio_config' in config_dict:
            config_dict = config_dict['audio_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)
    

has_flash_attn = False

class WhaleConv2dSubsampling4(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, config: WhaleConfig):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.config = config
        self.in_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.input_dim = config.input_dim

        self.conv_in = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.hidden_size, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, stride=2
            ),
            nn.ReLU(),
        )

        self.intermediate_size = self.hidden_size * (((self.input_dim - 1) // 2 - 1) // 2)
        self.out = nn.Linear(self.intermediate_size, self.hidden_size)
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv_in(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        return x, x_mask[:, 2::2][:, 2::2]
    

class WhalePositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self, config: WhaleConfig):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = config.hidden_size
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=config.dropout)
        self.max_len = config.max_position_embeddings

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self,
                x: torch.Tensor,
                offset: int = 0):
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
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int):
        """ For getting encoding in a streaming fashion
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
        return self.dropout(self.pe[:, offset:offset + size])
    

class RelPositionalEncoding(WhalePositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """
    def __init__(self, config: WhaleConfig):
        """Initialize class."""
        super().__init__(config)
        self.hidden_size = config.hidden_size
        # self.chunk_size = chunk_size
        # self.left_chunks = left_chunks
        # self.full_chunk_size = (self.left_chunks + 1) * self.chunk_size

        self.div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.hidden_size))
        self.max_length = config.max_position_embeddings
        # self.max_len = self.chunk_size * (max_len // self.chunk_size) - self.full_chunk_size

    @torch.jit.export
    def forward(self,
                x: torch.Tensor,
                offset: int = 0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)
    

class WhaleAudioEmbeddings(nn.Module):
    def __init__(self, config: WhaleConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.embed_dim = config.hidden_size
        self.dropout_rate = config.dropout
        self.input_dim = config.input_dim

        self.embedding = nn.Sequential(
            nn.Linear(config.hidden_size, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU()
        )

        self.positional_embedding = RelPositionalEncoding(config)
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:

        hidden_states = self.embedding(input_features)
        hidden_states, pos_embeds = self.positional_embedding(hidden_states)
        return hidden_states, pos_embeds



class WhaleAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: WhaleConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_drop = nn.Dropout(config.attention_dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.use_relative_pe = config.use_relative_pe
        if self.use_relative_pe:

            self.linear_pos = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
            self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
            nn.init.xavier_uniform_(self.pos_bias_u)
            nn.init.xavier_uniform_(self.pos_bias_v)


    def _naive_attn(self, x, attention_mask=None, pos_embeds=None):
        B, N, C = x.shape
        q = self.linear_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.linear_k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.linear_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        if self.use_relative_pe:

            q = q.transpose(1, 2)
            batch_size = pos_embeds.size(0)
            p = self.linear_pos(pos_embeds.to(q.dtype)).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            query_with_bias_u = (q + self.pos_bias_u.to(q.device)).transpose(1, 2)
            query_with_bias_v = (q + self.pos_bias_v.to(q.device)).transpose(1, 2)

            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            matrix_ac = torch.matmul(query_with_bias_u, k.transpose(-2, -1))
            # compute matrix b and matrix d
            matrix_bd = torch.matmul(query_with_bias_v, p.transpose(-2, -1))
            attn = (matrix_ac + matrix_bd) * self.scale

        else:
            attn = ((q * self.scale) @ k.transpose(-2, -1))

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attention_mask.bool(), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.linear_out(x)
        return x


    def forward(
            self, 
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,
            pos_embeds: torch.Tensor = None
        ) -> torch.Tensor:
        x = self._naive_attn(hidden_states, attention_mask, pos_embeds)
        return x

    
class WhaleMLP(nn.Module):
    def __init__(self, config: WhaleConfig, quant_config=None):
        super().__init__()
        self.config = config
        self.act = get_act_fn(config.hidden_act)
        self.w_1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config)
        self.w_2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.w_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, _ = self.w_2(hidden_states)
        return hidden_states


class WhaleAudioEncoderLayer(nn.Module):
    def __init__(self, config: WhaleConfig, quant_config=None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout_rate = config.dropout
        self.normalize_before = config.normalize_before
        self.concat_after = config.concat_after

        self.attn = WhaleAttention(config)
        self.feed_forward = WhaleMLP(config, quant_config=quant_config)
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

        if self.concat_after:
            self.concat_linear = nn.Linear(self.embed_dim * 2, self.embed_dim)
        else:
            self.concat_linear = nn.Identity()


    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            pos_emb: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.norm1(hidden_states)
        if self.concat_after:
            hidden_states = torch.cat(
                [hidden_states, self.attn(hidden_states, attention_mask, pos_emb)],
                dim=-1
            )
            hidden_states = self.concat_linear(hidden_states) + residual
        else:
            hidden_states = self.dropout(self.attn(hidden_states, attention_mask, pos_emb)) + residual
        if not self.normalize_before:
            hidden_states = self.norm1(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.norm2(hidden_states)
        hidden_states = self.dropout(self.feed_forward(hidden_states)) + residual
        if not self.normalize_before:
            hidden_states = self.norm2(hidden_states)

        return hidden_states


class WhaleAudioEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].
    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: WhaleConfig, quant_config=None):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            WhaleAudioEncoderLayer(config, quant_config=quant_config) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

        self.normalize_before = config.normalize_before
        if self.normalize_before:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            inputs_embeds,
            attention_mask: Optional[torch.FloatTensor] = None,
            pos_embeds: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states,
                    attention_mask,
                    pos_embeds,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    pos_embeds,
                )
            hidden_states = layer_outputs
        
        if self.normalize_before:
            hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )

        

class WhaleAudioModel(PreTrainedModel):
    main_input_name = 'input_features'
    config_class = WhaleConfig
    _no_split_modules = ['WhaleAudioEncoderLayer']

    def __init__(self, config: WhaleConfig, quant_config=None):
        super().__init__(config)
        self.config = config

        self.subsampling = WhaleConv2dSubsampling4(config)
        self.embeddings = WhaleAudioEmbeddings(config)
        self.encoder = WhaleAudioEncoder(config, quant_config=quant_config)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(input_features.shape) == 3:
                input_features, attention_mask = self.subsampling(input_features, attention_mask)
                hidden_states, pos_embeds = self.embeddings(input_features)
            else:
                raise ValueError(f'wrong pixel_values size: {input_features.shape}')
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            pos_embeds=pos_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
