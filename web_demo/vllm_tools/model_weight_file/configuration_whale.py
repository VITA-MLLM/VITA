# --------------------------------------------------------
# Copyright (c) 
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class WhaleConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a `Whale` model. It is used to instantiate a
    Whale model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a model with the specified default parameters.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        input_dim (`int`, *optional*, defaults to 80):
            The input dimension of the model.
        num_channels (`int`, *optional*, defaults to 1):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the query, key, value projections.
        hidden_size (`int`, *optional*, defaults to 1024):
            The size of the hidden layers.
        num_attention_heads (`int`, *optional*, defaults to 25):
            The number of attention heads.
        max_position_embeddings (`int`, *optional*, defaults to 5000):
            The maximum number of position embeddings.
        intermediate_size (`int`, *optional*, defaults to 4096):
            The size of the intermediate (feed-forward) layer.
        qk_normalization (`bool`, *optional*, defaults to `True`):
            Whether to apply normalization to the query and key projections.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            The number of hidden layers in the model.
        use_flash_attn (`bool`, *optional*, defaults to `True`):
            Whether to use flash attention.
        hidden_act (`str`, *optional*, defaults to `'relu'`):
            The activation function to use in the hidden layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon value for layer normalization.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the hidden layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        positional_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the positional encodings.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether to apply normalization before the attention and feed-forward layers.
        concat_after (`bool`, *optional*, defaults to `True`):
            Whether to concatenate the attention output with the input before the feed-forward layer.
        use_relative_pe (`bool`, *optional*, defaults to `True`):
            Whether to use relative position encodings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 0.1):
            A factor for initializing the weights.

    Example:

    ```python
    >>> from transformers import WhaleConfig, WhaleModel

    >>> # Initializing a Whale configuration
    >>> configuration = WhaleConfig()

    >>> # Initializing a model from the configuration
    >>> model = WhaleModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
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

