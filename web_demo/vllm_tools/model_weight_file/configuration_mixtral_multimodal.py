# coding=utf-8
# Copyright 2024 The Vita team. All rights reserved.


"""Multi-modal Mixtral model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)
from transformers.models.auto import CONFIG_MAPPING

from .configuration_intern_vit import InternVisionConfig
from .configuration_whale import WhaleConfig

class MixtralMultiModalConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a `MixtralMultiModal` model. It is used to instantiate a
    MixtralMultiModal model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a model with the specified default parameters.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`, *optional*, defaults to `None`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `None`):
            The config object or dictionary of the text backbone.
        audio_config (`Union[AutoConfig, dict]`, *optional*, defaults to `None`):
            The config object or dictionary of the audio backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vision_downsample_ratio (`float`, *optional*, defaults to 0.5):
            The downsample ratio for the vision features.
        dynamic_image_size (`bool`, *optional*, defaults to `True`):
            Whether to use dynamic image sizes.
        max_dynamic_patch (`int`, *optional*, defaults to 12):
            The maximum number of dynamic patches.
        min_dynamic_patch (`int`, *optional*, defaults to 1):
            The minimum number of dynamic patches.
        use_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to use thumbnails.
        audio_token_index (`int`, *optional*, defaults to 32000):
            The audio token index to encode the audio prompt.
        audio_projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the audio projector.
        audio_projector_kernel_size (`int`, *optional*, defaults to 5):
            The kernel size used by the audio projector.
        audio_downsample_ratio (`float`, *optional*, defaults to 0.125):
            The downsample ratio for the audio features.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    Example:

    ```python
    >>> from transformers import MixtralMultiModalConfig, MixtralMultiModalModel

    >>> # Initializing a MixtralMultiModal configuration
    >>> configuration = MixtralMultiModalConfig()

    >>> # Initializing a model from the configuration
    >>> model = MixtralMultiModalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "mixtral_multimodal"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        audio_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vision_downsample_ratio=0.5,
        dynamic_image_size=True,
        max_dynamic_patch=12,
        min_dynamic_patch=1,
        use_thumbnail=True,
        audio_token_index=32000,
        audio_projector_hidden_act="gelu",
        audio_projector_kernel_size=5,
        audio_downsample_ratio=0.125,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.vision_downsample_ratio = vision_downsample_ratio
        self.dynamic_image_size = dynamic_image_size
        self.max_dynamic_patch = max_dynamic_patch
        self.min_dynamic_patch = min_dynamic_patch
        self.use_thumbnail = use_thumbnail


        self.audio_token_index = audio_token_index
        self.audio_projector_hidden_act = audio_projector_hidden_act
        self.audio_projector_kernel_size = audio_projector_kernel_size
        self.audio_downsample_ratio = audio_downsample_ratio

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = InternVisionConfig(**vision_config)
            
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        if isinstance(audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"] if "model_type" in audio_config else "clip_vision_model"
            )
            audio_config = WhaleConfig(**audio_config)
        
        self.audio_config = audio_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

