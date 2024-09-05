# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Mixtral model."""
from typing import Iterable, List, Literal, Optional, Tuple, TypedDict, Union, Mapping, TypeVar
from typing_extensions import NotRequired

import torch
from torch import nn
from transformers import MixtralConfig
from transformers.activations import ACT2FN
from transformers import PreTrainedTokenizerBase
from PIL import Image

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, MultiModalConfig, ModelConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensors
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.multimodal.image import cached_get_image_processor
from vllm.logger import init_logger

from .clip import (dummy_image_for_clip, dummy_seq_data_for_clip,)

from .internvl import get_max_internvl_image_tokens, dynamic_preprocess, get_internvl_num_patches, calculate_num_blocks

from .interfaces import SupportsLoRA, SupportsMultiModal
from .utils import is_pp_missing_parameter, make_layers

from .intern_vit import InternVisionModel
from .whale import WhaleAudioModel

logger = init_logger(__name__)


class MixtralImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: BatchedTensors
    """
    Shape: `(batch_size, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different for each batch, in which case
    the data is passed as a list instead of a batched tensor.
    """

class MixtralAudioFeaturesInputs(TypedDict):
    type: Literal["audio_input"]
    data: BatchedTensors
    """
    Shape: `(batch_size, num_channels, time)`

    """
    mask: NotRequired[torch.Tensor]

MixtralImageInputs = MixtralImagePixelInputs
MixtralAudioInputs = MixtralAudioFeaturesInputs


# Utilities for input processors
_T = TypeVar("_T", str, int)


def repeat_and_pad_image_tokens(
    tokenizer: PreTrainedTokenizerBase,
    prompt: Optional[str],
    prompt_token_ids: List[int],
    *,
    image_token_id: Union[int, List[int]],
    repeat_count: Union[int, List[int]] = 1,
    pad_token_left: Optional[int] = None,
    pad_token_right: Optional[int] = None,
) -> Tuple[Optional[str], List[int]]:
    
    def repeat_and_pad_token(
        token: _T,
        *,
        repeat_count: int = 1,
        pad_token_left: Optional[_T] = None,
        pad_token_right: Optional[_T] = None,
    ) -> List[_T]:
        replacement = [token] * repeat_count
        if pad_token_left is not None:
            replacement = [pad_token_left] + replacement
        if pad_token_right is not None:
            replacement = replacement + [pad_token_right]

        return replacement
    

    if prompt is not None:
        image_token_str = tokenizer.decode(image_token_id)
        image_token_count = prompt.count(image_token_str)
    elif prompt_token_ids is not None:
        image_token_count = prompt_token_ids.count(image_token_id)
    else:
        raise ValueError("Either prompt or prompt_token_ids must be provided.")
    
    if isinstance(repeat_count, int):
        repeat_count = [repeat_count] * image_token_count
    assert len(repeat_count) == image_token_count, (
        f"Length of repeat_count ({len(repeat_count)}) does not match "
        f"the number of image tokens ({image_token_count})."
    )

    if prompt is None:
        new_prompt = None
    else:
        pad_token_str_left = (None if pad_token_left is None else
                                tokenizer.decode(pad_token_left))
        pad_token_str_right = (None if pad_token_right is None else
                                tokenizer.decode(pad_token_right))

        replacement_strs = []
        for i, rp_count in enumerate(repeat_count):

            replacement_str = "".join(
                repeat_and_pad_token(
                    image_token_str,
                    repeat_count=rp_count,
                    pad_token_left=pad_token_str_left,
                    pad_token_right=pad_token_str_right,
                )
            )
            replacement_strs.append(replacement_str)

        prompt_split = prompt.split(image_token_str)
        assert len(prompt_split) == len(replacement_strs) + 1, (
            f"Length of new_prompt ({len(prompt_split)}) does not match "
            f"the number of replacement strings ({len(replacement_strs)})."
        )

        new_prompt = []
        for a, b in zip(prompt_split, replacement_strs + [None]):
            new_prompt.append(a)
            if b is not None:
                new_prompt.append(b)
        new_prompt = "".join(new_prompt)

    new_token_ids: List[int] = []
    for i, token in enumerate(prompt_token_ids):
        if token == image_token_id:
            rp_count = repeat_count.pop(0)
            replacement_ids = repeat_and_pad_token(
                image_token_id,
                repeat_count=rp_count,
                pad_token_left=pad_token_left,
                pad_token_right=pad_token_right,
            )
            new_token_ids.extend(replacement_ids)
        else:
            new_token_ids.append(token)

    return new_prompt, new_token_ids



def input_processor_for_mixtral_multimodal_base(
    model_config: ModelConfig,
    hf_config,
    llm_inputs: LLMInputs,
    *,
    image_token_id: int,
    audio_token_id: int,
    image_feature_size_override: Optional[int] = None,
    audio_feature_size_override: Optional[int] = None,
):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None:
        return llm_inputs

    tokenizer = cached_get_tokenizer(model_config.tokenizer)

    image_modal_exists = "image" in multi_modal_data
    audio_modal_exists = "audio" in multi_modal_data

    if image_modal_exists:
        images = multi_modal_data["image"]
        vision_config = hf_config.vision_config

        if not isinstance(images, list):
            images = [images]

        image_feature_sizes = []
        patched_images = []

        for image in images:
            patched_image = dynamic_preprocess(
                image,
                min_num=hf_config.min_dynamic_patch,
                max_num=hf_config.max_dynamic_patch,
                image_size=hf_config.vision_config.image_size,
                use_thumbnail=hf_config.use_thumbnail
            )
            patched_images += patched_image

            width, height = image.size
            num_blocks, _, _ = calculate_num_blocks(
                width, height,
                hf_config.min_dynamic_patch,
                hf_config.max_dynamic_patch,
                vision_config.image_size,
            )

            if hf_config.use_thumbnail and num_blocks > 1:
                num_blocks += 1

            assert num_blocks == len(patched_image), (
                f"Number of patches ({len(patched_image)}) does not match "
                f"the number of blocks ({num_blocks})."
            )

            image_feature_size_per_patch = get_internvl_num_patches(
                vision_config.image_size, vision_config.patch_size,
                hf_config.downsample_ratio
            )
            image_feature_size = image_feature_size_per_patch * num_blocks
            image_feature_sizes.append(image_feature_size)

        if image_feature_size_override is None:
            image_feature_size = image_feature_sizes
        else:
            image_feature_size = image_feature_size_override
        
        new_prompt, new_token_ids = repeat_and_pad_image_tokens(
            tokenizer,
            llm_inputs.get("prompt"),
            llm_inputs["prompt_token_ids"],
            image_token_id=image_token_id,
            repeat_count=image_feature_size,
        )
        multi_modal_data["image"] = patched_images


    if audio_modal_exists:

        def get_audio_feature_size(audio: torch.Tensor) -> int:
            input_size = int(audio.shape[0])
            downsample_size = ((input_size - 1) // 2 - 1) // 2
            projected_size = (downsample_size - 1) // 2 + 1
            return projected_size

        if audio_feature_size_override is None:
            audio_feature_size = [get_audio_feature_size(x) for x in multi_modal_data["audio"]]
        else:
            audio_feature_size = audio_feature_size_override

        new_prompt, new_token_ids = repeat_and_pad_image_tokens(
            tokenizer,
            new_prompt if image_modal_exists else llm_inputs.get("prompt"),
            new_token_ids if image_modal_exists else llm_inputs["prompt_token_ids"],
            image_token_id=audio_token_id,
            repeat_count=audio_feature_size,
        )
    
    return LLMInputs(prompt_token_ids=new_token_ids,
                     prompt=new_prompt,
                     multi_modal_data=multi_modal_data)


def input_processor_for_mixtral_multimodal(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or ("image" not in multi_modal_data and "audio" not in multi_modal_data):
        return llm_inputs

    model_config = ctx.model_config
    hf_config = ctx.model_config.hf_config

    return input_processor_for_mixtral_multimodal_base(
        model_config,
        hf_config,
        llm_inputs,
        image_token_id=hf_config.image_token_index,
        audio_token_id=hf_config.audio_token_index,
    )

def vision_input_mapper_for_mixtral(ctx: InputContext, data: object):
    model_config = ctx.model_config

    if not isinstance(data, List):
        data = [data]
    
    if all(isinstance(x, Image.Image) for x in data):

        image_processor = cached_get_image_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code)
        if image_processor is None:
            raise RuntimeError("No HuggingFace processor is available "
                                "to process the image object")
        try:
            batch_data = image_processor \
                .preprocess(data, return_tensors="pt").to(model_config.dtype) \
                .data
        except Exception:
            logger.error("Failed to process image (%s)", data)
            raise

        return MultiModalInputs(batch_data)
    elif all(isinstance(x, torch.Tensor) for x in data):
        raise NotImplementedError("Embeddings input is not supported yet")
    
    raise TypeError(f"Invalid image type: {type(data)}")


def dummy_data_for_mixtral_multimodal(ctx: InputContext, seq_len: int,
                            mm_counts: Mapping[str, int]):
    
    num_images = mm_counts["image"]

    hf_config = ctx.get_hf_config()
    vision_config = hf_config.vision_config


    image_feature_size = get_internvl_num_patches(
        vision_config.image_size,
        vision_config.patch_size,
        hf_config.downsample_ratio,
    )

    seq_data = dummy_seq_data_for_clip(
        vision_config,
        seq_len,
        num_images=num_images,
        image_token_id=hf_config.image_token_index,
        image_feature_size_override=image_feature_size,
    )

    mm_data = dummy_image_for_clip(
        vision_config,
        num_images,
        image_width_override=vision_config.image_size,
        image_height_override=vision_config.image_size,
    )

    return seq_data, mm_data


class MixtralMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self,
                 num_experts: int,
                 top_k: int,
                 hidden_size: int,
                 intermediate_size: int,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 tp_size: Optional[int] = None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.

        self.gate = ReplicatedLinear(hidden_size,
                                     num_experts,
                                     bias=False,
                                     params_dtype=params_dtype,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        self.experts = FusedMoE(num_experts=num_experts,
                                top_k=top_k,
                                hidden_size=hidden_size,
                                intermediate_size=intermediate_size,
                                params_dtype=params_dtype,
                                reduce_results=True,
                                renormalize=True,
                                quant_config=quant_config,
                                tp_size=tp_size,
                                prefix=f"{prefix}.experts")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


class MixtralAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn")
        self.block_sparse_moe = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe")
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MixtralDecoderLayer(
                config, cache_config, quant_config=quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers")

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_tokens(input_ids) if input_ids is not None else input_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i - self.start_layer],
                                            attn_metadata, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module, SupportsLoRA):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.model = MixtralModel(config,
                                  cache_config,
                                  quant_config,
                                  lora_config=lora_config,
                                  prefix="model")
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=quant_config,
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts)

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)


class MixtralMultiModalVisionProjector(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()

        # self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_1 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
    

class MixtralMultiModalAudioProjector(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()

        self.audio_hidden_size = config.audio_config.hidden_size
        self.text_hidden_size = config.text_config.hidden_size
        self.kernel_size = config.audio_projector_kernel_size

        self.left_padding = nn.ConstantPad1d(padding=(0, self.kernel_size - 1), value=0.0) 
        self.conv1d = nn.Conv1d(
            in_channels=self.audio_hidden_size, 
            out_channels=2 * self.audio_hidden_size, 
            kernel_size=self.kernel_size, 
            stride=2, 
            padding=0
        )
        self.norm = nn.LayerNorm(2 * self.audio_hidden_size, eps=1e-3)
        self.act = ACT2FN[config.audio_projector_hidden_act]
        self.linear = nn.Linear(2 * self.audio_hidden_size, self.text_hidden_size)

    
    def forward(self, audio_features, mask_pad=None):
        """
            x: B, T, enc_out_dim
            mask: (B, T) 
        """
        if mask_pad is not None: 
            audio_features.masked_fill_(~mask_pad.bool().unsqueeze(-1), 0.0)

        audio_features = audio_features.transpose(1, 2)  # B, channels, T

        hidden_states = self.left_padding(audio_features)
        hidden_states = self.conv1d(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear(hidden_states)

        return hidden_states, mask_pad[:, 0::2]
    
def audio_input_mapper_for_mixtral(ctx: InputContext, data: object):
    model_config = ctx.model_config

    if not isinstance(data, List):
        data = [data]
    
    from torch.nn.utils.rnn import pad_sequence
    if all(isinstance(x, torch.Tensor) for x in data):

        # x of shape (length, hidden_size)
        lengths = [x.shape[0] for x in data]

        data = pad_sequence(data, batch_first=True, padding_value=0)
        num_samples, max_length = data.shape[:2]

        # Create mask
        mask = torch.zeros((num_samples, max_length), dtype=torch.float)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1

        batch_data = {
            "audio_input": data.to(model_config.dtype),
            "audio_mask": mask.to(model_config.dtype),
        }

        return MultiModalInputs(batch_data)
    raise TypeError(f"Invalid image type: {type(data)}")


@MULTIMODAL_REGISTRY.register_input_mapper("image", vision_input_mapper_for_mixtral)
@MULTIMODAL_REGISTRY.register_input_mapper("audio", audio_input_mapper_for_mixtral)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("image", get_max_internvl_image_tokens)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("audio", 1024)

@INPUT_REGISTRY.register_dummy_data(dummy_data_for_mixtral_multimodal)
@INPUT_REGISTRY.register_input_processor(input_processor_for_mixtral_multimodal)
class MixtralForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(self,
                 config: MixtralConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 lora_config: Optional[LoRAConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config
        self.lora_config = lora_config

        vision_feature_layer = config.vision_feature_layer
        if vision_feature_layer < 0:
            num_hidden_layers = config.vision_config.num_hidden_layers \
                + vision_feature_layer + 1
        else:
            num_hidden_layers = vision_feature_layer + 1

        # TODO: Optionally initializes this for supporting embeddings.
        if hasattr(config, "vision_config"):
            self.vision_tower = InternVisionModel(
                config=config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
            ) 

            self.vision_projector = MixtralMultiModalVisionProjector(config)

        if hasattr(config, "audio_config"):
            self.audio_tower = WhaleAudioModel(config=config.audio_config,  quant_config=quant_config)
            self.audio_projector = MixtralMultiModalAudioProjector(config)

        self.quant_config = quant_config
        self.language_model = MixtralModel(config.text_config, cache_config,
                                         quant_config)
        self.unpadded_vocab_size = config.text_config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=quant_config,
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.text_config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data
    

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[MixtralImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is None:
            return None

        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        return MixtralImagePixelInputs(
            type="pixel_values",
            data=self._validate_pixel_values(pixel_values),
        )

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")


    def _image_pixels_to_features(self, vision_tower: InternVisionModel,
                                  pixel_values: torch.Tensor) -> torch.Tensor:

        image_features = vision_tower(pixel_values)

        image_features = self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

        h = w = int(image_features.shape[1] ** 0.5)
        assert image_features.shape[1] == h * w
        image_features = image_features.reshape(image_features.shape[0], h, w, -1)
        image_features = self.pixel_shuffle(image_features * 0.5)
        image_features = image_features.reshape(image_features.shape[0], -1, image_features.shape[-1])

        return image_features


    def _process_image_input(self,
                             image_input: MixtralImageInputs) -> torch.Tensor:
        assert self.vision_tower is not None
        pixel_values = image_input["data"]

        image_features = self._image_pixels_to_features(self.vision_tower, pixel_values)

        return self.vision_projector(image_features)
    

    def _process_audio_input(self,
                             inputs: MixtralAudioInputs) -> torch.Tensor:
        assert self.audio_tower is not None

        audio_input = inputs["data"]
        audio_masks = inputs["mask"]
        audio_features = self.audio_tower(audio_input, audio_masks)["last_hidden_state"]
        audio_masks = audio_masks[:, 2::2][:, 2::2]

        return self.audio_projector(audio_features, audio_masks)


    def _validate_audio_input(self, data: torch.Tensor) -> torch.Tensor:
        c, t = self.config.audio_config.num_channels, self.config.audio_config.input_dim
        expected_dims = (c, t)

        assert t == data.shape[-1]

        return data
    
    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[MixtralAudioInputs]:
        audio_input = kwargs.pop("audio_input", None)
        audio_mask = kwargs.pop("audio_mask", None)

        if audio_input is None:
            return None

        if not isinstance(audio_input, torch.Tensor):
            raise ValueError("Incorrect type of audio input. "
                             f"Got type: {type(audio_input)}")
        if audio_mask is not None and not isinstance(audio_mask, torch.Tensor):
            raise ValueError("Incorrect type of audio mask. "
                             f"Got type: {type(audio_mask)}")

        return MixtralAudioInputs(
            type="audio_input",
            data=self._validate_audio_input(audio_input),
            mask=audio_mask,
        )
    
    def merge_multimodal_embeddings(self,
            input_ids: torch.Tensor,
            input_embeds: torch.Tensor,
            vision_embeddings: BatchedTensors,
            vision_masks: Optional[torch.Tensor],
            image_token_id: int) -> torch.Tensor:
        """
        Merge `vision_embeddings` into `input_embeds` by overwriting the positions
        in `input_embeds` corresponding to placeholder image tokens in `input_ids`.

        Note:
            This updates `input_embeds` in place.
        """
        mask = (input_ids == image_token_id) 

        num_expected_tokens = mask.sum()

        if isinstance(vision_embeddings, torch.Tensor):
            batch_size, batch_tokens, *_, embed_dim = vision_embeddings.shape
            vision_embeddings = vision_embeddings.view(batch_size * batch_tokens, embed_dim)

            if vision_masks is not None:
                vision_masks = vision_masks.reshape(batch_size * batch_tokens).bool()
                vision_embeddings = vision_embeddings[vision_masks]

            total_tokens = vision_embeddings.shape[0]
            if num_expected_tokens != total_tokens:
                expr = f"{batch_size} x {batch_tokens}"
                raise ValueError(
                    f"Attempted to assign {expr} = {total_tokens} "
                    f"image tokens to {num_expected_tokens} placeholders")

            input_embeds[mask] = vision_embeddings.view(total_tokens, embed_dim)
        else:
            size_per_batch = [t.shape[0] for t in vision_embeddings]
            total_tokens = sum(size_per_batch)
            if num_expected_tokens != total_tokens:
                expr = ' + '.join(map(str, size_per_batch))
                raise ValueError(
                    f"Attempted to assign {expr} = {total_tokens} "
                    f"image tokens to {num_expected_tokens} placeholders")

            input_embeds[mask] = torch.cat(vision_embeddings)

        return input_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> SamplerOutput:

        input_embeds = self.language_model.embed_tokens(input_ids)

        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is not None:
            vision_embeddings = self._process_image_input(image_input)

            input_embeds = self.merge_multimodal_embeddings(
                    input_ids, input_embeds, vision_embeddings, None,
                    self.config.image_token_index,
            )

        audio_input = self._parse_and_validate_audio_input(**kwargs)

        if audio_input is not None:
            audio_embeddings, audio_masks = self._process_audio_input(audio_input)

            input_embeds = self.merge_multimodal_embeddings(
                    input_ids, input_embeds, audio_embeddings, audio_masks,
                    self.config.audio_token_index,
            )
        
        if image_input is not None or audio_input is not None:
            input_ids = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            kv_caches,
                                            attn_metadata,
                                            intermediate_tensors=None,
                                            input_embeds=input_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        _KEYS_TO_MODIFY_MAPPING_LLM = {
            "language_model.lm_head": "lm_head",
            "language_model.model": "language_model",
            "model.layer": "language_model.layer",
            "model.embed_tokens": "language_model.embed_tokens",
            "model.norm": "language_model.norm",
        }

        _KEYS_TO_MODIFY_MAPPING_VISION = {
            "model.vision_tower.vision_tower": "vision_tower",
            "model.mm_projector.0": "vision_projector.linear_1",
            "model.mm_projector.2": "vision_projector.linear_2",
        }

        _KEYS_TO_MODIFY_MAPPING_AUDIO = {
            "self_attn": "attn",
            "model.audio_encoder.encoder.enc.0.core.conv": "audio_tower.subsampling.conv_in",
            "model.audio_encoder.encoder.enc.0.core.out.0": "audio_tower.subsampling.out",
            "model.audio_encoder.encoder.enc.1.embed": "audio_tower.embeddings.embedding",
            "model.audio_encoder.encoder.enc.1.encoders": "audio_tower.encoder.layers",
            "model.audio_encoder.encoder.enc.1.after_norm": "audio_tower.encoder.layer_norm",
            "model.audio_encoder.adpter.bn2": "audio_projector.norm",
            "model.audio_encoder.adpter.conv1d2": "audio_projector.conv1d",
            "model.audio_encoder.adpter.project": "audio_projector.linear",
        }

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.text_config.num_local_experts)

        params_dict = dict(self.named_parameters())

        intialized_dict = {}
        for name, loaded_weight in weights:

            orig_name = name
            intialized_dict[orig_name] = False 

            if not hasattr(self, "vision_tower") and ("vision_tower" in orig_name or "mm_projector" in orig_name):   
                continue
            if not hasattr(self, "audio_tower") and ("audio_encoder" in orig_name or "audio_projector" in orig_name):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING_LLM.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            
            if "vision_tower" in orig_name or "mm_projector" in orig_name:
                for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING_VISION.items():
                    if key_to_modify in name:
                        name = name.replace(key_to_modify, new_key)
            
            if "audio_encoder" in orig_name:
                if "global_cmvn" in name:
                    continue

                for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING_AUDIO.items():
                    if key_to_modify in name:
                        name = name.replace(key_to_modify, new_key)

            for (param_name, weight_name, shard_id) in stacked_params_mapping:

                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                intialized_dict[orig_name] = True
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    intialized_dict[orig_name] = True
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            print_warning_once(
                                "Found kv scale in the checkpoint "
                                f"(e.g. {name}), but not found the expected "
                                f"name in the model "
                                f"(e.g. {remapped_kv_scale_name}). "
                                "kv-scale is not loaded.")
                            continue
                        else:
                            name = remapped_kv_scale_name

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
   
                    weight_loader(param, loaded_weight)
                    intialized_dict[orig_name] = True

        uninitalized_names = [k for k, v in intialized_dict.items() if not v]
        if uninitalized_names:
            print(f"Uninitialized parameters: {uninitalized_names}")
