import os
import warnings

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, logging

from vita.constants import GLOBAL_WEIGHTS_PATH
from vita.model import *

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    model_type,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    if model_type not in {"mixtral-8x7b"}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    # Load VITA model
    if "lora" in model_name.lower() and model_base is None:
        warnings.warn(
            "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument."
        )
    if "lora" in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

        print("Loading VITA from base model...")
        if model_type == "mixtral-8x7b":
            # import pdb; pdb.set_trace()
            device_map = {
                "model.embed_tokens": 0,
                "model.layers.0": 0,
                "model.layers.1": 0,
                "model.layers.2": 0,
                "model.layers.3": 0,
                "model.layers.4": 0,
                "model.layers.5": 0,
                "model.layers.6": 0,
                "model.layers.7": 0,
                "model.layers.8": 0,
                "model.layers.9": 0,
                "model.layers.10": 0,
                "model.layers.11": 0,
                "model.layers.12": 0,
                "model.layers.13": 0,
                "model.layers.14": 0,
                "model.layers.15": 0,
                "model.layers.16": 1,
                "model.layers.17": 1,
                "model.layers.18": 1,
                "model.layers.19": 1,
                "model.layers.20": 1,
                "model.layers.21": 1,
                "model.layers.22": 1,
                "model.layers.23": 1,
                "model.layers.24": 1,
                "model.layers.25": 1,
                "model.layers.26": 1,
                "model.layers.27": 1,
                "model.layers.28": 1,
                "model.layers.29": 1,
                "model.layers.30": 1,
                "model.layers.31": 1,
                "model.norm": 1,
                "model.vision_tower": 1,
                "model.mm_projector": 1,
                "model.audio_encoder": 1,
                "lm_head": 1,
            }
            device_map["model.audio_encoder"] = 0
            kwargs.update(device_map=device_map)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = VITAMixtralForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
            )
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
            )

        print("Loading additional VITA weights...")
        if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(
                os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
            )
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download

            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id, filename=filename, subfolder=subfolder
                )
                return torch.load(cache_file, map_location="cpu")

            non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")

        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
            }
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel

        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        print("Model is loaded...")
    elif model_base is not None:
        # this may be mm projector only
        print("Loading VITA from base model...")

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_type == "mixtral-8x7b":
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = VITAMixtralForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            from types import SimpleNamespace

            model_args = {
                "vision_tower": f"{GLOBAL_WEIGHTS_PATH}/InternViT-300M-448px",
                "pretrain_mm_mlp_adapter": None,
                "mm_projector_type": "mlp2x_gelu",
            }
            model_args = SimpleNamespace(**model_args)
            model.get_model().initialize_vision_modules(model_args=model_args)

        mm_projector_weights = torch.load(
            os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
        )
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
        model.model.mm_projector.to(device="cuda", dtype=torch.float16)
        model.model.vision_tower.to(device="cuda", dtype=torch.float16)
    else:
        if model_type == "mixtral-8x7b":
            # import pdb; pdb.set_trace()
            device_map = {
                "model.embed_tokens": 0,
                "model.layers.0": 0,
                "model.layers.1": 0,
                "model.layers.2": 0,
                "model.layers.3": 0,
                "model.layers.4": 0,
                "model.layers.5": 0,
                "model.layers.6": 0,
                "model.layers.7": 0,
                "model.layers.8": 0,
                "model.layers.9": 0,
                "model.layers.10": 0,
                "model.layers.11": 0,
                "model.layers.12": 0,
                "model.layers.13": 0,
                "model.layers.14": 0,
                "model.layers.15": 0,
                "model.layers.16": 1,
                "model.layers.17": 1,
                "model.layers.18": 1,
                "model.layers.19": 1,
                "model.layers.20": 1,
                "model.layers.21": 1,
                "model.layers.22": 1,
                "model.layers.23": 1,
                "model.layers.24": 1,
                "model.layers.25": 1,
                "model.layers.26": 1,
                "model.layers.27": 1,
                "model.layers.28": 1,
                "model.layers.29": 1,
                "model.layers.30": 1,
                "model.layers.31": 1,
                "model.norm": 1,
                "model.vision_tower": 1,
                "model.mm_projector": 1,
                "model.audio_encoder": 1,
                "lm_head": 1,
            }
            device_map["model.audio_encoder"] = 0
            kwargs.update(device_map=device_map)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = VITAMixtralForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
            # model.hf_device_map

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    num_params = sum(p.numel() for p in vision_tower.parameters())
    print("the number of vision encoder params: {}M".format(num_params / 1024 / 1024))

    if getattr(model.config, "unfreeze_vision_tower", False):
        if "lora" in model_name.lower():
            assert model_base is not None
            vision_non_lora_trainables = {
                k[19:]: v
                for k, v in non_lora_trainables.items()
                if k.startswith("model.vision_tower.")
            }
            vision_tower.load_state_dict(vision_non_lora_trainables, strict=False)
        else:
            assert model_base is None
            from safetensors.torch import load_file

            vision_weights = {}
            for file_name in os.listdir(model_path):
                if file_name.endswith("safetensors"):
                    vision_weights.update(
                        {
                            k[19:]: v
                            for k, v in load_file(os.path.join(model_path, file_name)).items()
                            if k.startswith("model.vision_tower.")
                        }
                    )
            vision_tower.load_state_dict(vision_weights, strict=True)

    # from types import SimpleNamespace
    # model_args = {
    #    'audio_encoder': f"{GLOBAL_WEIGHTS_PATH}/audio-encoder-2wh_zh_en_audioset_Mixtral-8x7B_New-base-tunning',
    #    'freeze_audio_encoder': True,
    #    'freeze_audio_encoder_adapter': True
    # }
    # model_args = SimpleNamespace(**model_args)
    # model.get_model().initialize_audio_modules(model_args=model_args)
    # audio_encoder = model.get_audio_encoder()

    # import pdb; pdb.set_trace()
    # if (not getattr(model.config, "freeze_audio_encoder", True)) and (not getattr(model.config, "freeze_audio_encoder_adapter", True)):
    #    from safetensors.torch import load_file
    #    audio_weights = {}
    #    for file_name in os.listdir(model_path):
    #        if file_name.endswith('safetensors'):
    #            audio_weights.update(
    #                {k[20:]: v for k, v in load_file(os.path.join(model_path, file_name)).items() if
    #                    k.startswith('model.audio_encoder.')})
    #    audio_encoder.load_state_dict(audio_weights, strict=True)
    #    audio_encoder.eval()
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # from safetensors.torch import load_file
    # audio_weights = {}
    # for file_name in os.listdir(model_path):
    #    if file_name.endswith('safetensors'):
    #        audio_weights.update(
    #            {k[20:]: v for k, v in load_file(os.path.join(model_path, file_name)).items() if
    #                k.startswith('model.audio_encoder.')})
    # import pdb; pdb.set_trace()

    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if model_type == "phi-3":
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    return tokenizer, model, image_processor, context_len
