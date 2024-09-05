import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from vita.constants import AUDIO_TOKEN_INDEX, IGNORE_INDEX, IMAGE_TOKEN_INDEX

from .multimodal_encoder.builder import build_audio_encoder, build_vision_tower
from .multimodal_projector.builder import build_vision_projector


class VITAMetaModel:
    def __init__(self, config):
        super(VITAMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(
                config, delay_load=not getattr(config, "continuous_training", False)
            )
            if getattr(config, "continuous_training", False):
                config.continuous_training = False
            self.mm_projector = build_vision_projector(config)

        if hasattr(config, "mm_audio_encoder"):
            self.audio_encoder = build_audio_encoder(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_encoder(self):
        audio_encoder = getattr(self, "audio_encoder", None)
        return audio_encoder

    def initialize_vision_modules(self, model_args):
        vision_tower = model_args.vision_tower

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type")
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))

    def initialize_audio_modules(self, model_args):
        audio_encoder = model_args.audio_encoder

        setattr(self.config, "mm_audio_encoder", audio_encoder)

        if self.get_audio_encoder() is None:
            #            audio_encoder = build_audio_encoder(model_args)
            audio_encoder = build_audio_encoder(self.config)
            self.audio_encoder = audio_encoder

        # from safetensors.torch import load_file
        # import os
        # audio_weights = {}
        # import pdb; pdb.set_trace()
        # for file_name in os.listdir(model_args.model_name_or_path):
        #    if file_name.endswith('safetensors'):
        #        audio_weights.update(
        #            {k[20:]: v for k, v in load_file(os.path.join(model_args.model_name_or_path, file_name)).items() if
        #                k.startswith('model.audio_encoder.')})
        # import pdb; pdb.set_trace()
        # self.audio_encoder.load_state_dict(audio_weights, strict=True)

        checkpoint = torch.load(model_args.audio_encoder + "/final.pt", map_location="cpu")
        model_dict = self.audio_encoder.state_dict()
        for key in model_dict.keys():
            if key in checkpoint.keys():
                if model_dict[key].shape == checkpoint[key].shape:
                    model_dict[key] = checkpoint[key]
                else:
                    print(
                        "Key {} has different shape, {} VS {}".format(
                            key, model_dict[key].shape, checkpoint[key].shape
                        )
                    )
            else:
                print("Key {} has not in resume model".format(key))
        # import pdb; pdb.set_trace()
        self.audio_encoder.load_state_dict(model_dict)


class VITAMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_audio_encoder(self):
        return self.get_model().get_audio_encoder()

    def pool_feats(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = F.adaptive_avg_pool2d(x, (12, 12))
        num_tokens = x.shape[2] * x.shape[3]  # Recalculate the number of tokens after pooling
        x = x.reshape(b, c, num_tokens).permute(0, 2, 1)
        return x

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_frameCat(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        assert len(image_features) % 5 == 0

        concatenated_features = []
        for i in range(0, len(image_features), 5):
            tensors_to_concat = [image_features[j] for j in range(i, i + 5)]
            concatenated_tensor = torch.cat(tensors_to_concat, dim=-1)
            concatenated_features.append(concatenated_tensor)
        concatenated_features = torch.stack(concatenated_features)
        image_features = concatenated_features

        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, audios
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            # image_features = self.encode_images(concat_images)
            image_features = self.encode_images_frameCat(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            # image_features = self.encode_images(images).to(self.device)
            image_features = self.encode_images_frameCat(images).to(self.device)

        audio_encoder = self.get_audio_encoder()
        # audio_features = audio_encoder(audios['audios'], audios['lengths'])
        if audios is not None:
            audio_features = audio_encoder(audios["audios"], audios["lengths"])
        else:
            audio_features = None

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_audio_idx = 0
        # assert sum([(cur == IMAGE_TOKEN_INDEX).sum() for cur in input_ids]) <= image_features.shape[0]
        # assert sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids]) <= audio_features['inputs_embeds'].shape[0]
        assert (
            sum([(cur == IMAGE_TOKEN_INDEX).sum() for cur in input_ids])
            + sum([(IMAGE_TOKEN_INDEX not in cur) for cur in input_ids])
            == image_features.shape[0]
        )
        assert (
            sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids])
            + sum([(AUDIO_TOKEN_INDEX not in cur) for cur in input_ids])
            == audio_features["inputs_embeds"].shape[0]
        )
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_audio_frames = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
            if num_images == 0 and num_audio_frames == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_audio_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                cur_audio_idx += 1
                continue

            image_audio_token_indices = (
                [-1]
                + torch.where(
                    (cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == AUDIO_TOKEN_INDEX)
                )[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim_noau = []
            cur_labels = labels[batch_idx]
            cur_labels_noim_noau = []
            for i in range(len(image_audio_token_indices) - 1):
                cur_input_ids_noim_noau.append(
                    cur_input_ids[
                        image_audio_token_indices[i] + 1 : image_audio_token_indices[i + 1]
                    ]
                )
                cur_labels_noim_noau.append(
                    cur_labels[image_audio_token_indices[i] + 1 : image_audio_token_indices[i + 1]]
                )

            split_sizes = [x.shape[0] for x in cur_labels_noim_noau]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim_noau))
            cur_input_embeds_no_im_no_au = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + num_audio_frames + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im_no_au[i])
                cur_new_labels.append(cur_labels_noim_noau[i])
                if i < num_images + num_audio_frames:
                    if cur_input_ids[image_audio_token_indices[i + 1]] == IMAGE_TOKEN_INDEX:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                    elif cur_input_ids[image_audio_token_indices[i + 1]] == AUDIO_TOKEN_INDEX:
                        cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                        cur_audio_idx += 1
                        cur_new_input_embeds.append(cur_audio_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_audio_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                    else:
                        raise ValueError

            if num_images != 0 and num_audio_frames == 0:
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_audio_idx += 1
                cur_new_input_embeds.append(cur_audio_features[0:0])
            elif num_images == 0 and num_audio_frames != 0:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features[0:0])
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        assert cur_image_idx == image_features.shape[0]
        assert cur_audio_idx == audio_features["inputs_embeds"].shape[0]
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
