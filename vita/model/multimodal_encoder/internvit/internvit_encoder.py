import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

from .modeling_intern_vit import InternVisionModel


class InternViTVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -1
        self.scale_pix_shuffle = 0.5

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(
                self.vision_tower_name, trust_remote_code=True
            )

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        self.vision_tower = InternVisionModel.from_pretrained(
            self.vision_tower_name, trust_remote_code=True
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        image_features = image_features[:, 1:]

        return image_features

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        h = w = int(image_features.shape[1] ** 0.5)
        assert image_features.shape[1] == h * w
        image_features = image_features.reshape(image_features.shape[0], h, w, -1)
        image_features = self.pixel_shuffle(image_features * self.scale_pix_shuffle)
        image_features = image_features.reshape(
            image_features.shape[0], -1, image_features.shape[-1]
        )

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size * (int(1 / self.scale_pix_shuffle) ** 2)

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
