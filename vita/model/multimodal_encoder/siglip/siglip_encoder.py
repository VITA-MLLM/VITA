import torch
import torch.nn as nn
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

from vita.util.s2wrapper import forward as multiscale_forward


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.crop_size = self.image_processor.size
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        return image_features

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
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class SiglipVisionTowerS2(SiglipVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.s2_scales = getattr(args, "s2_scales", "384,768,1152")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        self.multiscale_forward = multiscale_forward

        if not delay_load:
            self.image_processor.size["height"] = self.image_processor.size[
                "width"
            ] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size[
                "width"
            ] = self.s2_image_size

    def load_model(self):
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.crop_size = self.image_processor.size
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size["height"] = self.image_processor.size[
            "width"
        ] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size[
            "width"
        ] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size,
                )
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size,
            )

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
