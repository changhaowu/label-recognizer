import torch
from torch import nn
from PIL import Image
from einops import rearrange
import numpy as np
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
    ToTensor,
    CenterCrop,
    Pad,
)
import timm


class VisualHolder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual = model

    def forward(self, x):
        return self.visual(x)


class ModelHolder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class LinearPatchEmbedding(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.linear = nn.Linear(588, 1152)
        self.linear.weight.data = conv.weight.data.view(1152, -1)
        if conv.bias is not None:
            self.linear.bias.data = conv.bias.data

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

        torch.nn.init.kaiming_normal_(
            self.fc1.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_normal_(
            self.fc2.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class VisionProjection(nn.Module):
    def __init__(self):
        super().__init__()

        image_embedding_dim = 1152
        model_dim = 2048
        hidden_dim = model_dim * 4

        self.mlp = MLP(image_embedding_dim, hidden_dim, model_dim)

    @property
    def device(self):
        return self.mlp.fc1.weight.device

    def forward(self, x):
        return self.mlp(x)


class VisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = ModelHolder(
            VisualHolder(timm.create_model("vit_so400m_patch14_siglip_384"))
        )
        self.encoder.model.visual.patch_embed = LinearPatchEmbedding(
            self.encoder.model.visual.patch_embed.proj
        )
        self.encoder.model.visual.attn_pool = nn.Identity()

        self.projection = VisionProjection()

        self.preprocess_wide = self.create_preprocess_pipeline((1500, 500))
        self.preprocess_tall = self.create_preprocess_pipeline((1000, 1500))

    def create_preprocess_pipeline(self, target_size):
        def dynamic_resize(image):
            width, height = image.size
            target_width, target_height = target_size
            if width / height > target_width / target_height:
                # If image is wider than the target aspect ratio
                new_width = 378
                new_height = int(378 * height / width)
            else:
                # If image is taller than the target aspect ratio
                new_height = 378
                new_width = int(378 * width / height)
            return Resize(
                (new_height, new_width), interpolation=InterpolationMode.BICUBIC
            )(image)

        def pad_to_target_size(image):
            width, height = image.size
            padding_left = (378 - width) // 2
            padding_top = (378 - height) // 2
            padding_right = 378 - width - padding_left
            padding_bottom = 378 - height - padding_top
            avg_color = tuple(np.array(image).mean(axis=(0, 1)).astype(int))
            return Pad(
                (padding_left, padding_top, padding_right, padding_bottom),
                fill=avg_color,
            )(image)

        return Compose(
            [
                dynamic_resize,
                pad_to_target_size,
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def preprocess_images(self, images):
        single_image = False
        if not isinstance(images, list):
            images = [images]
            single_image = True

        processed_images = []

        for image in images:
            width, height = image.size
            aspect_ratio = width / height
            if aspect_ratio > 1:
                processed_image = self.preprocess_wide(image)
            else:
                processed_image = self.preprocess_tall(image)
            processed_images.append(processed_image)

        preprocessed_images = torch.stack(processed_images)

        if single_image:
            return preprocessed_images[0]
        else:
            return preprocessed_images

    @property
    def device(self):
        return self.projection.mlp.fc1.weight.device

    @property
    def dtype(self):
        return self.projection.mlp.fc1.weight.dtype

    def __call__(self, images) -> torch.Tensor:
        single_image = False
        if not isinstance(images, list):
            images = [images]
            single_image = True

        with torch.no_grad():

            x = torch.stack(
                [self.preprocess_images(image.convert("RGB")) for image in images]
            ).to(self.device, dtype=self.dtype)

            x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)

            x = self.encoder(x)
            x = self.projection(x)

            # if single_image:
            #     x = x[0]

            return x

    def forward(self, images) -> torch.Tensor:
        return self.__call__(images)
