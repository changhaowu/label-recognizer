import torch
from torch import nn
from PIL import Image
from einops import rearrange
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
    ToTensor,
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


# class VisionEncoder(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#         self.encoder = ModelHolder(
#             VisualHolder(timm.create_model("vit_so400m_patch14_siglip_384"))
#         )
#         self.encoder.model.visual.patch_embed = LinearPatchEmbedding(
#             self.encoder.model.visual.patch_embed.proj
#         )
#         self.encoder.model.visual.attn_pool = nn.Identity()

#         self.projection = VisionProjection()

#         # self.preprocess = Compose(
#         #     [
#         #         Resize(size=(378, 378), interpolation=InterpolationMode.BICUBIC),
#         #         ToImage(),
#         #         ToDtype(torch.float32, scale=True),
#         #         Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         #     ]
#         # )
#         self.preprocess = self.create_preprocess_pipeline()

#     def create_preprocess_pipeline(self):
#         return Compose(
#             [
#                 self.dynamic_resize,
#                 ToTensor(),
#                 Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#             ]
#         )

#     def dynamic_resize(self, images):
#         if isinstance(images, list):
#             resized_images = []
#             for image in images:
#                 width, height = image.size
#                 aspect_ratio = width / height
#                 if aspect_ratio > 1:
#                     # Ensure both dimensions are multiples of 14
#                     target_size = (1904, 1204)  # 大图片的尺寸
#                 else:
#                     # Ensure both dimensions are multiples of 14
#                     target_size = (392, 252)  # 小图片的尺寸
#                 resized_image = Resize(
#                     size=target_size, interpolation=InterpolationMode.BICUBIC
#                 )(image)
#                 resized_images.append(resized_image)
#             return resized_images
#         else:
#             # 单个图像处理
#             width, height = images.size
#             aspect_ratio = width / height
#             if aspect_ratio > 1:
#                 target_size = (1900, 1200)  # 大图片的尺寸
#             else:
#                 target_size = (400, 250)  # 小图片的尺寸
#             return Resize(size=target_size, interpolation=InterpolationMode.BICUBIC)(
#                 images
#             )

#     def forward(self, images):
#         # Ensure images are preprocessed
#         preprocessed_images = [self.preprocess(image) for image in images]
#         # Further processing can be added here as needed
#         return preprocessed_images

#     @property
#     def device(self):
#         return self.projection.mlp.fc1.weight.device

#     @property
#     def dtype(self):
#         return self.projection.mlp.fc1.weight.dtype

#     def __call__(self, images) -> torch.Tensor:
#         if not isinstance(images, list):
#             images = [images]

#         with torch.no_grad():
#             x = torch.stack(
#                 [self.preprocess(image.convert("RGB")) for image in images]
#             ).to(self.device, dtype=self.dtype)

#             x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)

#             x = self.encoder(x)
#             x = self.projection(x)

#             return x


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

        self.preprocess_large = self.create_preprocess_pipeline((1904, 1204))
        self.preprocess_small = self.create_preprocess_pipeline((392, 252))

    def create_preprocess_pipeline(self, target_size):
        def dynamic_resize(image):
            return Resize(size=target_size, interpolation=InterpolationMode.BICUBIC)(
                image
            )

        return Compose(
            [
                dynamic_resize,
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def preprocess_images(self, images):
        large_images = []
        small_images = []
        preprocessed_large_images = []
        preprocessed_small_images = []

        for image in images:
            width, height = image.size
            aspect_ratio = width / height
            if aspect_ratio > 1:
                large_images.append(image)
            else:
                small_images.append(image)

        if large_images:
            preprocessed_large_images = [
                self.preprocess_large(image) for image in large_images
            ]
            preprocessed_large_images = torch.stack(preprocessed_large_images)
            preprocessed_large_images = rearrange(
                preprocessed_large_images,
                "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                p1=14,
                p2=14,
            )

        if small_images:
            preprocessed_small_images = [
                self.preprocess_small(image) for image in small_images
            ]
            preprocessed_small_images = torch.stack(preprocessed_small_images)
            preprocessed_small_images = rearrange(
                preprocessed_small_images,
                "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                p1=14,
                p2=14,
            )

        return preprocessed_large_images, preprocessed_small_images

    def forward(self, images):
        # Ensure images are preprocessed
        preprocessed_images = [self.preprocess(image) for image in images]
        # Further processing can be added here as needed
        return preprocessed_images

    @property
    def device(self):
        return self.projection.mlp.fc1.weight.device

    @property
    def dtype(self):
        return self.projection.mlp.fc1.weight.dtype

    def __call__(self, images) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        with torch.no_grad():
            x = torch.stack(
                [self.preprocess(image.convert("RGB")) for image in images]
            ).to(self.device, dtype=self.dtype)

            x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)

            x = self.encoder(x)
            x = self.projection(x)

            return x
