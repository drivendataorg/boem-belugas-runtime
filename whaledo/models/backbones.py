from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Union

from conduit.logging import init_logger
from ranzen.decorators import implements
import timm  # type: ignore
import timm.models as tm  # type: ignore
import torch.nn as nn
import torchvision.models as tvm  # type: ignore

from whaledo.models.base import BackboneFactory, ModelFactoryOut

__all__ = [
    "Beit",
    "ConvNeXt",
    "ResNet",
    "Swin",
    "SwinV2",
    "ViT",
]


LOGGER = init_logger(__file__)


class ResNetVersion(Enum):
    RN18 = "18"
    RN34 = "34"
    RN50 = "50"
    RN101 = "101"


@dataclass
class ResNet(BackboneFactory):
    in_channels: int = 3
    pretrained: bool = True
    version: ResNetVersion = ResNetVersion.RN18

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[tvm.ResNet]:
        model: tvm.ResNet = getattr(tvm, f"resnet{self.version.value}")(pretrained=self.pretrained)
        out_dim = model.fc.in_features
        model.fc = nn.Identity()  # type: ignore
        return model, out_dim


class ConvNeXtVersion(Enum):
    TINY = "convnext_tiny"
    SMALL = "convnext_small"
    BASE = "convnext_base"
    BASE_21K = "convnext_base_in22k"
    LARGE = "convnext_large"
    LARGE_21K = "convnext_large_in22k"
    XLARGE_21K = "convnext_xlarge_in22k"


@dataclass
class ConvNeXt(BackboneFactory):
    in_channels: int = 3
    pretrained: bool = True
    version: ConvNeXtVersion = ConvNeXtVersion.BASE
    checkpoint_path: str = ""

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[nn.Sequential]:
        classifier: tm.ConvNeXt = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        num_features = classifier.num_features
        backbone = nn.Sequential(
            classifier.stem,
            classifier.stages,
            classifier.norm_pre,
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()),
        )
        return backbone, num_features


class ViTVersion(Enum):
    TINY_P16_224 = "vit_tiny_patch16_224"
    TINY_P16_224_21K = "vit_tiny_patch16_224_in21k"
    TINY_P16_384 = "vit_tiny_patch16_384"

    SMALL_P16_224_21K = "vit_small_patch16_224_in21k"
    SMALL_P32_224_21K = "vit_small_patch32_224_in21k"
    SMALL_P16_384 = "vit_small_patch16_384"
    SMALL_P32_384 = "vit_small_patch32_384"

    BASE_P8_224_21K = "vit_base_patch8_224_in21k"
    BASE_P16_224_21K = "vit_base_patch16_224_in21k"
    BASE_P16_384 = "vit_base_patch16_384"
    BASE_P32_384 = "vit_base_patch32_384"

    LARGE_P32_224_21K = "vit_large_patch32_224_in21k"
    LARGE_P16_224_21K = "vit_large_patch16_224_in21k"
    LARGE_P16_384 = "vit_large_patch16_384"
    LARGE_P32_384 = "vit_large_patch32_384"

    HUGE_P14_224_21K = "vit_huge_patch14_224_in21k"


@dataclass
class ViT(BackboneFactory):
    pretrained: bool = True
    version: ViTVersion = ViTVersion.BASE_P16_224_21K
    checkpoint_path: str = ""

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[tm.VisionTransformer]:
        model: tm.VisionTransformer = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        model.head = nn.Identity()
        return model, model.num_features


class SwinVersion(Enum):
    BASE_P4_W7_224_21K = "swin_base_patch4_window7_224_in22k"
    BASE_P4_W12_384_21K = "swin_base_patch4_window12_384_in22k"
    LARGE_P4_W12_224_21K = "swin_large_patch4_window12_224_in22k"
    LARGE_P4_W12_384_21K = "swin_large_patch4_window12_384_in22k"


@dataclass
class Swin(BackboneFactory):
    pretrained: bool = True
    version: SwinVersion = SwinVersion.BASE_P4_W7_224_21K
    checkpoint_path: str = ""

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[tm.SwinTransformer]:
        model: tm.SwinTransformer = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        model.head = nn.Identity()
        return model, model.num_features


class SwinV2Version(Enum):
    BASE_W8_256 = "swinv2_base_window8_256"
    BASE_W12_196 = "swinv2_base_window12_192_22k"
    BASE_W12TO24_192TO256 = "swinv2_base_window12_192to256_22k"
    LARGE_W12TO24_192TO384 = "swinv2_large_window12to24_192to384_22kft1k"
    LARGE_W12_192 = "swinv2_large_window12_192_22k"
    CR_BASE_224 = "swinv2_cr_base_224"
    CR_BASE_384 = "swinv2_cr_base_384"
    CR_LARGE_224 = "swinv2_cr_large_224"
    CR_LARGE_384 = "swinv2_cr_large_384"
    CR_HUGE_224 = "swinv2_cr_huge_224"
    CR_HUGE_384 = "swinv2_cr_huge_384"
    CR_GIANT_224 = "swinv2_cr_giant_224"
    CR_GIANT_384 = "swinv2_cr_giant_384"


@dataclass
class SwinV2(BackboneFactory):
    pretrained: bool = True
    version: SwinV2Version = SwinV2Version.BASE_W8_256
    checkpoint_path: str = ""
    freeze_patch_embedder: bool = True

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[Union[tm.SwinTransformerV2, tm.SwinTransformerV2Cr]]:
        model: Union[tm.SwinTransformerV2, tm.SwinTransformerV2Cr] = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        if self.freeze_patch_embedder:
            for param in model.patch_embed.parameters():
                param.requires_grad_(False)
        model.head = nn.Identity()
        return model, model.num_features


class BeitVersion(Enum):
    BASE_P16_224 = "beit_base_patch16_224"
    BASE_P16_224_21K = "beit_base_patch16_224_in22k"
    BASE_P16_384 = "beit_base_patch16_384"
    LARGE_P16_224_21K = "beit_large_patch16_224_in22k"
    LARGE_P16_384 = "beit_large_patch16_384"
    LARGE_P16_512 = "beit_large_patch16_512"


@dataclass
class Beit(BackboneFactory):

    pretrained: bool = True
    version: BeitVersion = BeitVersion.BASE_P16_224_21K
    checkpoint_path: str = ""

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[tm.Beit]:
        model: tm.Beit = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        model.head = nn.Identity()
        return model, model.num_features
