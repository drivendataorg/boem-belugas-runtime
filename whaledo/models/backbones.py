from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, OrderedDict, Union, cast

from classy_vision.models import RegNet as ClassyRegNet  # type: ignore
from classy_vision.models.anynet import (  # type: ignore
    ActivationType,
    BlockType,
    StemType,
)
from classy_vision.models.regnet import RegNetParams  # type: ignore
from conduit.logging import init_logger
from hydra.utils import to_absolute_path
from ranzen.decorators import implements
import timm  # type: ignore
import timm.models as tm  # type: ignore
import torch
import torch.nn as nn
import torchvision.models as tvm  # type: ignore

from whaledo.models.base import BackboneFactory, ModelFactoryOut

__all__ = [
    "Beit",
    "ConvNeXt",
    "RegNet",
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


@dataclass
class RegNet(BackboneFactory):
    """
    Wrapper for ClassyVision RegNet model so we can map layers into feature
    blocks to facilitate feature extraction and benchmarking at several layers.
    This model is defined on the fly from a RegNet base class and a configuration file.
    We follow the feature naming convention defined in the ResNet vissl trunk.
    [ Adapted from VISSL ]
    """

    depth: int
    w_0: int
    w_a: float
    w_m: float
    group_width: int
    bottleneck_multiplier: float = 1.0
    stem_type: StemType = StemType.SIMPLE_STEM_IN
    stem_width: int = 32
    block_type: BlockType = BlockType.RES_BOTTLENECK_BLOCK
    activation: ActivationType = ActivationType.RELU
    use_se: bool = True
    se_ratio: float = 0.25
    bn_epsilon: float = 1e-05
    bn_momentum: float = 0.1
    checkpoint: Optional[str] = None
    regnet_params: RegNetParams = field(init=False)

    def __post_init__(self) -> None:
        self.regnet_params = RegNetParams(
            depth=self.depth,
            w_0=self.w_0,
            w_a=self.w_a,
            w_m=self.w_m,
            group_width=self.group_width,
            bottleneck_multiplier=self.bottleneck_multiplier,
            stem_type=self.stem_type,
            stem_width=self.stem_width,
            block_type=self.block_type,
            activation=self.activation,
            use_se=self.use_se,
            se_ratio=self.se_ratio,
            bn_epsilon=self.bn_epsilon,
            bn_momentum=self.bn_momentum,  # type: ignore
        )

    def _load_from_checkpoint(self, model: nn.Module, *, checkpoint: str | Path) -> None:
        checkpoint = Path(to_absolute_path(str(checkpoint)))
        if not checkpoint.exists():
            raise AttributeError(f"Checkpoint '{checkpoint}' does not exist.")
        LOGGER.info(
            f"Attempting to load {self.__class__.__name__} model from path '{str(checkpoint)}'."
        )

        state_dict = torch.load(f=checkpoint, map_location=torch.device("cpu"))
        trunk_params = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        model.load_state_dict(trunk_params)
        LOGGER.info(
            f"Successfully loaded {self.__class__.__name__} model from path '{str(checkpoint)}'."
        )

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[nn.Sequential]:
        regnet = ClassyRegNet(self.regnet_params)  # type: ignore
        # Now map the models to the structure we want to expose for SSL tasks
        # The upstream RegNet model is made of :
        # - `stem`
        # - n x blocks in trunk_output, named `block1, block2, ..`

        # We're only interested in the stem and successive blocks
        # everything else is not picked up on purpose
        stem = cast(nn.Sequential, regnet.stem)
        feature_blocks_d: dict[str, nn.Module] = OrderedDict({"conv1": stem})
        # - get all the feature blocks
        for name, module in regnet.trunk_output.named_children():  # type: ignore
            if not name.startswith("block"):
                raise AttributeError(f"Unexpected layer name {name}")
            block_index = len(feature_blocks_d) + 1

            feature_blocks_d[f"res{block_index}"] = module

        # - finally, add avgpool and flatten.
        feature_blocks_d["avgpool"] = nn.AdaptiveAvgPool2d((1, 1))
        feature_blocks_d["flatten"] = nn.Flatten()

        feature_blocks = nn.Sequential(feature_blocks_d)

        if self.checkpoint is not None:
            self._load_from_checkpoint(model=feature_blocks, checkpoint=self.checkpoint)

        out_dim: int = cast(int, regnet.trunk_output[-1][0].proj.out_channels)  # type: ignore

        return feature_blocks, out_dim


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
            classifier.stem, classifier.stages, nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
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
    BASE_W12TO24_192TO384 = "swinv2_base_window12_192_22k"
    CR_BASE_224 = "swinv2_cr_base_224"
    CR_BASE_384 = "swinv2_cr_base_384"
    LARGE_W12TO24_192TO384 = "swinv2_large_window12to24_192to384_22kft1k"
    LARGE_W12_192 = "swinv2_large_window12_192_22k"
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
