from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, OrderedDict, cast

from classy_vision.models import RegNet as ClassyRegNet  # type: ignore
from classy_vision.models.anynet import (  # type: ignore
    ActivationType,
    BlockType,
    StemType,
)
from classy_vision.models.regnet import RegNetParams  # type: ignore
import clip  # type: ignore
from conduit.logging import init_logger
from hydra.utils import to_absolute_path
from ranzen.decorators import implements
import timm  # type: ignore
import timm.models as tm  # type: ignore
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchvision.models as tvm  # type: ignore

from src.models.base import BackboneFactory, ModelFactoryOut

__all__ = [
    "Beit",
    "Clip",
    "ConvNeXt",
    "RegNet",
    "ResNet",
    "Swin",
    "ViT",
]


LOGGER = init_logger(__file__)


@torch.no_grad()
def expand_in_channels_(conv: nn.Conv2d, *, in_channels: int) -> None:
    in_channels_curr = conv.weight.size(1)
    if in_channels == in_channels_curr:
        return None
    elif in_channels < in_channels_curr:
        raise ValueError("'in_channels' cannot be less than the current number of input channels.")
    rgb_mean = conv.weight.mean(dim=1, keepdim=True)
    weight_padding = rgb_mean.expand(-1, in_channels - in_channels_curr, -1, -1)
    new_weight = Parameter(torch.cat((conv.weight, weight_padding), dim=1))
    new_weight *= in_channels_curr / in_channels
    conv.weight = new_weight
    conv.in_channels = in_channels


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
        expand_in_channels_(model.conv1, in_channels=self.in_channels)
        out_dim = model.fc.in_features
        model.fc = nn.Identity()  # type: ignore
        return model, out_dim


class ClipVersion(Enum):
    RN50 = "RN50"
    RN101 = "RN101"
    RN50x4 = "RN50x4"
    RN50x16 = "RN50x16"
    RN50x64 = "RN50x64"
    ViT_B32 = "ViT-B/32"
    ViT_B16 = "ViT-B/16"
    ViT_L14 = "ViT-L/14"


@dataclass
class Clip(BackboneFactory):
    download_root: Optional[str] = None
    version: ClipVersion = ClipVersion.RN50
    in_channels: int = 3

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[clip.model.CLIP]:  # type: ignore

        model, _ = clip.load(
            name=self.version.value, device="cpu", download_root=self.download_root  # type: ignore
        )
        visual_model = model.visual
        expand_in_channels_(visual_model.conv1, in_channels=self.in_channels)
        out_dim = visual_model.output_dim
        return visual_model, out_dim


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
    def __call__(self) -> ModelFactoryOut[tm.ConvNeXt]:
        model: tm.ConvNeXt = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        expand_in_channels_(cast(nn.Conv2d, model.stem[0]), in_channels=self.in_channels)
        model.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        return model, model.num_features


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
