from __future__ import annotations

from ranzen.hydra import Option
import torch.multiprocessing

from whaledo.algorithms import Moco, SimClr
from whaledo.data.datamodule import WhaledoDataModule
from whaledo.models.artifact import ArtifactLoader
from whaledo.models.backbones import Beit, ConvNeXt, ResNet, Swin, SwinV2, ViT
from whaledo.models.meta.ema import EmaModel
from whaledo.models.meta.ft import BitFit
from whaledo.relay import WhaledoRelay

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    dm_ops: list[Option] = [
        Option(WhaledoDataModule, name="whaledo"),
    ]
    alg_ops: list[Option] = [
        Option(Moco, "moco"),
        Option(SimClr, "simclr"),
    ]
    bb_ops: list[Option] = [
        Option(Beit, "beit"),
        Option(ConvNeXt, "convnext"),
        Option(ResNet, "resnet"),
        Option(Swin, "swin"),
        Option(SwinV2, "swinv2"),
        Option(ViT, "vit"),
        Option(ArtifactLoader, "artifact"),
    ]

    mm_ops: list[Option] = [
        Option(BitFit, "bitfit"),
        Option(EmaModel, "ema"),
    ]

    WhaledoRelay.with_hydra(
        root="conf",
        dm=dm_ops,
        alg=alg_ops,
        backbone=bb_ops,
        meta_model=mm_ops,
        clear_cache=True,
    )
