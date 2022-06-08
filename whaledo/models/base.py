from dataclasses import dataclass
import logging
from typing import Any, Protocol, Tuple, TypeVar

from conduit.logging import init_logger
from ranzen.decorators import implements
from src.data.datamodules.base import WILDSDataModule  # type: ignore
from torch import Tensor
import torch.nn as nn
from typing_extensions import Self, TypeAlias

__all__ = [
    "BackboneFactory",
    "Model",
]


M = TypeVar("M", bound=nn.Module)
ModelFactoryOut: TypeAlias = Tuple[M, int]


class BackboneFactory(Protocol):
    def __call__(self) -> ModelFactoryOut:
        ...


@dataclass(unsafe_hash=True)
class Model(nn.Module):
    backbone: nn.Module
    feature_dim: int

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        return obj

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = init_logger(self.__class__.__name__)
        return self._logger

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
