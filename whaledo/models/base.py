from dataclasses import dataclass
import logging
from typing import Any, Literal, Protocol, Tuple, TypeVar, Union, overload

import attr
from conduit.data.structures import InputContainer, concatenate_inputs
from conduit.logging import init_logger
from ranzen.decorators import implements
from ranzen.misc import gcopy
from torch import Tensor
import torch.nn as nn
from typing_extensions import Self, TypeAlias

from src.data.datamodules.base import WILDSDataModule  # type: ignore

__all__ = [
    "BackboneFactory",
    "FeaturesLogits",
    "Model",
    "PredictorFactory",
]


@attr.define(kw_only=True)
class FeaturesLogits(InputContainer):
    features: Tensor
    logits: Tensor

    def __post_init__(self) -> None:
        if len(self.features) != len(self.logits):
            raise AttributeError("'features' and 'logits' must match in size at dimension 0.")

    def __len__(self) -> int:
        return len(self.features)

    def __add__(self, other: Self) -> Self:
        copy = gcopy(self)
        copy.features = concatenate_inputs(x1=copy.features, x2=other.features, is_batched=True)
        copy.logits = concatenate_inputs(x1=copy.logits, x2=other.logits, is_batched=True)
        return copy


M = TypeVar("M", bound=nn.Module)
ModelFactoryOut: TypeAlias = Tuple[M, int]


class BackboneFactory(Protocol):
    def __call__(self) -> ModelFactoryOut:
        ...


class PredictorFactory(Protocol):
    def __call__(self, in_dim: int, *, out_dim: int) -> ModelFactoryOut:
        ...


@dataclass(unsafe_hash=True)
class Model(nn.Module):
    backbone: nn.Module
    predictor: nn.Module
    feature_dim: int
    out_dim: int

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        return obj

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = init_logger(self.__class__.__name__)
        return self._logger

    def features(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    @overload
    def forward(self, x: Tensor, *, return_features: Literal[True]) -> FeaturesLogits:
        ...

    @overload
    def forward(self, x: Tensor, *, return_features: Literal[False] = ...) -> Tensor:
        ...

    @implements(nn.Module)
    def forward(self, x: Tensor, *, return_features: bool = False) -> Union[Tensor, FeaturesLogits]:
        features = self.backbone(x)
        logits = self.predictor(features)
        if return_features:
            return FeaturesLogits(features=features, logits=logits)
        return logits
