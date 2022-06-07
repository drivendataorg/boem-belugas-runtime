from dataclasses import dataclass, field
from typing import Any, Literal, Union, overload

from ranzen.decorators import implements
from torch import Tensor
import torch.nn as nn
from typing_extensions import Self

from src.models.base import FeaturesLogits, Model

__all__ = ["MetaModel"]


@dataclass(unsafe_hash=True)
class MetaModel(nn.Module):
    model: Union[Model, "MetaModel"]
    # Expose the backbone/predictor attributes.
    backbone: nn.Module = field(init=False)
    predictor: nn.Module = field(init=False)
    feature_dim: int = field(init=False)
    out_dim: int = field(init=False)

    def __post_init__(self) -> None:
        # Expose the backbone/predictor attributes.
        self.backbone = self.model.backbone
        self.predictor = self.model.predictor
        self.feature_dim = self.model.feature_dim
        self.out_dim = self.model.out_dim

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        return obj

    @overload
    def forward(self, x: Tensor, *, return_features: Literal[True]) -> FeaturesLogits:
        ...

    @overload
    def forward(self, x: Tensor, *, return_features: Literal[False] = ...) -> Tensor:
        ...

    @implements(nn.Module)
    def forward(self, x: Tensor, *, return_features: bool = False) -> Union[Tensor, FeaturesLogits]:
        return self.model.forward(x=x, return_features=return_features)  # type: ignore
