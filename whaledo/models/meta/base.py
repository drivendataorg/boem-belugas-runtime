from dataclasses import dataclass, field
from typing import Any, Optional, Union

from ranzen.decorators import implements
from torch import Tensor
import torch.nn as nn
from typing_extensions import ParamSpec, Self

from whaledo.models.base import Model
from whaledo.types import Prediction

__all__ = ["MetaModel"]


P = ParamSpec("P")


@dataclass(unsafe_hash=True)
class MetaModel(nn.Module):
    model: Union[Model, "MetaModel"]
    # Expose the base model's attributes.
    backbone: nn.Module = field(init=False)
    feature_dim: int = field(init=False)

    def __post_init__(self) -> None:
        # Expose the backbone/predictor attributes.
        self.backbone = self.model.backbone
        self.feature_dim = self.model.feature_dim

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        return obj

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x=x)

    def predict(
        self, queries: Tensor, *, db: Optional[Tensor] = None, k: int = 20, sorted: bool = True
    ) -> Prediction:
        return self.model.predict(queries=queries, db=db, k=k, sorted=sorted)
