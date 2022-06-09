from dataclasses import dataclass
import logging
from typing import Any, Optional, Protocol, Tuple, TypeVar

from conduit.logging import init_logger
from ranzen.decorators import implements
from src.data.datamodules.base import WILDSDataModule  # type: ignore
from torch import Tensor
import torch.nn as nn
from typing_extensions import Final, Self, TypeAlias

from whaledo.types import Prediction

__all__ = [
    "BackboneFactory",
    "Model",
]

# Number of positive predictions for a given sample is capped at 20.
K_MAX: Final[int] = 20
M = TypeVar("M", bound=nn.Module)
ModelFactoryOut: TypeAlias = Tuple[M, int]


class BackboneFactory(Protocol):
    def __call__(self) -> ModelFactoryOut:
        ...


@dataclass(unsafe_hash=True)
class Model(nn.Module):
    backbone: nn.Module
    feature_dim: int
    threshold: float = 0

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

    def threshold_scores(self, scores: Tensor) -> Tensor:
        return scores > self.threshold

    def predict(self, queries: Tensor, *, db: Optional[Tensor] = None) -> Prediction:
        mask_diag = False
        if db is None:
            db = queries
            mask_diag = True

        sim_mat = queries @ db.T
        db_size = sim_mat.size(1)
        if mask_diag:
            # Mask the diagonal to prevent self matches.
            sim_mat.fill_diagonal_(0)
            db_size -= 1

        k = min(K_MAX, db_size)
        probs = sim_mat.softmax(dim=1)
        scores, topk_inds = probs.topk(dim=1, largest=True, k=k)
        mask = self.threshold_scores(scores=scores)
        mask_inds = mask.nonzero(as_tuple=True)
        scores, retrieved_inds = scores[mask_inds], topk_inds[mask_inds]

        return Prediction(
            query_inds=mask_inds[0],
            retrieved_inds=retrieved_inds,
            scores=scores,
        )
