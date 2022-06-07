from typing import Any, Optional

from conduit.data.structures import NamedSample
import pytorch_lightning as pl
from torch import Tensor
import torch.nn as nn

__all__ = ["DatasetEncoder"]


class DatasetEncoder(pl.LightningModule):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def predict_step(
        self, batch: NamedSample[Tensor], batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tensor:
        return self.forward(batch.x).to("cpu")

    def forward(self, x: Tensor) -> Any:
        return self.model.forward(x)
