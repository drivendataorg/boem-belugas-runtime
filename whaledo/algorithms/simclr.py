from dataclasses import dataclass, field
from typing import Optional

from conduit.data.structures import BinarySample
from conduit.logging import init_logger
from conduit.models.utils import prefix_keys
from conduit.types import Stage
import pytorch_lightning as pl
from ranzen import implements
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import TypeAlias

from whaledo.algorithms.base import Algorithm
from whaledo.schedulers import CosineWarmup
from whaledo.transforms import MultiViewPair
from whaledo.utils import to_item

from .loss import supcon_loss
from .multicrop import MultiCropWrapper

__all__ = ["SimClr"]

LOGGER = init_logger(name=__file__)

TrainBatch: TypeAlias = BinarySample[MultiViewPair]


@dataclass(unsafe_hash=True)
class SimClr(Algorithm):

    proj_dim: int = 256
    mlp_head: bool = True
    dcl: bool = True

    temp_start: float = 1.0
    temp_end: float = 1.0
    temp_warmup_steps: int = 0
    temp: CosineWarmup = field(init=False)

    def __post_init__(self) -> None:
        if self.temp_start <= 0:
            raise AttributeError("'temp_start' must be positive.")
        if self.temp_end <= 0:
            raise AttributeError("'temp_end' must be positive.")
        if self.temp_warmup_steps < 0:
            raise AttributeError("'temp_warmup_steps' must be non-negative.")

        # initialise the encoders
        embed_dim = self.model.feature_dim
        head = nn.Sequential(nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, self.proj_dim))
        if self.mlp_head:
            head = nn.Sequential(
                nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, embed_dim), nn.ReLU(), head
            )
        self.student = MultiCropWrapper(backbone=self.model.backbone, head=head)
        self.temp = CosineWarmup(
            start_val=self.temp_start, end_val=self.temp_end, warmup_steps=self.temp_warmup_steps
        )

    @implements(Algorithm)
    def on_after_batch_transfer(
        self,
        batch: TrainBatch,
        dataloader_idx: Optional[int] = None,
    ) -> TrainBatch:
        if self.training:
            if isinstance(batch.x, MultiViewPair):
                batch.x.v1 = self._apply_batch_transforms(batch.x.v1)
                batch.x.v2 = self._apply_batch_transforms(batch.x.v2)
            else:
                raise ValueError(
                    "Inputs from the training data must be 'MultiCropOutput' or 'MultiViewPair'"
                    " objects."
                )
        return batch

    @implements(pl.LightningModule)
    def training_step(
        self,
        batch: TrainBatch,
        batch_idx: int,
    ) -> Tensor:
        logits_v1 = self.student.forward(batch.x.v1)
        logits_v1 = F.normalize(logits_v1, dim=1, p=2)

        logits_v2 = self.student.forward(batch.x.v2)
        logits_v2 = F.normalize(logits_v2, dim=1, p=2)

        logits = torch.cat((logits_v1, logits_v2), dim=0)
        temp = self.temp.val
        loss = supcon_loss(
            anchors=logits,
            anchor_labels=batch.y.repeat(2),
            temperature=temp,
            exclude_diagonal=True,
            dcl=self.dcl,
        )
        loss *= 2 * temp

        # Anneal the temperature parameter by one step.
        self.temp.step()

        logging_dict = {"supcon": to_item(loss)}
        logging_dict = prefix_keys(
            dict_=logging_dict,
            prefix=f"{str(Stage.fit)}/batch_loss",
            sep="/",
        )

        self.log_dict(logging_dict)

        return loss
