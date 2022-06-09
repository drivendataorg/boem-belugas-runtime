from dataclasses import dataclass
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
from whaledo.transforms import MultiViewPair
from whaledo.utils import to_item

from .loss import supcon_loss
from .multicrop import MultiCropWrapper

__all__ = ["SimClr"]

LOGGER = init_logger(name=__file__)

TrainBatch: TypeAlias = BinarySample[MultiViewPair]


@dataclass(unsafe_hash=True)
class SimClr(Algorithm):

    out_dim: int = 256
    mlp_head: bool = True
    temp: float = 1.0
    temp_cd: float = 1.0

    def __post_init__(self) -> None:
        if self.temp <= 0:
            raise AttributeError("'temp' must be positive.")

        # initialise the encoders
        embed_dim = self.model.feature_dim
        head = nn.Linear(embed_dim, self.out_dim)
        if self.mlp_head:
            head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), head)
        self.student = MultiCropWrapper(backbone=self.model.backbone, head=head)

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
        loss = supcon_loss(
            anchors=logits,
            anchor_labels=batch.y.repeat(2),
            temperature=self.temp,
            exclude_diagonal=True,
        )

        logging_dict = {"supcon": to_item(loss)}
        logging_dict = prefix_keys(
            dict_=logging_dict,
            prefix=f"{str(Stage.fit)}/batch_loss",
            sep="/",
        )

        self.log_dict(logging_dict)

        return loss
