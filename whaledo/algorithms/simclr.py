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
from whaledo.transforms import MultiViewPair
from whaledo.utils import to_item

from .loss import DecoupledContrastiveLoss, supcon_loss
from .multicrop import MultiCropWrapper

__all__ = ["SimClr"]

LOGGER = init_logger(name=__file__)

TrainBatch: TypeAlias = BinarySample[MultiViewPair]


@dataclass(unsafe_hash=True)
class SimClr(Algorithm):

    out_dim: int = 128
    mlp_head: bool = True

    temp_id: float = 1.0
    sigma: float = 0.5
    symmetrize_loss: bool = True
    vmf_weighting: bool = False
    temp_cd: float = 1.0
    scl_weight: float = 0.0
    op_stop_grad: bool = True

    online_predictor: nn.Module = field(init=False)
    loss_fn: DecoupledContrastiveLoss = field(init=False)

    def __post_init__(self) -> None:
        if self.temp_id <= 0:
            raise AttributeError("'temp_id' must be positive.")
        if self.temp_cd <= 0:
            raise AttributeError("'temp_cd' must be positive.")

        # initialise the encoders
        embed_dim = self.model.feature_dim
        head = nn.Linear(embed_dim, self.out_dim)
        if self.mlp_head:
            head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), head)
        self.student = MultiCropWrapper(backbone=self.model.backbone, head=head)
        if self.vmf_weighting:
            self.loss_fn = DecoupledContrastiveLoss.with_vmf_weighting(
                temperature=self.temp_id, sigma=self.sigma
            )
        else:
            self.loss_fn = DecoupledContrastiveLoss(temperature=self.temp_id)

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
        inputs = batch.x
        logits_v1 = self.student.forward(inputs.v1)
        logits_v1 = F.normalize(logits_v1, dim=1, p=2)

        logits_v2 = self.student.forward(inputs.v2)
        logits_v2 = F.normalize(logits_v2, dim=1, p=2)

        id_loss = self.loss_fn(logits_v1, logits_v2)
        if self.symmetrize_loss:
            id_loss += self.loss_fn(logits_v2, logits_v1)
        loss = id_loss

        cd_loss = None
        if self.scl_weight > 0:
            logits = torch.cat((logits_v1, logits_v2), dim=0)
            cd_loss = supcon_loss(
                anchors=logits,
                anchor_labels=batch.y,
                temperature=self.temp_cd,
            )
            loss += cd_loss

        logging_dict = {
            "instance_discrimination": to_item(id_loss),
            "total": to_item(loss),
        }
        if cd_loss is not None:
            logging_dict["class_discrimination"] = to_item(cd_loss)
        logging_dict = prefix_keys(
            dict_=logging_dict,
            prefix=f"{str(Stage.fit)}/batch_loss",
            sep="/",
        )

        self.log_dict(logging_dict)

        return loss
