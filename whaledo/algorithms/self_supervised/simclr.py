from dataclasses import dataclass, field
from typing import ClassVar, Optional, TypeVar, Union

from conduit.data.structures import BinarySample, NamedSample
from conduit.logging import init_logger
from conduit.models.utils import prefix_keys
from conduit.types import Stage
import pytorch_lightning as pl
from ranzen import implements
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import Self, TypeAlias

from src.algorithms.base import Algorithm
from src.algorithms.erm import default_supervised_loss
from src.algorithms.self_supervised.multicrop import MultiCropWrapper
from src.data.datamodules.base import WILDSDataModule
from src.transforms import MultiViewPair
from src.types import PartiallyLabeledBatch
from src.utils import to_item

from .base import SelfSupervisedAlgorithm
from .loss import DecoupledContrastiveLoss, scl_loss

__all__ = ["SimCLR"]

LOGGER = init_logger(name=__file__)

B = TypeVar(
    "B",
    bound=Union[
        PartiallyLabeledBatch[
            BinarySample[Union[MultiViewPair, Tensor]],
            NamedSample[Union[MultiViewPair, Tensor]],
        ],
        NamedSample[Union[MultiViewPair, Tensor]],
    ],
)
TrainBatch: TypeAlias = PartiallyLabeledBatch[
    BinarySample[MultiViewPair], NamedSample[MultiViewPair]
]


@dataclass(unsafe_hash=True)
class SimCLR(SelfSupervisedAlgorithm):
    IGNORE_INDEX: ClassVar[int] = -100

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
            head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), head)  # type: ignore
        self.student = MultiCropWrapper(backbone=self.model.backbone, head=head)
        self.online_predictor = nn.Sequential(
            nn.Flatten(), nn.Linear(embed_dim, self.model.out_dim)
        )
        if self.vmf_weighting:
            self.loss_fn = DecoupledContrastiveLoss.with_vmf_weighting(
                temperature=self.temp_id, sigma=self.sigma
            )
        else:
            self.loss_fn = DecoupledContrastiveLoss(temperature=self.temp_id)

    @implements(Algorithm)
    def on_after_batch_transfer(
        self,
        batch: B,
        dataloader_idx: Optional[int] = None,
    ) -> B:
        if self.training:
            if isinstance(batch, NamedSample):
                if isinstance(batch.x, MultiViewPair):
                    batch.x.v1 = self._apply_batch_transforms(batch.x.v1)
                    batch.x.v2 = self._apply_batch_transforms(batch.x.v2)
                else:
                    raise ValueError(
                        "Inputs from the training data must be 'MultiCropOutput' or 'MultiViewPair'"
                        " objects."
                    )
            else:
                batch["labeled"] = self.on_after_batch_transfer(batch["labeled"])
                batch["unlabeled"] = self.on_after_batch_transfer(batch["unlabeled"])
        return batch

    @implements(pl.LightningModule)
    def training_step(
        self,
        batch: TrainBatch,
        batch_idx: int,
    ) -> Tensor:
        batch_l = batch["labeled"]
        batch_u = batch["unlabeled"]
        # Compute the student's logits using both the global and local crops.
        inputs = batch_l.x + batch_u.x
        features_logits_v1 = self.student.forward(inputs.v1, return_features=True)
        features_logits_v2 = self.student.forward(inputs.v2, return_features=True)
        logits_v1 = F.normalize(features_logits_v1.logits)
        logits_v2 = F.normalize(features_logits_v2.logits)

        id_loss = self.loss_fn(logits_v1, logits_v2)
        if self.symmetrize_loss:
            id_loss += self.loss_fn(logits_v2, logits_v1)
        loss = id_loss

        cd_loss = None
        if self.scl_weight > 0:
            logits_l = torch.cat(
                (logits_v1[: batch_l.x.num_sources], logits_v2[: batch_l.x.num_sources]),
                dim=0,
            )
            cd_loss = scl_loss(
                anchors=logits_l,
                anchor_labels=batch_l.y,
                temperature=self.temp_cd,
            )
            loss += cd_loss

        # Prepare the inputs for the online-evaluation network.
        op_inputs = torch.cat(
            (
                features_logits_v1.features[: batch_l.x.num_sources],
                features_logits_v2.features[: batch_l.x.num_sources],
            ),
            dim=0,
        )
        op_targets = batch_l.y.repeat(2)
        # Compute the loss for the online-evluation network.
        if self.op_stop_grad:
            op_inputs.detach_()
        op_logits = self.online_predictor(op_inputs)
        op_loss = default_supervised_loss(logits=op_logits, targets=op_targets)
        loss += op_loss

        logging_dict = {
            "instance_discrimination": to_item(id_loss),
            "online_predictor": to_item(op_loss),
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

    @implements(Algorithm)
    def forward(self, x: Tensor) -> Tensor:
        return self.online_predictor(self.student.backbone(x))

    @implements(Algorithm)
    def run(self, datamodule: WILDSDataModule, *, trainer: pl.Trainer, test: bool = True) -> Self:
        if not datamodule.use_unlabeled:
            LOGGER.info(
                f"'use_unlabeled' must be 'True' when running with self-supervised algorithms."
                " Now resetting the data-module again with 'use_unlabeled=True' forced."
            )
            datamodule.use_unlabeled = True
            datamodule.setup(force_reset=True)
        return super().run(datamodule=datamodule, trainer=trainer, test=test)
