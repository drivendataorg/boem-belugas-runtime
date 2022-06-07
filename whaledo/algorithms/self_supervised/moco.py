from dataclasses import dataclass, field
from typing import ClassVar, Optional, TypeVar

from conduit.data.structures import BinarySample
from conduit.models.utils import prefix_keys
from conduit.types import Stage
import pytorch_lightning as pl
from ranzen import implements
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import TypeAlias

from whaledo.algorithms.base import Algorithm
from whaledo.algorithms.mean_teacher import MeanTeacher
from whaledo.algorithms.memory_bank import MemoryBank
from whaledo.algorithms.self_supervised.multicrop import MultiCropWrapper
from whaledo.transforms import MultiCropOutput, MultiViewPair
from whaledo.utils import to_item

from .base import SelfSupervisedAlgorithm
from .loss import moco_loss, scl_loss

__all__ = ["Moco"]

M = TypeVar("M", MultiCropOutput, MultiViewPair)
TrainBatch: TypeAlias = BinarySample[M]


@dataclass(unsafe_hash=True)
class Moco(SelfSupervisedAlgorithm):
    IGNORE_INDEX: ClassVar[int] = -100

    out_dim: int = 128
    mlp_head: bool = True

    mb_capacity: int = 16_384
    ema_decay_start: float = 0.9
    ema_decay_end: float = 0.999
    ema_warmup_steps: int = 0

    temp_id: float = 1.0
    dcl: bool = True
    temp_cd: float = 0.1
    scl_weight: float = 1.0
    op_stop_grad: bool = True

    logit_mb: MemoryBank = field(init=False)
    label_mb: Optional[MemoryBank] = field(init=False)

    def __post_init__(self) -> None:
        if self.temp_id <= 0:
            raise AttributeError("'temp_id' must be positive.")
        if self.temp_cd <= 0:
            raise AttributeError("'temp_cd' must be positive.")
        if self.scl_weight < 0:
            raise AttributeError("'loss_s_weight' must be non-negative.")
        if not (0 <= self.ema_decay_start < 1):
            raise AttributeError("'ema_decay_start' must be in the range [0, 1).")
        if not (0 <= self.ema_decay_end < 1):
            raise AttributeError("'ema_decay_end' must be in the range [0, 1).")
        if self.ema_warmup_steps < 0:
            raise AttributeError("'ema_warmup_steps' must be non-negative.")

        # initialise the encoders
        embed_dim = self.model.feature_dim
        head = nn.Linear(embed_dim, self.out_dim)
        if self.mlp_head:
            head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), head)  # type: ignore
        self.student = MultiCropWrapper(backbone=self.model.backbone, head=head)
        self.teacher = MeanTeacher(
            self.student,
            decay_start=self.ema_decay_start,
            decay_end=self.ema_decay_end,
            warmup_steps=self.ema_warmup_steps,
            auto_update=False,
        )
        self.online_predictor = nn.Sequential(
            nn.Flatten(), nn.Linear(embed_dim, self.model.out_dim)
        )

        # initialise the memory banks
        self.logit_mb = MemoryBank.with_l2_hypersphere_init(
            dim=self.out_dim, capacity=self.mb_capacity
        )
        if self.scl_weight > 0:
            self.label_mb = MemoryBank.with_constant_init(
                dim=1, capacity=self.mb_capacity, value=self.IGNORE_INDEX, dtype=torch.long
            )
        else:
            self.label_mb = None

    @implements(Algorithm)
    def on_after_batch_transfer(
        self,
        batch: TrainBatch,
        dataloader_idx: Optional[int] = None,
    ) -> TrainBatch:
        if self.training:
            if isinstance(batch.x, MultiCropOutput):
                batch.x.global_views.v1 = self._apply_batch_transforms(batch.x.global_views.v1)
                batch.x.global_views.v2 = self._apply_batch_transforms(batch.x.global_views.v2)
                batch.x.local_views = self._apply_batch_transforms(batch.x.local_views)
            elif isinstance(batch.x, MultiViewPair):
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
        # Compute the student's logits using both the global and local crops.
        inputs = batch.x
        student_features_logits = self.student.forward(inputs.anchor, return_features=True)
        student_logits = student_features_logits.logits
        student_logits = F.normalize(student_logits, dim=-1)

        # Compute the teacher's logits using only the global crops.
        with torch.no_grad():
            self.teacher.update()
            teacher_logits = self.teacher.forward(inputs.target)
            teacher_logits = F.normalize(teacher_logits, dim=1)

        logits_past = self.logit_mb.clone()

        loss = id_loss = moco_loss(
            anchors=student_logits,
            positives=teacher_logits,
            negatives=logits_past,
            temperature=self.temp_id,
            dcl=self.dcl,
        )

        self.logit_mb.push(teacher_logits)

        cd_loss = None
        if self.label_mb is not None:
            labels_past = self.label_mb.clone()
            lp_mask = (labels_past != self.IGNORE_INDEX).squeeze(-1)
            if lp_mask.count_nonzero():
                cd_loss = scl_loss(
                    anchors=student_logits,
                    anchor_labels=batch.y,
                    candidates=logits_past[lp_mask],
                    candidate_labels=labels_past[lp_mask],
                    temperature=self.temp_cd,
                )
                loss += cd_loss
            self.label_mb.push(batch.y)

        # Prepare the inputs for the online-evaluation network.
        student_features = student_features_logits.features

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

    @implements(Algorithm)
    def forward(self, x: Tensor) -> Tensor:
        return self.online_predictor(self.student.backbone(x))
