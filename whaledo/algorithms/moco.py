from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional, TypeVar

from conduit.data.structures import BinarySample
from conduit.models.utils import prefix_keys
from conduit.types import MetricDict, Stage
import pytorch_lightning as pl
from ranzen import implements
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import TypeAlias

from whaledo.algorithms.base import Algorithm
from whaledo.algorithms.mean_teacher import MeanTeacher
from whaledo.algorithms.memory_bank import MemoryBank
from whaledo.schedulers import CosineWarmup
from whaledo.transforms import MultiCropOutput, MultiViewPair
from whaledo.utils import to_item

from .base import Algorithm
from .loss import moco_v2_loss, simclr_loss, supcon_loss
from .multicrop import MultiCropWrapper

__all__ = ["Moco"]

M = TypeVar("M", MultiCropOutput, MultiViewPair)
TrainBatch: TypeAlias = BinarySample[M]


class LossFn(Enum):
    ID = "instance_discrimination"
    SUPCON = "supcon"


@dataclass(unsafe_hash=True)
class Moco(Algorithm):
    IGNORE_INDEX: ClassVar[int] = -100

    proj_dim: int = 128
    mlp_head: bool = True

    mb_capacity: int = 16_384
    ema_decay_start: float = 0.99
    ema_decay_end: float = 0.99
    ema_warmup_steps: int = 0

    temp_start: float = 1.0
    temp_end: float = 1.0
    temp_warmup_steps: int = 0
    temp: CosineWarmup = field(init=False)

    cross_sample_only: bool = False
    loss_fn: LossFn = LossFn.SUPCON
    dcl: bool = True

    logit_mb: Optional[MemoryBank] = field(init=False)
    label_mb: Optional[MemoryBank] = field(init=False)

    def __post_init__(self) -> None:
        if self.temp_start <= 0:
            raise AttributeError("'temp_start' must be positive.")
        if self.temp_end <= 0:
            raise AttributeError("'temp_end' must be positive.")
        if self.temp_warmup_steps < 0:
            raise AttributeError("'temp_warmup_steps' must be non-negative.")

        if not (0 <= self.ema_decay_start <= 1):
            raise AttributeError("'ema_decay_start' must be in the range [0, 1].")
        if not (0 <= self.ema_decay_end <= 1):
            raise AttributeError("'ema_decay_end' must be in the range [0, 1].")
        if self.ema_warmup_steps < 0:
            raise AttributeError("'ema_warmup_steps' must be non-negative.")

        if self.mb_capacity < 0:
            raise AttributeError("'mb_capacity' must be non-negative.")

        # initialise the encoders
        embed_dim = self.model.feature_dim
        head = nn.Sequential(nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, self.proj_dim))
        if self.mlp_head:
            head = nn.Sequential(
                nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, embed_dim), nn.ReLU(), head
            )
        self.student = MultiCropWrapper(backbone=self.model.backbone, head=head)
        self.teacher = MeanTeacher(
            self.student,
            decay_start=self.ema_decay_start,
            decay_end=self.ema_decay_end,
            warmup_steps=self.ema_warmup_steps,
            auto_update=False,
        )

        # initialise the memory banks (if needed)
        self.logit_mb = None
        self.label_mb = None
        if self.mb_capacity > 0:
            self.logit_mb = MemoryBank.with_l2_hypersphere_init(
                dim=self.proj_dim, capacity=self.mb_capacity
            )
            if self.loss_fn is LossFn.SUPCON:
                self.label_mb = MemoryBank.with_constant_init(
                    dim=1, capacity=self.mb_capacity, value=self.IGNORE_INDEX, dtype=torch.long
                )
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
        logging_dict: MetricDict = {}
        # Compute the student's logits using both the global and local crops.
        inputs = batch.x
        student_logits = self.student.forward(inputs.anchor)
        student_logits = F.normalize(student_logits, dim=1, p=2)

        # Compute the teacher's logits using only the global crops.
        with torch.no_grad():
            self.teacher.update()
            teacher_logits = self.teacher.forward(inputs.target)
            teacher_logits = F.normalize(teacher_logits, dim=1, p=2)

        temp = self.temp.val
        if self.loss_fn is LossFn.SUPCON:
            candidates = teacher_logits
            candidate_labels = batch.y
            if (self.logit_mb is not None) and (self.label_mb is not None):
                lp_mask = (self.label_mb.memory != self.IGNORE_INDEX).squeeze(-1)
                labels_past = self.label_mb.clone(lp_mask).squeeze(-1)
                self.label_mb.push(batch.y)
                candidate_labels = torch.cat((candidate_labels, labels_past), dim=0)
                logits_past = self.logit_mb.clone(lp_mask)
                candidates = torch.cat((candidates, logits_past), dim=0)

            loss = supcon_loss(
                anchors=student_logits,
                anchor_labels=batch.y,
                candidates=candidates,
                candidate_labels=candidate_labels,
                temperature=temp,
                dcl=self.dcl,
                exclude_diagonal=self.cross_sample_only,
            )
        else:
            if self.logit_mb is None:
                loss = simclr_loss(
                    anchors=student_logits,
                    targets=teacher_logits,
                    temperature=temp,
                    dcl=self.dcl,
                )
            else:
                logits_past = self.logit_mb.clone()
                loss = moco_v2_loss(
                    anchors=student_logits,
                    positives=teacher_logits,
                    negatives=logits_past,
                    temperature=temp,
                    dcl=self.dcl,
                )
        loss *= 2 * temp

        if self.logit_mb is not None:
            self.logit_mb.push(teacher_logits)
        # Anneal the temperature parameter by one step.
        self.temp.step()

        logging_dict[self.loss_fn.value] = to_item(loss)
        logging_dict = prefix_keys(
            dict_=logging_dict,
            prefix=f"{str(Stage.fit)}/batch_loss",
            sep="/",
        )

        self.log_dict(logging_dict)

        return loss
