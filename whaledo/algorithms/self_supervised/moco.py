from dataclasses import dataclass, field
from typing import ClassVar, Optional, TypeVar, Union

from conduit.data.structures import BinarySample, NamedSample
from conduit.models.utils import prefix_keys
from conduit.types import Stage
import pytorch_lightning as pl
from ranzen import implements
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import TypeAlias

from whaledo.algorithms.base import Algorithm
from whaledo.algorithms.loss import default_supervised_loss
from whaledo.algorithms.mean_teacher import MeanTeacher
from whaledo.algorithms.memory_bank import MemoryBank
from whaledo.algorithms.self_supervised.multicrop import MultiCropWrapper
from whaledo.transforms import MultiCropOutput, MultiViewPair
from whaledo.types import PartiallyLabeledBatch
from whaledo.utils import to_item

from .base import SelfSupervisedAlgorithm
from .loss import moco_loss, scl_loss

__all__ = ["Moco"]

M = TypeVar("M", MultiCropOutput, MultiViewPair)
B = TypeVar(
    "B",
    bound=Union[
        PartiallyLabeledBatch[
            BinarySample[Union[MultiCropOutput, MultiViewPair, Tensor]],
            NamedSample[Union[MultiCropOutput, MultiViewPair, Tensor]],
        ],
        NamedSample[Union[MultiCropOutput, MultiViewPair, Tensor]],
    ],
)
TrainBatch: TypeAlias = PartiallyLabeledBatch[BinarySample[M], NamedSample[M]]


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

    logit_l_mb: MemoryBank = field(init=False)
    logit_u_mb: MemoryBank = field(init=False)
    label_mb: Optional[MemoryBank] = field(init=False)
    online_predictor: nn.Module = field(init=False)

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
        self.logit_l_mb = MemoryBank.with_l2_hypersphere_init(
            dim=self.out_dim, capacity=self.mb_capacity
        )
        self.logit_u_mb = MemoryBank.with_l2_hypersphere_init(
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
        batch: B,
        dataloader_idx: Optional[int] = None,
    ) -> B:
        if self.training:
            if isinstance(batch, NamedSample):
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
        student_features_logits = self.student.forward(inputs.anchor, return_features=True)
        student_logits = student_features_logits.logits
        student_logits = F.normalize(student_logits, dim=-1)

        # Compute the teacher's logits using only the global crops.
        with torch.no_grad():
            self.teacher.update()
            teacher_logits = self.teacher.forward(inputs.target)
            teacher_logits = F.normalize(teacher_logits, dim=1)

        logits_l_past = self.logit_l_mb.clone()
        logits_u_past = self.logit_u_mb.clone()
        logits_past = torch.cat((logits_l_past, logits_u_past), dim=0)

        loss = id_loss = moco_loss(
            anchors=student_logits,
            positives=teacher_logits,
            negatives=logits_past,
            temperature=self.temp_id,
            dcl=self.dcl,
        )

        student_logits_l = student_logits[: batch_l.x.num_sources]
        teacher_logits_l, teacher_logits_u = teacher_logits.tensor_split(
            [batch_l.x.num_sources], dim=0
        )

        self.logit_l_mb.push(teacher_logits_l)
        self.logit_u_mb.push(teacher_logits_u)

        cd_loss = None
        if self.label_mb is not None:
            labels_past = self.label_mb.clone()
            lp_mask = (labels_past != self.IGNORE_INDEX).squeeze(-1)
            if lp_mask.count_nonzero():
                cd_loss = scl_loss(
                    anchors=student_logits_l,
                    anchor_labels=batch_l.y,
                    candidates=logits_l_past[lp_mask],
                    candidate_labels=labels_past[lp_mask],
                    temperature=self.temp_cd,
                )
                loss += cd_loss
            self.label_mb.push(batch_l.y)

        # Prepare the inputs for the online-evaluation network.
        student_features = student_features_logits.features
        op_inputs = student_features[: batch_l.x.num_sources].view(-1, student_features.size(-1))
        if student_features.ndim == 3:
            op_targets = batch_l.y.repeat_interleave(op_inputs.size(0) // batch_l.y.size(0))
        else:
            op_targets = batch_l.y
        # Compute the loss for the online-evluation network.
        if self.op_stop_grad:
            op_inputs = op_inputs.detach()
        op_logits = self.online_predictor(op_inputs)
        op_loss = default_supervised_loss(input=op_logits, target=op_targets)
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
