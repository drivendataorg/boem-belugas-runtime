from dataclasses import dataclass
from typing import Optional

from conduit.data.structures import BinarySample
from conduit.models.utils import prefix_keys
from conduit.types import Loss, Stage
from ranzen.decorators import implements
from ranzen.torch.loss import cross_entropy_loss
from torch import Tensor
from typing_extensions import TypeAlias

from whaledo.algorithms.base import Algorithm
from whaledo.utils import to_item

__all__ = ["Erm"]

TrainBatch: TypeAlias = BinarySample


@dataclass(unsafe_hash=True)
class Erm(Algorithm):
    loss_fn: Optional[Loss] = None

    def _compute_loss(self, logits: Tensor, *, batch: BinarySample[Tensor]) -> Tensor:
        if self.loss_fn is None:
            return cross_entropy_loss(input=logits, target=batch.y)
        return self.loss_fn(input=logits, target=batch.y)

    @implements(Algorithm)
    def training_step(self, batch: TrainBatch, batch_idx: int) -> Tensor:
        logits = self.forward(batch.x)
        loss = self._compute_loss(logits=logits, batch=batch)

        results_dict = {"batch_loss": to_item(loss)}
        results_dict = prefix_keys(dict_=results_dict, prefix=str(Stage.fit), sep="/")
        self.log_dict(results_dict)

        return loss
