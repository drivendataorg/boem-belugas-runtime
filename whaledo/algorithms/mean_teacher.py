from typing import Any, Generic, TypeVar

from ranzen.decorators import implements
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel

from whaledo.schedulers import LinearWarmup

__all__ = ["MeanTeacher"]


M = TypeVar("M", bound=nn.Module)


class MeanTeacher(nn.Module, Generic[M]):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(
        self,
        model: M,
        *,
        decay_start: float,
        decay_end: float,
        warmup_steps: int,
        update_frequency: int = 1,
        auto_update: bool = True,
    ) -> None:
        super().__init__()
        self.decay = LinearWarmup(
            start_val=decay_start, end_val=decay_end, warmup_steps=warmup_steps
        )
        self.update_frequency = update_frequency
        self._training_iteration = 0
        self.auto_update = auto_update
        self.model = model
        self.ema_model = AveragedModel(self.model, avg_fn=self._ema_update)

    @torch.no_grad()
    def _ema_update(
        self,
        avg_model_param: Tensor,
        model_param: Tensor,
        num_averaged: int,
    ) -> Tensor:
        """
        Perform an EMA update of the model's parameters.
        """
        return self.decay.val * avg_model_param + (1.0 - self.decay.val) * model_param

    @torch.no_grad()
    def update(self) -> None:
        if (self._training_iteration % self.update_frequency) == 0:
            self.ema_model.update_parameters(self.model)
            self.decay.step()

    @torch.no_grad()
    @implements(nn.Module)
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.auto_update:
            self.update()
        return self.ema_model(*args, **kwargs)
