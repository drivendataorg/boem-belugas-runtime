from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.optim.swa_utils import AveragedModel

from whaledo.models.meta.base import MetaModel

__all__ = ["EmaModel"]


@dataclass(unsafe_hash=True)
class EmaModel(MetaModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    decay: float
    update_frequency: int = 1
    auto_update = True
    _training_iteration: int = field(init=False)
    ema_model: AveragedModel = field(init=False)

    def __post_init__(self) -> None:
        self._training_iteration = 0
        self.ema_model = AveragedModel(self.model, avg_fn=self._ema_update)
        super().__post_init__()

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
        return self.decay * avg_model_param + (1 - self.decay) * model_param

    @torch.no_grad()
    def update(self) -> None:
        if (self._training_iteration % self.update_frequency) == 0:
            self.ema_model.update_parameters(self.model)

    @torch.no_grad()
    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        if self.training:
            if self.auto_update:
                self.update()
            return self.model(x, **kwargs)
        return self.ema_model(x, **kwargs)
