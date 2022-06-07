from typing import Optional

import torch.nn as nn

from src.models.base import ModelFactoryOut, PredictorFactory

__all__ = ["Fcn"]


class Fcn(PredictorFactory):
    def __call__(
        self, in_dim: int, *, out_dim: int, hidden_dim: Optional[int] = None, num_hidden: int = 0
    ) -> ModelFactoryOut:
        predictor = nn.Sequential(nn.Flatten(), nn.LayerNorm(in_dim))
        if num_hidden > 0:
            if hidden_dim is None:
                hidden_dim = in_dim
            for _ in range(num_hidden):
                predictor.append(nn.Linear(in_dim, hidden_dim))
                predictor.append(nn.LayerNorm(in_dim))

        predictor.append(nn.Linear(in_dim, out_dim))

        return predictor, out_dim
