from ranzen.torch.loss import cross_entropy_loss
from torch import Tensor
import torch.nn.functional as F

__all__ = ["default_supervised_loss"]


def default_supervised_loss(input: Tensor, *, target: Tensor) -> Tensor:
    if target.is_floating_point():
        return F.mse_loss(input, target)
    return cross_entropy_loss(input=input, target=target)
