from typing import Optional, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
from torch import Tensor
from torch.types import Number

__all__ = [
    "to_item",
    "to_numpy",
]


DT = TypeVar("DT", bound=Union[np.number, np.bool_])


@overload
def to_numpy(tensor: Tensor, *, dtype: DT) -> npt.NDArray[DT]:
    ...


@overload
def to_numpy(tensor: Tensor, *, dtype: None = ...) -> npt.NDArray:
    ...


def to_numpy(tensor: Tensor, *, dtype: Optional[DT] = None) -> Union[npt.NDArray[DT], npt.NDArray]:
    arr = tensor.detach().cpu().numpy()
    if dtype is not None:
        arr.astype(dtype)
    return arr


def to_item(tensor: Tensor) -> Number:
    return tensor.detach().cpu().item()
