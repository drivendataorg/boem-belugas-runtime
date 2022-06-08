from dataclasses import dataclass
from typing import Dict, List, Literal, TypeVar, Union

from conduit.data.structures import InputContainer
from conduit.types import Addable
from ranzen.misc import gcopy
import torch
from torch import Tensor
from typing_extensions import Self, TypeAlias

__all__ = [
    "AddableDict",
    "EvalOutputs",
    "EvalEpochOutput",
]


@dataclass
class EvalOutputs(InputContainer):
    logits: Tensor
    ids: Tensor

    def __len__(self) -> int:
        return len(self.logits)

    def __add__(self, other: Union[Self, Literal[0]]) -> Self:
        if other == 0:
            return self
        copy = gcopy(self, deep=False)
        copy.logits = torch.cat((copy.logits, other.logits))
        copy.ids = torch.cat((copy.ids, other.ids))
        return copy


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class AddableDict(dict[_KT, _VT], Addable):
    def __add__(self: Self, other: Self) -> Dict[_KT, Union[_VT, List[_VT]]]:
        copy: AddableDict[_KT, Union[_VT, List[_VT]]] = AddableDict()
        copy |= gcopy(self, deep=False)
        for key_o, value_o in other.items():
            if key_o in self:
                value_s = self[key_o]
                if isinstance(value_s, Addable) and isinstance(value_o, Addable):
                    copy[key_o] = value_s + value_o
                else:
                    copy[key_o] = [value_s, value_o]
            else:
                copy[key_o] = value_o
        return copy


EvalStepOutput: TypeAlias = EvalOutputs
EvalEpochOutput: TypeAlias = List[EvalStepOutput]
