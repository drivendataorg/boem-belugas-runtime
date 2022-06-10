from dataclasses import dataclass
from typing import Dict, List, Literal, TypeVar, Union

from conduit.data.structures import InputContainer
from conduit.types import Addable
from ranzen.decorators import implements
from ranzen.misc import gcopy
import torch
from torch import Tensor
from typing_extensions import Self, TypeAlias

__all__ = [
    "AddableDict",
    "EvalOutputs",
    "EvalEpochOutput",
    "Prediction",
]


@dataclass(unsafe_hash=True)
class EvalOutputs(InputContainer[Tensor]):
    logits: Tensor
    ids: Tensor

    @implements(InputContainer)
    def __len__(self) -> int:
        return len(self.logits)

    @implements(InputContainer)
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
                    copy[key_o] = value_s + value_o  # type: ignore
                else:
                    copy[key_o] = [value_s, value_o]
            else:
                copy[key_o] = value_o
        return copy


EvalStepOutput: TypeAlias = EvalOutputs
EvalEpochOutput: TypeAlias = List[EvalStepOutput]


@dataclass(unsafe_hash=True)
class Prediction:
    query_inds: Tensor
    retrieved_inds: Tensor
    n_retrieved_per_query: Tensor
    scores: Tensor

    def __post_init__(self) -> None:
        if len(self.query_inds) != len(self.retrieved_inds) != len(self.scores):
            raise AttributeError(
                "'query_inds', 'retrieved_inds', and 'scores' must be equal in length."
            )
