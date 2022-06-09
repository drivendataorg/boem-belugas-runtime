from __future__ import annotations
from enum import Enum
from typing import Iterator, Sized, Union

from ranzen.decorators import implements
from ranzen.misc import str_to_enum
from ranzen.torch.data import BatchSamplerBase
from ranzen.torch.sampling import batched_randint
import torch
from torch import Tensor

__all__ = ["QueryKeySampler"]


class BaseSampler(Enum):
    WEIGHTED = "weighted"
    RANDOM = "random"


class QueryKeySampler(BatchSamplerBase):
    def __init__(
        self,
        data_source: Sized,
        *,
        batch_size: int,
        ids: Tensor,
        generator: torch.Generator | None = None,
        base_sampler: Union[BaseSampler, str] = BaseSampler.WEIGHTED,
    ) -> None:
        self.base_sampler = str_to_enum(base_sampler, enum=BaseSampler)
        self.data_source = data_source
        self.batch_size = batch_size
        self.generator = generator

        can_sample = ids[:, None] == ids
        counts = can_sample.count_nonzero(dim=1) - 1
        # For samples which are unique in their ID we simply abandon the cross-sample constraint
        # and allow any other samples to be paird with them in order to complete the batch.
        no_pairs = counts == 0
        # Prevent self-matches.
        can_sample.fill_diagonal_(False)
        can_sample[no_pairs] = True
        counts[no_pairs] = len(no_pairs) - 1
        self.counts = counts
        self.can_sample = can_sample

        if self.base_sampler == "weighted":
            _, inverse, counts = ids.unique(return_counts=True, return_inverse=True)
            id_weights = 1 - (counts / len(inverse))
            self.sample_weights = id_weights[inverse]
        else:
            self.sample_weights = None

        super().__init__(epoch_length=None)

    def _add_key_idxs(self, query_idxs: Tensor) -> Tensor:
        # Sample the keys via intra-class sampling
        rel_key_idxs = batched_randint(self.counts[query_idxs])
        # 2) Convert the row-wise indices into row-major indices, considering
        # only the postive entries in the rows.
        offsets = self.counts[query_idxs].cumsum(dim=0)[:-1]
        rel_key_idxs[1:] += offsets
        # 3) Finally, map from group-relative indices to absolute ones.
        _, abs_pos_idxs = self.can_sample[query_idxs].nonzero(as_tuple=True)
        key_idxs = abs_pos_idxs[rel_key_idxs]
        return torch.cat((query_idxs, key_idxs), dim=0)

    @implements(BatchSamplerBase)
    def __iter__(self) -> Iterator[list[int]]:
        while True:
            if self.sample_weights is None:
                batch_idxs = torch.randint(
                    low=0,
                    high=len(self.data_source),
                    generator=self.generator,
                    size=(self.batch_size,),
                )
            else:
                batch_idxs = torch.multinomial(
                    self.sample_weights,
                    num_samples=self.batch_size,
                    replacement=True,
                    generator=self.generator,
                )
            batch_idxs = self._add_key_idxs(batch_idxs)
            yield batch_idxs.tolist()
