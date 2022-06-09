from pathlib import Path
from typing import Optional, Union, cast

from conduit.data import CdtVisionDataset
from conduit.data.datasets.utils import ImageTform
from conduit.data.structures import BinarySample, TrainTestSplit
import pandas as pd  # type: ignore
import torch
from torch import Tensor
from typing_extensions import Literal, Self, TypeAlias

__all__ = ["WhaledoDataset"]

SampleType: TypeAlias = BinarySample


class WhaledoDataset(CdtVisionDataset[BinarySample[Tensor], Tensor, Tensor]):
    """
    Pytorch dataset for the
    `Where's Whale-do? competition <https://www.drivendata.org/competitions/96/beluga-whales/>`_
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        transform: Optional[ImageTform] = None,
    ) -> None:
        self.root = Path(root)
        self.metadata = cast(pd.DataFrame, pd.read_csv(self.root / "metadata.csv"))

        x = self.metadata["path"].to_numpy()
        y = torch.as_tensor(self.metadata["whale_id"].factorize()[0], dtype=torch.long)

        vps = torch.as_tensor(self.metadata["viewpoint"].factorize()[0], dtype=torch.long)
        years = torch.as_tensor(
            self.metadata["date"].str.split("-", expand=True)[0].factorize()[0], dtype=torch.long
        )
        s = torch.stack([vps, years], dim=-1)

        super().__init__(x=x, y=y, s=s, image_dir=self.root, transform=transform)

    def train_test_split(
        self,
        prop: float,
        seed: Optional[int] = None,
        side: Literal["left", "right"] = "right",
    ) -> TrainTestSplit[Self]:
        if not (0.0 < prop < 1.0):
            raise ValueError("'prop` must be in the range (0, 1)].")
        generator = torch.default_generator if seed is None else torch.Generator().manual_seed(seed)
        ids_u, counts = self.y.unique(
            return_counts=True,
        )
        mask = counts > 1
        ids_u = cast(Tensor, ids_u[mask])
        shuffle_inds = torch.randperm(len(ids_u), generator=generator)
        shuffled = ids_u[shuffle_inds]
        counts = cast(Tensor, counts[mask][shuffle_inds])
        cutoff = round(prop * len(self.y))
        split_ind = int(torch.searchsorted(counts.cumsum(dim=0), cutoff, side=side))
        test_ids, train_ids = shuffled.tensor_split([split_ind], dim=0)

        test_indices = (
            (self.y.unsqueeze(1) == test_ids.unsqueeze(0)).nonzero(as_tuple=True)[0].tolist()
        )
        train_indices = list(set(torch.arange(len(self.y)).tolist()) - set(test_indices))

        return TrainTestSplit(
            train=self.subset(indices=train_indices), test=self.subset(indices=test_indices)
        )
