from pathlib import Path
from typing import Optional, TypeAlias, Union, cast

from conduit.data import CdtVisionDataset
from conduit.data.datasets.utils import ImageTform
from conduit.data.structures import BinarySample
import pandas as pd  # type: ignore
import torch
from torch import Tensor

__all__ = ["WhaledoDataset"]

SampleType: TypeAlias = BinarySample


class WhaledoDataset(CdtVisionDataset[BinarySample, Tensor, None]):
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
        y = torch.as_tensor(self.metadata["image_id"].factorize()[0].to_numpy(), dtype=torch.long)
        x = self.metadata["path"].to_numpy()
        super().__init__(x=x, y=y, image_dir=self.root)
