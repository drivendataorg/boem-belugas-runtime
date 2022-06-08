"""Whaledo data-module."""
from typing import Any, List

import attr
from conduit.data.constants import IMAGENET_STATS
from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import ImageTform
from conduit.data.structures import TrainValTestSplit
from pytorch_lightning import LightningDataModule
from ranzen import implements
import torchvision.transforms as T  # type: ignore

from whaledo.data.dataset import SampleType, WhaledoDataset
from whaledo.transforms import ResizeAndPadToSize

__all__ = ["WhaledoDataModule"]


@attr.define(kw_only=True)
class WhaledoDataModule(CdtVisionDataModule[WhaledoDataset, SampleType]):
    """Data-module for the 'Where's Whale-do' dataset."""

    @property
    def _default_train_transforms(self) -> ImageTform:
        transform_ls: List[ImageTform] = [
            ResizeAndPadToSize(224),
            T.ToTensor(),
            T.Normalize(*IMAGENET_STATS),
        ]
        return T.Compose(transform_ls)

    @property
    def _default_test_transforms(self) -> ImageTform:
        return self._default_train_transforms

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        WhaledoDataset(root=self.root)

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit[WhaledoDataset]:
        all_data = WhaledoDataset(root=self.root, transform=None)
        train, test = all_data.train_test_split(prop=self.test_prop, seed=self.seed)
        return TrainValTestSplit(train=train, val=test, test=test)
