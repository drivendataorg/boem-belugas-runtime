"""Whaledo data-module."""
from typing import Any

import albumentations as A  # type: ignore
import attr
from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.structures import TrainValTestSplit
from pytorch_lightning import LightningDataModule
from ranzen import implements

from whaledo.data.dataset import SampleType, WhaledoDataset

__all__ = ["WhaledoDataModule"]


@attr.define(kw_only=True)
class WhaledoDataModule(CdtVisionDataModule[WhaledoDataset, SampleType]):
    """Data-module for the 'Where's Whale-do' dataset."""

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        WhaledoDataset(root=self.root)

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit[WhaledoDataset]:
        # Split the data according to the pre-defined split indices
        # Split the data randomly according to test- and val-prop
        all_data = WhaledoDataset(root=self.root, transform=None)
        val, test, train = all_data.random_split(
            props=(self.val_prop, self.test_prop),
            seed=self.seed,
        )
        return TrainValTestSplit(train=train, val=val, test=test)
