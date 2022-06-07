from dataclasses import dataclass

from conduit.logging import init_logger
import pytorch_lightning as pl
from ranzen import implements
from typing_extensions import Self

from src.algorithms.base import Algorithm
from src.data.datamodules.base import WILDSDataModule

__all__ = ["SelfSupervisedAlgorithm"]

LOGGER = init_logger(name=__file__)


@dataclass(unsafe_hash=True)
class SelfSupervisedAlgorithm(Algorithm):
    @implements(Algorithm)
    def run(self, datamodule: WILDSDataModule, *, trainer: pl.Trainer, test: bool = True) -> Self:
        if not datamodule.use_unlabeled:
            LOGGER.info(
                f"'use_unlabeled' must be 'True' when running with self-supervised algorithms."
                " Now resetting the data-module again with 'use_unlabeled=True' forced."
            )
            datamodule.use_unlabeled = True
            datamodule.setup(force_reset=True)
        return super().run(datamodule=datamodule, trainer=trainer, test=test)
