from abc import abstractmethod
from dataclasses import dataclass, field
from functools import reduce
import operator
from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union

from conduit.data.structures import BinarySample, NamedSample
from conduit.models.utils import prefix_keys
from conduit.types import LRScheduler, MetricDict, Stage
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from ranzen import implements
from ranzen.torch.data import TrainingMode
import torch
from torch import Tensor, optim
import torch.nn as nn
from torchmetrics.retrieval.average_precision import RetrievalMAP
from typing_extensions import Self

from whaledo.models import MetaModel, Model
from whaledo.transforms import BatchTransform
from whaledo.types import EvalEpochOutput, EvalOutputs, EvalStepOutput

__all__ = ["Algorithm"]

T = TypeVar("T", bound=Union[Tensor, NamedSample[Tensor]])


@dataclass(unsafe_hash=True)
class Algorithm(pl.LightningModule):
    model: Union[Model, MetaModel]
    lr: float = 5.0e-5
    optimizer_cls: str = "torch.optim.AdamW"
    optimizer_kwargs: Optional[DictConfig] = None
    use_sam: bool = False
    sam_rho: float = 0.05
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[DictConfig] = None
    lr_sched_interval: TrainingMode = TrainingMode.step
    lr_sched_freq: int = 1
    batch_transforms: Optional[List[BatchTransform]] = None
    test_on_best: bool = False
    rmap: RetrievalMAP = field(default=RetrievalMAP())

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        pl.LightningModule.__init__(obj)
        return obj

    def _apply_batch_transforms(self, batch: T) -> T:
        if self.batch_transforms is not None:
            for tform in self.batch_transforms:
                if isinstance(batch, Tensor):
                    batch = tform(inputs=batch, targets=None)  # type: ignore
                else:
                    if isinstance(batch, BinarySample):
                        transformed_x, transformed_y = tform(inputs=batch.x, targets=batch.y)
                        batch.y = transformed_y
                    else:
                        transformed_x = tform(inputs=batch.x, targets=None)
                    batch.x = transformed_x
        return batch

    @implements(pl.LightningModule)
    def on_after_batch_transfer(
        self,
        batch: T,
        dataloader_idx: Optional[int] = None,
    ) -> T:
        if self.training:
            batch = self._apply_batch_transforms(batch)
        return batch

    @abstractmethod
    def training_step(
        self,
        batch: BinarySample,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        ...

    @torch.no_grad()
    def inference_step(self, batch: BinarySample) -> EvalOutputs:
        logits = self.forward(batch.x)
        return EvalOutputs(
            logits=logits.cpu(),
            ids=batch.y.cpu(),
        )

    @implements(pl.LightningModule)
    @torch.no_grad()
    def validation_step(
        self,
        batch: BinarySample,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> EvalStepOutput:
        return self.inference_step(batch=batch)

    @torch.no_grad()
    def _evaluate(self, outputs: EvalOutputs) -> MetricDict:
        same_id = (outputs.ids.unsqueeze(1) == outputs.ids).long()
        pred = self.model.predict(queries=outputs.logits)
        y_true = same_id[pred.retrieved_inds]
        map = self.rmap(preds=pred.scores, target=y_true, indexes=pred.query_inds)
        return {"mean_average_precision": map.item()}

    def _epoch_end(self, outputs: Union[List[EvalOutputs], EvalEpochOutput]) -> MetricDict:
        outputs_agg = reduce(operator.add, outputs)
        return self._evaluate(outputs_agg)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def validation_epoch_end(self, outputs: EvalEpochOutput) -> None:
        results_dict = self._epoch_end(outputs=outputs)
        results_dict = prefix_keys(results_dict, prefix=str(Stage.validate))
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def test_step(
        self,
        batch: BinarySample,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> EvalStepOutput:
        return self.inference_step(batch=batch)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def test_epoch_end(self, outputs: EvalEpochOutput) -> None:
        results_dict = self._epoch_end(outputs=outputs)
        results_dict = prefix_keys(results_dict, prefix=str(Stage.test))
        self.log_dict(results_dict)

    def predict_step(
        self, batch: BinarySample[Tensor], batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> BinarySample:

        return BinarySample(x=self.forward(batch.x), y=batch.y).to("cpu")

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> Union[
        Tuple[
            Union[List[optim.Optimizer], optim.Optimizer],
            List[Mapping[str, Union[LRScheduler, int, TrainingMode]]],
        ],
        Union[List[optim.Optimizer], optim.Optimizer],
    ]:
        optimizer_config = DictConfig({"_target_": self.optimizer_cls})
        if self.optimizer_kwargs is not None:
            optimizer_config.update(self.optimizer_kwargs)

        optimizer = instantiate(optimizer_config, params=self.parameters(), lr=self.lr)

        if self.scheduler_cls is not None:
            scheduler_config = DictConfig({"_target_": self.scheduler_cls})
            if self.scheduler_kwargs is not None:
                scheduler_config.update(self.scheduler_kwargs)
            scheduler = instantiate(scheduler_config, optimizer=optimizer)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": self.lr_sched_interval.name,
                "frequency": self.lr_sched_freq,
            }
            return [optimizer], [scheduler_config]
        return optimizer

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _run_internal(
        self, datamodule: pl.LightningDataModule, *, trainer: pl.Trainer, test: bool = True
    ) -> Self:
        # Run routines to tune hyperparameters before training.
        trainer.tune(model=self, datamodule=datamodule)
        # Train the model
        trainer.fit(model=self, datamodule=datamodule)
        if test:
            # Test the model if desired
            trainer.test(
                model=self,
                ckpt_path="best" if self.test_on_best else None,
                datamodule=datamodule,
            )
        return self

    def run(
        self, datamodule: pl.LightningDataModule, *, trainer: pl.Trainer, test: bool = True
    ) -> Self:
        return self._run_internal(datamodule=datamodule, trainer=trainer, test=test)
