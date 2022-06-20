from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import attr
from fairscale.nn import auto_wrap  # type: ignore
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from ranzen.decorators import implements
from ranzen.hydra import Option, Relay

from whaledo.algorithms.base import Algorithm
from whaledo.conf import WandbLoggerConf
from whaledo.data.datamodule import WhaledoDataModule
from whaledo.models import MetaModel, Model
from whaledo.models.artifact import save_model_artifact

__all__ = ["WhaledoRelay"]


@attr.define(kw_only=True)
class WhaledoRelay(Relay):
    dm: DictConfig
    alg: DictConfig
    backbone: DictConfig
    trainer: DictConfig
    logger: DictConfig
    checkpointer: DictConfig
    meta_model: Optional[DictConfig] = None
    seed: Optional[int] = 42
    output_dir: str = "outputs"
    save_model: bool = False
    save_best: bool = True

    @classmethod
    @implements(Relay)
    def with_hydra(
        cls,
        root: Path | str,
        *,
        dm: list[Option],
        alg: list[Option],
        backbone: list[Option],
        meta_model: list[Option],
        clear_cache: bool = False,
    ) -> None:

        configs = dict(
            dm=dm,
            alg=alg,
            backbone=backbone,
            meta_model=meta_model,
            trainer=[Option(class_=pl.Trainer, name="base")],
            logger=[Option(class_=WandbLoggerConf, name="base")],
            checkpointer=[Option(class_=ModelCheckpoint, name="base")],
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=False,
            clear_cache=clear_cache,
            **configs,
        )

    @implements(Relay)
    def run(self, raw_config: Dict[str, Any]) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        pl.seed_everything(self.seed, workers=True)

        dm: WhaledoDataModule = instantiate(self.dm)
        dm.prepare_data()
        dm.setup()

        backbone, feature_dim = instantiate(self.backbone)()
        model: Union[Model, MetaModel]
        model = Model(backbone=backbone, feature_dim=feature_dim)

        # enable parameter sharding with fairscale.
        # Note: when fully-sharded training is not enabled this is a no-op
        model = auto_wrap(model)  # type: ignore

        if self.logger.get("group", None) is None:
            default_group = f"{dm.__class__.__name__.removesuffix('DataModule').lower()}_"
            default_group += "_".join(
                dict_conf["_target_"].split(".")[-1].lower()
                for dict_conf in (self.backbone, self.alg)
            )
            self.logger["group"] = default_group
        logger: WandbLogger = instantiate(self.logger, reinit=True)
        if raw_config is not None:
            logger.log_hyperparams(raw_config)  # type: ignore

        # Disable checkpointing when instantiating the trainer as we want to use
        # a hydra-instantiated checkpointer.
        trainer: pl.Trainer = instantiate(
            self.trainer,
            logger=logger,
            enable_checkpointing=False,
        )
        output_dir = Path(to_absolute_path(self.output_dir))
        output_dir.mkdir(exist_ok=True, parents=True)
        checkpointer: ModelCheckpoint = instantiate(
            self.checkpointer, dirpath=output_dir, save_weights_only=True
        )
        trainer.callbacks.append(checkpointer)

        if self.meta_model is not None:
            model = instantiate(self.meta_model, _partial_=True)(model=model)
        alg: Algorithm = instantiate(self.alg, _partial_=True)(model=model)
        alg.run(datamodule=dm, trainer=trainer)

        if self.save_model and (not logger.experiment.offline):
            if self.save_best and cast(str, best_model_path := checkpointer.best_model_path):
                self.log(f"Loading best model from checkpoint '{best_model_path}'.")
                alg.load_from_checkpoint
                alg_kwargs = dict(self.alg)
                alg_kwargs.pop("_target_")
                alg_kwargs["model"] = model
                alg.load_from_checkpoint(
                    checkpoint_path=best_model_path,
                    **alg_kwargs, # type: ignore
                )
                self.log("Checkpoint successfully loaded.")

            save_model_artifact(
                model=model,
                run=logger.experiment,
                config=raw_config,
                image_size=dm.image_size,
                filename="final_model.pt",
            )
