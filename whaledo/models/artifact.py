from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Final, Optional, Tuple, Union

from conduit.logging import init_logger
from hydra.utils import instantiate
from ranzen.decorators import implements
import torch
import torch.nn as nn
import wandb
from wandb.sdk.lib.disabled import RunDisabled
from wandb.wandb_run import Run

from whaledo.models.base import BackboneFactory, Model, ModelFactoryOut
from whaledo.models.meta import MetaModel

__all__ = [
    "DEFAULT_FILENAME",
    "load_model_from_artifact",
    "save_model_artifact",
    "ArtifactLoader",
]

LOGGER = init_logger(__file__)

DEFAULT_FILENAME: Final[str] = "final_model.pt"


def save_model_artifact(
    model: Union[Model, MetaModel],
    *,
    run: Union[Run, RunDisabled],
    config: Dict[str, Any],
    filename: str = DEFAULT_FILENAME,
) -> None:
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_save_path = tmpdir / filename
        save_dict = {
            "state": {
                "backbone": model.backbone.state_dict(),
            },
            "config": config,
        }
        torch.save(save_dict, f=model_save_path)
        LOGGER.info(f"Model config and state saved to '{model_save_path.resolve()}'")
        artifact_name = config["dm"]["_target_"].split(".")[-1].removesuffix("DataModule").lower()
        artifact_name += "_"
        artifact_name += "_".join(
            dict_conf["_target_"].split(".")[-1].lower()
            for dict_conf in (
                config["alg"],
                config["backbone"],
            )
        )
        model_artifact = wandb.Artifact(artifact_name, type="model", metadata=config)
        model_artifact.add_file(str(model_save_path.resolve()), name=filename)
        run.log_artifact(model_artifact)
        model_artifact.wait()
        LOGGER.info(
            "Model artifact saved to "
            f"'{run.entity}/{run.project}/{artifact_name}:{model_artifact.version}'"
        )


def load_model_from_artifact(
    name: str,
    *,
    run: Optional[Union[Run, RunDisabled]],
    project: Optional[str] = None,
    filename: str = DEFAULT_FILENAME,
    target_dim: Optional[int] = None,
    root: Optional[Union[Path, str]] = None,
) -> Tuple[nn.Module, int]:
    if root is None:
        root = Path("artifacts") / "models"
    root = Path(root)
    artifact_dir = root / name
    filepath = artifact_dir / filename
    if (run is not None) and (project is None):
        project = f"{run.entity}/{run.project}"
        full_name = f"{project}/{name}"
        artifact = run.use_artifact(full_name)
        LOGGER.info("Downloading model artifact...")
        artifact.download(root=artifact_dir)
    else:
        if not filepath.exists():
            raise RuntimeError(
                f"No pre-existing model-artifact found at location '{filepath.resolve()}'"
                "and because no wandb run has been specified, it can't be downloaded."
            )
        full_name = artifact_dir
    state_dict = torch.load(filepath)
    LOGGER.info("Loading saved parameters and buffers...")
    bb_fn: BackboneFactory = instantiate(state_dict["config"]["backbone"])
    backbone, feature_dim = bb_fn()
    backbone.load_state_dict(state_dict["state"]["backbone"])
    LOGGER.info(f"Model successfully loaded from artifact '{full_name}'.")
    return backbone, feature_dim


@dataclass
class ArtifactLoader(BackboneFactory):
    name: str
    project: Optional[str] = None
    filename: str = DEFAULT_FILENAME
    root: Optional[str] = None

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[nn.Module]:
        return load_model_from_artifact(
            name=self.name,
            run=wandb.run,
            project=self.project,
            filename=self.filename,
            root=self.root,
            target_dim=None,
        )
