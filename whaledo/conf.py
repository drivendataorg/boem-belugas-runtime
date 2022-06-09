from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class WandbLoggerConf:
    _target_: str = "pytorch_lightning.loggers.wandb.WandbLogger"
    name: Optional[str] = None
    save_dir: Optional[str] = None
    offline: Optional[bool] = False
    id: Optional[str] = None
    anonymous: Optional[bool] = None
    version: Optional[str] = None
    project: Optional[str] = None
    log_model: Optional[bool] = False
    experiment: Any = None
    prefix: Optional[str] = None
    group: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[List] = None
    reinit: bool = False
    job_type: Optional[str] = None
    mode: Optional[str] = None
    resume: Optional[str] = None
