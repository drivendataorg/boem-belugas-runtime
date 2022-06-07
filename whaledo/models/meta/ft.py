from dataclasses import dataclass
from src.models.meta.base import MetaModel

__all__ = ["LinearProbe", "BitFit"]


@dataclass(unsafe_hash=True)
class LinearProbe(MetaModel):
    def __post_init__(self) -> None:
        for param in self.model.backbone.parameters():
            param.requires_grad_(False)
        super().__post_init__()


@dataclass(unsafe_hash=True)
class BitFit(MetaModel):
    def __post_init__(self) -> None:
        for name, param in self.model.backbone.named_parameters():
            if "bias" not in name:
                param.requires_grad_(False)
        super().__post_init__()
