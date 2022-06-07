from dataclasses import dataclass

from conduit.logging import init_logger

from whaledo.algorithms.base import Algorithm

__all__ = ["SelfSupervisedAlgorithm"]

LOGGER = init_logger(name=__file__)


@dataclass(unsafe_hash=True)
class SelfSupervisedAlgorithm(Algorithm):
    ...
