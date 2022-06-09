from typing import Optional, Tuple, Union

from ranzen.decorators import implements
import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import TypeAlias

__all__ = ["MultiCropWrapper"]

MultiResInput: TypeAlias = Tuple[Tensor, Tensor]


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are concatenated and a
    single forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone: nn.Module, *, head: Optional[nn.Module]) -> None:
        super().__init__()
        self.backbone = nn.Sequential(backbone, nn.Flatten())
        self.head = nn.Identity() if head is None else head

    @implements(nn.Module)
    def forward(self, x: Union[MultiResInput, Tensor], *, return_features: bool = False) -> Tensor:
        if isinstance(x, Tensor):
            features = self.backbone(x)
            logits = self.head(features)
        else:
            features_ls = []
            for x_i in x:
                logits_i = self.backbone(x_i)
                features_ls.append(logits_i)
            features = torch.cat(features_ls, dim=0)
            # Reshape the logits to align each global/local crop with its respective global crop
            # yielding a tensor of shape [N, L, C].
            features_global, features_local = features.tensor_split([len(x[0])], dim=0)
            features_global = features_global.unsqueeze(1)
            features_local = features_local.view(
                features_global.size(0), -1, features_global.size(-1)
            )
            # Interleave the local crops with the global crops
            features = torch.cat((features_global, features_local), dim=1)
            # Run the head forward on the concatenated features.
            logits = self.head(features.view(-1, features.size(-1)))
            logits = logits.view(*features.shape[:2], -1)

        return logits
