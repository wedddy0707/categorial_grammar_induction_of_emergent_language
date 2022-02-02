from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn

from ..msg import get_mask, get_length


class Loss(nn.Module):
    def __init__(self):
        super().__init__()


class LossRR(Loss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_s:  torch.Tensor,
        output_s: Optional[torch.Tensor],
        input_r:  Optional[torch.Tensor],
        output_r: torch.Tensor,
        label:    Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask_input_s = get_mask(input_s).long()
        len_input_s = get_length(input_s).float()

        input_s = input_s * mask_input_s
        output_r = output_r * mask_input_s

        loss = (input_s != output_r).float().sum(dim=1)
        acc = torch.as_tensor(1) - (loss / len_input_s)

        return loss, {'acc': acc}
