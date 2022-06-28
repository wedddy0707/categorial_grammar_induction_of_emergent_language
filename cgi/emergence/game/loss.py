from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..msg import get_mask, get_length


class Loss(nn.Module):
    def __init__(self):
        super().__init__()


class LossRR(Loss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_s: torch.Tensor,
        output_s: Optional[torch.Tensor],
        input_r: Optional[torch.Tensor],
        output_r: torch.Tensor,
        label: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask_input_s = get_mask(input_s).long()
        len_input_s = get_length(input_s).float()

        input_s = input_s * mask_input_s
        output_r = output_r * mask_input_s

        loss = (input_s != output_r).float().sum(dim=1)
        acc = torch.as_tensor(1) - (loss / len_input_s)

        # perfect_match_loss = ((input_s != output_r).float().sum(dim=1) != input_s.shape[1]).float() * 10
        # loss = loss + perfect_match_loss

        return loss, {'acc': acc}


class LossRS(Loss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_s: torch.Tensor,
        output_s: Optional[torch.Tensor],
        input_r: Optional[torch.Tensor],
        output_r: torch.Tensor,
        label: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        length = get_length(input_s).cpu()

        target: nn.utils.rnn.PackedSequence = nn.utils.rnn.pack_padded_sequence(
            input_s,
            length,
            batch_first=True,
            enforce_sorted=False,
        )
        output: nn.utils.rnn.PackedSequence = nn.utils.rnn.pack_padded_sequence(
            output_r,
            length,
            batch_first=True,
            enforce_sorted=False,
        )

        assert (target.sorted_indices == output.sorted_indices).all()
        assert (target.unsorted_indices == output.unsorted_indices).all()

        acc = (target.data == output.data.argmax(dim=1)).float().mean()

        loss = nn.utils.rnn.PackedSequence(
            data=F.cross_entropy(output.data, target.data, reduction='none'),
            batch_sizes=target.batch_sizes,
            sorted_indices=target.sorted_indices,
            unsorted_indices=target.unsorted_indices,
        )
        loss, _ = nn.utils.rnn.pad_packed_sequence(
            loss,
            batch_first=True,
            total_length=int(length.max().item()))
        loss = loss.sum(dim=1)

        return loss, {'acc': acc}
