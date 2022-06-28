from typing import (
    Callable,
    Optional,
    Type,
    Tuple,
    Dict,
    Any,
)
from collections import defaultdict
import torch
import torch.nn as nn
from ...io import make_logger
from ..msg import get_length
from .baseline import SimpleBaseline


def compute_effective_value(
    origin: torch.Tensor,
    length: torch.Tensor,
    take_average: bool,
):
    batch_size, seq_len = origin.shape[:2]
    device = origin.device
    effective = torch.zeros(batch_size, device=device)
    not_eosed = torch.ones(batch_size, dtype=torch.bool, device=device)
    for i in range(seq_len):
        not_eosed = torch.logical_and(not_eosed, i < length)
        effective = effective + origin[:, i] * not_eosed.float()
    if take_average:
        effective = effective / length
    return effective


class SingleGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        recver: nn.Module,
        sender_entropy_coeff: float,
        recver_entropy_coeff: float,
        loss: Callable[..., Tuple[torch.Tensor, Dict[str, Any]]],
        baseline_type: Type[SimpleBaseline] = SimpleBaseline,
        length_cost: float = 0.0,
    ):
        self.logger = make_logger(self.__class__.__name__)
        ##################
        # Initialization #
        ##################
        super().__init__()
        self.sender = sender
        self.recver = recver
        self.sender_entr_coeff = sender_entropy_coeff
        self.recver_entr_coeff = recver_entropy_coeff
        self.loss = loss
        self.baseline: defaultdict[str, SimpleBaseline] = defaultdict(baseline_type)
        self.length_cost = length_cost

    def forward(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: Optional[torch.Tensor] = None,
        aux_input: Optional[torch.Tensor] = None,
    ):
        sender_output, sender_logprob, sender_entropy = self.sender.forward(sender_input)
        recver_output, recver_logprob, recver_entropy = self.recver.forward(sender_output, receiver_input)

        sender_length = get_length(sender_output)
        recver_length = get_length(sender_input)
        logprob = compute_effective_value(
            sender_logprob, sender_length, take_average=False
        ) + compute_effective_value(
            recver_logprob, recver_length, take_average=False
        )
        entropy = self.sender_entr_coeff * compute_effective_value(
            sender_entropy, sender_length, take_average=True
        ).mean() + self.recver_entr_coeff * compute_effective_value(
            recver_entropy, recver_length, take_average=True
        ).mean()

        loss, rest = self.loss(sender_input, sender_output, receiver_input, recver_output, labels)

        policy_loss = (
            (loss.detach() - self.baseline['loss']()) * logprob).mean()
        optimized_loss = (
            loss.mean() + policy_loss - entropy
        )

        if self.training:
            self.baseline['loss'].update(loss)

        rest.update({
            'loss': optimized_loss,
            'original_loss': loss,
            'sender_entropy': sender_entropy,
            'recver_entropy': recver_entropy,
            'sender_mean_length': sender_length,
            'recver_mean_length': recver_length,
        })
        for k, v in rest.items():
            if isinstance(v, torch.Tensor):
                rest[k] = v.float().mean().item()
        return optimized_loss, rest
