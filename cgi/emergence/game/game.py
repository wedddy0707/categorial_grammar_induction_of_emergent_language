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


class SingleGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        recver: nn.Module,
        sender_entropy_coeff: float,
        recver_entropy_coeff: float,
        loss: Callable[..., Tuple[torch.Tensor, Dict[str, Any]]],
        baseline_type: Type[SimpleBaseline] = SimpleBaseline,
        length_cost:  float = 0.0,
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

    def __compute_effective_entropy(
        self,
        entropy: torch.Tensor,
        length: torch.Tensor,
    ):
        batch_size = entropy.size(0)
        device = entropy.device
        effective_entropy = torch.zeros(batch_size, device=device)
        not_eosed = torch.ones(batch_size, dtype=torch.bool, device=device)
        for i in range(len(entropy.size(1))):
            not_eosed = torch.logical_and(not_eosed, i < length)
            effective_entropy = effective_entropy + entropy[:, i] * not_eosed.float()
        effective_entropy = effective_entropy / length
        return effective_entropy

    def forward(
        self,
        sender_input: torch.LongTensor,
        recver_input: Optional[torch.LongTensor] = None,
        label:        Optional[torch.LongTensor] = None,
    ):
        sender_output, sender_logprob, sender_entropy = self.sender(sender_input)  # noqa: E501
        recver_output, recver_logprob, recver_entropy = self.recver(sender_output, recver_input)  # noqa: E501

        logprob = sender_logprob + recver_logprob
        entropy = (
            self.__compute_effective_entropy(sender_entropy, get_length(sender_output)).mean() * self.sender_entr_coeff +
            self.__compute_effective_entropy(recver_entropy, get_length(recver_output)).mean() * self.recver_entr_coeff
        )

        loss, rest = self.loss(sender_input, sender_output,
                               recver_input, recver_output, label)

        length = get_length(sender_output)

        policy_loss = (
            (loss.detach() - self.baseline['loss']()) * logprob).mean()
        length_loss = self.length_cost * (
            (length.float() - self.baseline['len']()) * sender_logprob).mean()
        optimized_loss = (
            loss.mean() +
            policy_loss +
            length_loss +
            entropy
        )

        if self.training:
            self.baseline['loss'].update(loss)
            self.baseline['len'].update(length.float())

        rest.update({
            'loss':           optimized_loss,
            'original_loss':  loss,
            'sender_entropy': sender_entropy,
            'recver_entropy': recver_entropy,
            'mean_length':    length})
        for k, v in rest.items():
            if isinstance(v, torch.Tensor):
                rest[k] = v.float().mean().item()
        return optimized_loss, rest
