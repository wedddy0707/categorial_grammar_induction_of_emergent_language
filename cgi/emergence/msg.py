import torch


def get_mask(sequence: torch.Tensor, eos_id: int = 0) -> torch.BoolTensor:
    zero = (sequence == eos_id).long()
    return ((zero.cumsum(dim=-1) - zero) == 0)


def get_length(sequence: torch.Tensor, eos_id: int = 0) -> torch.Tensor:
    return (get_mask(sequence, eos_id=eos_id) > 0).long().sum(dim=-1)
