from typing import Tuple, Union

import torch
import torch.nn as nn

MeaningTensor = Union[Tuple['MeaningTensor', 'MeaningTensor'], torch.LongTensor]  # noqa: E501


class Composer(nn.Module):
    def __init__(
        self,
        input_vocab_size:  int,
        output_vocab_size: int,
        output_len:        int,
    ):
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.output_len = output_len

        self.emb = nn.Embedding(
            input_vocab_size, output_vocab_size * output_len)
        self.lproj = nn.Linear(output_len, output_len, bias=False)
        self.rproj = nn.Linear(output_len, output_len, bias=False)

    def forward(self, x: MeaningTensor) -> torch.Tensor:
        if isinstance(x, torch.LongTensor):
            return self.emb(x).view(
                self.output_vocab_size, self.output_len)
        else:
            return (
                self.lproj(self.forward(x[0])) +
                self.rproj(self.forward(x[1])))


class Objective(nn.Module):
    def __init__(self, composer: 'Composer'):
        super().__init__()
        self.composer = composer

    def forward(
        self,
        input: MeaningTensor,
        target: torch.LongTensor
    ) -> torch.Tensor:
        composed = self.composer(input)
        return torch.abs(composed - target).sum()
