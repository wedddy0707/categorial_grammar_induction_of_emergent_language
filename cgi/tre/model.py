from typing import Tuple, Union, Literal

import torch
import torch.nn as nn

MeaningTensor = Union[Tuple["MeaningTensor", "MeaningTensor"], torch.Tensor]


class Composer(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        output_len: int,
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
            return self.emb(x).view(self.output_vocab_size, self.output_len)
        else:
            return self.lproj(self.forward(x[0])) + self.rproj(self.forward(x[1]))


class Objective(nn.Module):
    composer: Composer
    error_fn: Literal["L1-distance", "cross_entropy"]

    def __init__(
        self,
        composer: Composer,
        error_fn: Literal["L1-distance", "cross_entropy"] = "L1-distance",
    ):
        super().__init__()
        self.composer = composer
        self.error_fn = error_fn

    def forward(
        self,
        input: MeaningTensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        composed = self.composer(input)
        if self.error_fn == "L1-distance":
            error = torch.abs(composed - target).sum()
        else:
            raise NotImplementedError

        return error
