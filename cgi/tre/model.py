from typing import Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

MeaningTensor = Union[Tuple["MeaningTensor", "MeaningTensor"], torch.Tensor]


class Composer(nn.Module):
    input_vocab_size: int
    output_vocab_size: int
    output_len: int

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
            input_vocab_size,
            output_vocab_size * output_len,
        )
        self.lproj = nn.Linear(output_len, output_len, bias=False)
        self.rproj = nn.Linear(output_len, output_len, bias=False)

    def forward(self, x: MeaningTensor) -> torch.Tensor:
        if isinstance(x, torch.LongTensor):
            return self.emb.forward(x).reshape(
                self.output_vocab_size,
                self.output_len,
            )
        else:
            return \
                self.lproj.forward(self.forward(x[0])) + \
                self.rproj.forward(self.forward(x[1]))


class Objective(nn.Module):
    composer: Composer
    error_fn: Literal["L1-distance", "L2-distance", "cross_entropy"]

    def __init__(
        self,
        composer: Composer,
        error_fn: Literal["L1-distance", "L2-distance", "cross_entropy"] = "L1-distance",
    ):
        super().__init__()
        self.composer = composer
        self.error_fn = error_fn

    def forward(
        self,
        input: MeaningTensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        composed = self.composer.forward(input)
        if self.error_fn == "L1-distance":
            error = torch.abs(composed - target).sum()
        elif self.error_fn == "L2-distance":
            error = torch.abs(composed - target)
            error = (error * error).sum()
        elif self.error_fn == "cross_entropy":
            error = F.cross_entropy(
                composed.permute(1, 0),
                target.argmax(dim=0),
                reduction="none",
            ).sum()
        else:
            raise NotImplementedError

        return error
