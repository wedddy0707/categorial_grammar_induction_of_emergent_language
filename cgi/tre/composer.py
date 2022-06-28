from typing import Any, Tuple, Union
import torch
import torch.nn as nn


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

    def forward(self, x: Union[Tuple[Any, Any], torch.Tensor]):
        if isinstance(x, tuple):
            return self.lproj(self.forward(x[0])) + self.rproj(self.forward(x[1]))
        else:
            return self.emb(x).view(
                self.output_vocab_size, self.output_len)
