import torch
import torch.nn as nn


class AttentionMechanism(nn.Module):
    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()
        self.attention_c_matrix = nn.Linear(hidden_size, hidden_size, bias=False)  # noqa: E501
        self.attention_h_matrix = nn.Linear(hidden_size, hidden_size, bias=False)  # noqa: E501
        self.nonlinearity = torch.tanh

    def reset_parameters(self):
        self.attention_c_matrix.reset_parameters()
        self.attention_h_matrix.reset_parameters()

    def forward(
        self,
        decoder_h: torch.Tensor,
        encoder_hs: torch.Tensor,
    ):
        attention_score: torch.Tensor = torch.softmax(torch.einsum('bij,bj->bi', encoder_hs, decoder_h), dim=-1)
        context_h:       torch.Tensor = torch.einsum('bij,bi->bj', encoder_hs, attention_score)
        return self.nonlinearity(
            self.attention_c_matrix(context_h) +
            self.attention_h_matrix(decoder_h))
