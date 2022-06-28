import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyAttentionMechanism:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        encoder_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(encoder_feature, decoder_feature)

    def forward(
        self,
        encoder_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        return decoder_feature


class AttentionMechanism(nn.Module):
    w1: nn.Linear
    w2: nn.Linear

    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()

    def forward(
        self,
        encoder_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameter
        ---------
        encoder_feature: torch.Tensor of size (batch_size, n_nodes, hidden_size)
        decoder_feature: torch.Tensor of size (batch_size, hidden_size)

        Return
        ------
        new_feature: torch.Tensor of size (batch_size, hidden_size)
            New feature tensor just after attention mechanism.
        """
        logits = torch.bmm(encoder_feature, decoder_feature.unsqueeze(2)).squeeze(2)  # (batch_size, n_nodes)

        context_feature = torch.bmm(F.softmax(logits, dim=1).unsqueeze(1), encoder_feature).squeeze(1)  # (batch_size, hidden_dim)

        w1h = self.w1(decoder_feature)
        w2c = self.w2(context_feature)
        new_feature = torch.tanh(w1h + w2c)  # (B, H)

        return new_feature
