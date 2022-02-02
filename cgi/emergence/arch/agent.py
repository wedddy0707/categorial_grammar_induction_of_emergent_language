from typing import (
    Optional,
    Union,
)
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import (
    Decoder,
    Decoder_REINFORCE,
    Discriminator,
)


class Agent(nn.Module):
    encoder: Encoder
    decoder: Union[Decoder, Decoder_REINFORCE, Discriminator]

    def __init__(
        self,
        encoder: Encoder,
        decoder: Union[Decoder, Decoder_REINFORCE, Discriminator],
    ):
        ###############
        # Constraints #
        ###############
        assert hasattr(encoder, 'isLSTM')
        ##################
        # Initialization #
        ##################
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: Optional[torch.Tensor] = None
    ):
        encoder_hs, last_encoder_h = self.encoder(encoder_input)
        if self.encoder.isLSTM:
            last_encoder_h, _ = last_encoder_h
        return self.decoder(
            encoder_hs,
            last_encoder_h,
            decoder_input)
