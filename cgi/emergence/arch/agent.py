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
        assert hasattr(encoder, 'reset_parameters')
        assert hasattr(decoder, 'reset_parameters')
        assert hasattr(encoder, 'reborn')
        assert hasattr(decoder, 'reborn')
        assert hasattr(encoder, 'isLSTM')
        ##################
        # Initialization #
        ##################
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def reborn(self):
        new_encoder = self.encoder.reborn()
        new_decoder = self.decoder.reborn()
        return self.__class__(new_encoder, new_decoder)

    def copy(self):
        new_encoder = self.encoder.copy()
        new_decoder = self.decoder.copy()
        return self.__class__(new_encoder, new_decoder)

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
