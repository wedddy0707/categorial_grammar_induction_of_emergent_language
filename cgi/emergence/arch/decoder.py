from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .params import RNN_CELL_TYPE
from .encoder import Encoder
from .attention import AttentionMechanism, DummyAttentionMechanism


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        max_length: int,
        cell: str = "lstm",
        enable_attention: bool = False,
        dropout_p: float = 0,
    ):
        super().__init__()
        ###############
        # Constraints #
        ###############
        assert isinstance(vocab_size, int) and vocab_size > 1
        assert isinstance(embed_size, int) and embed_size > 0
        assert isinstance(hidden_size, int) and hidden_size > 0
        assert isinstance(max_length, int) and max_length > 0
        assert isinstance(dropout_p, (int, float)) and 0 <= dropout_p <= 1
        assert isinstance(cell, str)
        cell = cell.lower()
        assert cell in RNN_CELL_TYPE, (
            f"Unkown cell name {cell}. "
            f"cell name expected to be one of {RNN_CELL_TYPE}"
        )
        ##########
        # Config #
        ##########
        self.config: Dict[Any, Any] = {
            "vocab_size": vocab_size,
            "embed_size": embed_size,
            "hidden_size": hidden_size,
            "max_length": max_length,
            "cell": cell,
            "enable_attention": enable_attention,
            "dropout_p": dropout_p,
        }
        ##################
        # Initialization #
        ##################
        # Attention mechanism
        if enable_attention:
            self.attention = AttentionMechanism(hidden_size)
        else:
            self.attention = DummyAttentionMechanism()

        # Dropout mechanism
        self.dropout_p = dropout_p

        # Decoder Body
        self.rnn = RNN_CELL_TYPE[cell](embed_size, hidden_size)
        self.bos = nn.parameter.Parameter(torch.randn(embed_size))
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_length

    @property
    def no_dropout(self):
        return self.dropout_p == 0.0

    @property
    def isLSTM(self):
        return isinstance(self.rnn, (nn.LSTM, nn.LSTMCell))

    def reset_parameters(self):
        if isinstance(self.attention, AttentionMechanism):
            self.attention.reset_parameters()
        nn.init.normal_(self.bos, 0.0, 0.01)
        self.rnn.reset_parameters()
        self.emb.reset_parameters()
        self.out.reset_parameters()

    def reborn(self):
        return self.__class__(**self.config)

    def forward(
        self,
        enc_hs: torch.Tensor,
        last_enc_h: torch.Tensor,
        gold_output_seq: Optional[torch.Tensor] = None,
    ):
        #############
        # Meta Info #
        #############
        device = enc_hs.device
        batch_size = enc_hs.shape[0]

        # self.training -> gold_output_seq is given
        assert (not self.training) or gold_output_seq is not None

        ###############
        # Preparation #
        ###############
        input = self.bos.unsqueeze(0).expand(batch_size, -1)
        dec_h = last_enc_h
        dec_c = torch.zeros_like(dec_h)
        output_seq: List[torch.Tensor] = []

        if self.training:
            dropout_mask = torch.bernoulli(torch.full_like(dec_h, 1.0 - self.dropout_p))
            dropout_mask = dropout_mask * dropout_mask.shape[1]
            dropout_mask = dropout_mask / dropout_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            dropout_mask = torch.ones_like(dec_h)

        ###########
        # Forward #
        ###########
        for step in range(self.max_len):
            #######################
            # Forward propagation
            if isinstance(self.rnn, nn.LSTMCell):
                dec_h, dec_c = self.rnn.forward(input, (dec_h, dec_c))
            else:
                dec_h = self.rnn.forward(input, dec_h)
            ###########################
            # Apply Dropout mechanism
            dec_h = dec_h * dropout_mask
            #############################
            # Apply Attention mechanism
            dec_h_for_output = self.attention.forward(enc_hs, dec_h)
            ################
            # Output layer
            output = self.out.forward(dec_h_for_output)
            #######################
            # Define a next input
            if self.training and gold_output_seq is not None:
                input = self.emb.forward(gold_output_seq[:, step])
            elif self.training and gold_output_seq is None:
                raise ValueError("gold_output_seq should not be None.")
            else:
                input = self.emb.forward(output.argmax(dim=1))
            ###########################
            # Make an output sequence
            output_seq.append(output)

        torch_output_seq = torch.stack(output_seq).permute(1, 0, 2)

        dummy = torch.zeros((batch_size, self.max_len)).to(device)

        return torch_output_seq, dummy, dummy


class Decoder_REINFORCE(Decoder):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        max_length: int,
        cell: str = "lstm",
        enable_attention: bool = False,
        dropout_p: float = 0.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_size=embed_size,
            max_length=max_length,
            hidden_size=hidden_size,
            cell=cell,
            enable_attention=enable_attention,
            dropout_p=dropout_p
        )

    def sample_from(self, distr: Categorical) -> torch.Tensor:
        if self.training:
            return distr.sample()
        else:
            return distr.probs.argmax(dim=1)

    def forward(
        self,
        enc_hs: torch.Tensor,
        last_enc_h: torch.Tensor,
        gold_output_seq: Optional[torch.Tensor] = None,
    ):
        #############
        # Meta Info #
        #############
        device = enc_hs.device
        batch_size = enc_hs.shape[0]

        ###############
        # Preparation #
        ###############
        input = self.bos.unsqueeze(0).expand(batch_size, -1)
        dec_h = last_enc_h
        dec_c = torch.zeros_like(dec_h)
        output_seq: List[torch.Tensor] = []
        logprob: List[torch.Tensor] = []
        entropy: List[torch.Tensor] = []

        if not self.no_dropout and self.training:
            dropout_mask = torch.bernoulli(torch.full_like(dec_h, 1.0 - self.dropout_p))
            dropout_mask = dropout_mask * dropout_mask.shape[1]
            dropout_mask = dropout_mask / dropout_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            dropout_mask = torch.ones_like(dec_h)

        ###########
        # Forward #
        ###########
        for _ in range(self.max_len):
            # Forward propagation
            if isinstance(self.rnn, nn.LSTMCell):
                dec_h, dec_c = self.rnn.forward(input, (dec_h, dec_c))
            else:
                dec_h = self.rnn.forward(input, dec_h)
            # Apply Dropout mechanism
            dec_h = dec_h * dropout_mask
            # Apply Attention mechanism
            dec_h_for_output = self.attention.forward(enc_hs, dec_h)
            # Output layer
            distr = Categorical(logits=self.out.forward(dec_h_for_output))
            output = self.sample_from(distr)
            # Define next input
            input = self.emb.forward(output)
            # Update logprob and entropy
            logprob.append(distr.log_prob(output))
            entropy.append(distr.entropy())
            # Update output_seq
            output_seq.append(output)

        torch_logprob = torch.stack(logprob).permute(1, 0)
        torch_entropy = torch.stack(entropy).permute(1, 0)
        torch_output_seq = torch.stack(output_seq).permute(1, 0)

        zeros = torch.zeros((batch_size, 1), device=device)
        torch_entropy = torch.cat([torch_entropy, zeros], dim=1)
        torch_logprob = torch.cat([torch_logprob, zeros], dim=1)
        torch_output_seq = torch.cat([torch_output_seq, zeros.long()], dim=1)

        return torch_output_seq, torch_logprob, torch_entropy


class Discriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        cell: str = "lstm",
        dropout_p: float = 0.0,
        length_type: str = "eos",
        impatient: bool = False,
    ):
        ###############
        # Constraints #
        ###############
        assert isinstance(impatient, bool)
        ##########
        # Config #
        ##########
        self.config: Dict[Any, Any] = {
            "vocab_size": vocab_size,
            "embed_size": embed_size,
            "hidden_size": hidden_size,
            "cell": cell,
            "dropout_p": dropout_p,
            "length_type": length_type,
            "impatient": impatient,
        }
        ##################
        # Initialization #
        ##################
        super().__init__()
        self.encoder = Encoder(
            vocab_size,
            embed_size,
            hidden_size,
            cell=cell,
            dropout_p=dropout_p,
            length_type=length_type)
        self.impatient = impatient

    @property
    def isLSTM(self):
        return self.encoder.isLSTM

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def reborn(self):
        return self.__class__(**self.config)

    def forward(
        self,
        encoder_hs: torch.Tensor,
        encoder_last_h: torch.Tensor,
        candidates: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #############
        # Meta Info #
        #############
        device = candidates.device
        batch_size, k = candidates.shape[:2]

        ###########
        # Forward #
        ###########
        _, candidates_h = self.encoder(candidates.view(batch_size * k, -1))

        if self.isLSTM:
            candidates_h, _ = candidates_h

        candidates_h = candidates_h.view(batch_size, k, -1)

        if self.impatient:
            score: torch.Tensor = torch.einsum("bnh,bth->bnt", candidates_h, encoder_hs)
        else:
            score: torch.Tensor = torch.einsum("bnh,bh->bn", candidates_h, encoder_last_h)

        dummy = torch.zeros(batch_size).to(device)

        return score, dummy, dummy
