import torch
import torch.nn as nn

from ..msg import get_length

from .params import RNN_CELL_TYPE
from .params import RNN_TYPE


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        embed_size:  int,
        hidden_size: int,
        cell:        str = 'lstm',
        dropout_p:   float = 0.0,
        length_type: str = 'eos',
    ):
        ###############
        # Constraints #
        ###############
        assert isinstance(vocab_size,  int) and vocab_size > 1
        assert isinstance(embed_size,  int) and embed_size > 0
        assert isinstance(hidden_size, int) and hidden_size > 0
        assert isinstance(dropout_p, float) and 0.0 <= dropout_p <= 1.0
        assert isinstance(cell,        str)
        cell = cell.lower()
        assert cell in RNN_CELL_TYPE, (
            f'Unkown cell name {cell}. '
            f'cell name expected to be one of {RNN_CELL_TYPE}')
        assert length_type in {'fixed', 'eos'}
        ##########
        # Config #
        ##########
        self.config = {
            'vocab_size':  vocab_size,
            'embed_size':  embed_size,
            'hidden_size': hidden_size,
            'cell':        cell,
            'dropout_p':   dropout_p,
            'length_type': length_type,
        }
        ##################
        # Initialization #
        ##################
        super().__init__()

        self.length_type = length_type
        self.dropout_p = dropout_p
        # Encoder
        self.emb = nn.Embedding(vocab_size, embed_size)
        if self.no_dropout:
            self.rnn = RNN_TYPE[cell](
                input_size=embed_size,
                hidden_size=hidden_size,
                batch_first=True)
        else:
            self.rnn = RNN_CELL_TYPE[cell](
                input_size=embed_size,
                hidden_size=hidden_size)

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    @property
    def embed_size(self):
        return self.rnn.input_size

    @property
    def no_dropout(self):
        return self.dropout_p == 0.0

    @property
    def isLSTM(self):
        return isinstance(self.rnn, (nn.LSTM, nn.LSTMCell))

    def reset_parameters(self):
        self.emb.reset_parameters()
        self.rnn.reset_parameters()

    def reborn(self):
        return self.__class__(**self.config)

    def copy(self):
        new_model = self.reborn()
        new_model.load_state_dict(self.state_dict())
        return new_model

    def forward(self, input: torch.LongTensor):
        #############
        # Meta Info #
        #############
        device = input.device
        max_batch_size, total_length = input.shape[:2]

        if self.length_type == 'fixed':
            length = torch.full(
                (max_batch_size,),
                total_length,
                dtype=torch.long,
                device=device)
        else:  # self.length_type == 'eos':
            length = get_length(input)

        ###############
        # Preparation #
        ###############
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(input),
            length.cpu(),
            batch_first=True,
            enforce_sorted=False)

        # If we do not apply dropout (i.e., self.dropout_p == 0.0),
        # then we can speed up the forward propagation.
        if self.no_dropout:
            hs, last_h = self.rnn(packed)
            if self.isLSTM:
                last_h = tuple(map(lambda x: x[-1], last_h))
            else:
                last_h = last_h[-1]
        else:
            hs = []
            old_h = torch.zeros(max_batch_size, self.hidden_size).to(device)
            if self.isLSTM:
                old_c = torch.zeros_like(old_h)
            #####################
            # Make dropout mask #
            #####################
            if self.training:
                dropout_mask = torch.bernoulli(
                    torch.full_like(old_h, 1.0 - self.dropout_p))
                dropout_mask = dropout_mask * (
                    float(dropout_mask.size(1)) /
                    dropout_mask.sum(dim=1, keepdim=True).clamp(min=1.0))
            else:
                dropout_mask = torch.ones_like(old_h)
            ###########
            # Forward #
            ###########
            input_idx = 0
            for batch_size in packed.batch_sizes.tolist():
                #######################
                # Input for each step
                step_input = packed.data[input_idx:input_idx + batch_size]
                input_idx += batch_size
                #######################
                # Forward propagation
                if self.isLSTM:
                    new_h = self.rnn(
                        step_input, (old_h[:batch_size], old_c[:batch_size]))
                    new_h, new_c = new_h
                else:
                    new_h = self.rnn(step_input, old_h[:batch_size])
                #################
                # Apply dropout
                new_h = new_h * dropout_mask[:batch_size]
                #############
                # Update hs
                hs.append(new_h)
                ########################
                # Update old_h (old_c)
                old_h = torch.cat([new_h, old_h[batch_size:]])
                if self.isLSTM:
                    old_c = torch.cat([new_c, old_c[batch_size:]])

            last_h = torch.index_select(old_h, 0, packed.unsorted_indices)
            if self.isLSTM:
                last_c = torch.index_select(
                    old_c, 0, packed.unsorted_indices)
                last_h = last_h, last_c

            hs = nn.utils.rnn.PackedSequence(
                data=torch.cat(hs),
                batch_sizes=packed.batch_sizes,
                sorted_indices=packed.sorted_indices,
                unsorted_indices=packed.unsorted_indices)
        # End of else sentence

        hs, _ = nn.utils.rnn.pad_packed_sequence(
            hs,
            batch_first=True,
            total_length=total_length)

        return hs, last_h
