import torch.nn as nn


RNN_TYPE = {
    "rnn": nn.RNN,
    "gru": nn.GRU,
    "lstm": nn.LSTM,
}

RNN_CELL_TYPE = {
    "rnn": nn.RNNCell,
    "gru": nn.GRUCell,
    "lstm": nn.LSTMCell,
}
