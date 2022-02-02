from typing import Sequence, Optional
import argparse
import egg.core as core

from ...util import s2b

def get_common_params(
    params: Sequence[str],
    parser: Optional[argparse.ArgumentParser] = None,
):
    if parser is None:
        parser = argparse.ArgumentParser()
    ###############
    # on training #
    ###############
    parser.add_argument('--early_stopping_thr', type=float, default=0.9999)
    parser.add_argument('--batches_per_epoch', type=int, default=1000)
    # entropy
    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1)
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1)
    # length cost
    parser.add_argument('--length_cost', type=float, default=0.0)
    ##############################
    # hyper-parameters of models #
    ##############################
    # size
    parser.add_argument('--embed_size', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=100)
    # attention
    parser.add_argument('--enable_sender_attention', type=s2b, default=False)
    parser.add_argument('--enable_receiver_attention', type=s2b, default=False)
    # cell type {rnn, gru, lstm}
    parser.add_argument('--cell', type=str, default='rnn')
    ###############
    # game config #
    ###############
    parser.add_argument('--max_n_conj', type=int, default=0)

    args = core.init(parser, params)
    return args
