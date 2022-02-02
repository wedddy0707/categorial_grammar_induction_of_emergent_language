from typing import Sequence, Optional
import argparse
import egg.core as core


def get_common_params(
    params: Sequence[str],
    parser: Optional[argparse.ArgumentParser] = None,
):
    if parser is None:
        parser = argparse.ArgumentParser()
    ###############
    # on training #
    ###############
    parser.add_argument(
        '--early_stopping_thr', type=float, default=0.9999,
        help="Early stopping threshold on accuracy (default: 0.9999)")
    parser.add_argument(
        '--batches_per_epoch', type=int, default=1000,
        help="number of batches at each epoch")
    # entropy
    parser.add_argument(
        '--sender_entropy_coeff', type=float, default=1e-1,
        help='The entropy regularisation coeff for Sender (default: 1e-1)')
    parser.add_argument(
        '--receiver_entropy_coeff', type=float, default=1e-1,
        help='The entropy regularisation coeff for Receiver (default: 1e-1)')
    # length cost
    parser.add_argument(
        '--length_cost', type=float, default=0.0,
        help="length cost")
    # receiver impatience
    parser.add_argument(
        '--impatient', default=False, action='store_true',
        help="receiver impatience")
    ##############################
    # hyper-parameters of models #
    ##############################
    # size
    parser.add_argument(
        '--embed_size', type=int, default=10,
        help='embed size of message symbols (default: 10)')
    parser.add_argument(
        '--hidden_size', type=int, default=100,
        help='hidden size of agents (default: 100)')
    # attention
    parser.add_argument(
        '--sender_attention', default=False,
        help='whether to use attention mechanism for sender.',
        action='store_true')
    parser.add_argument(
        '--receiver_attention', default=False,
        help='whether to use attention mechanism for receiver.',
        action='store_true')
    # cell type {rnn, gru, lstm}
    parser.add_argument(
        '--cell', type=str, default='rnn',
        help='Cell for agents {rnn, gru, lstm} (default: rnn)')
    ###############
    # game config #
    ###############
    parser.add_argument(
        '--max_n_conj', type=int, default=0)

    args = core.init(parser, params)
    return args
