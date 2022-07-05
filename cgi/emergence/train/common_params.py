from typing import List, Optional, Literal
import argparse
import egg.core as core
import torch

from ...util import s2b


class NamespaceForEmergentLang:
    vocab_size: int
    max_len: int
    random_seed: int
    batch_size: int
    device: torch.device
    checkpoint_dir: Optional[str]
    checkpoint_freq: int
    n_epochs: int
    batches_per_epoch: int
    early_stopping_thr: float
    sender_entropy_coeff: float
    receiver_entropy_coeff: float
    sender_life_span: Optional[int]
    receiver_life_span: Optional[int]
    length_cost: float
    test_p: float
    embed_size: int
    hidden_size: int
    enable_sender_attention: bool
    enable_receiver_attention: bool
    cell: Literal["rnn", "gru", "lstm"]
    max_n_predicates: int
    language_dump_freq: Optional[int]
    stats_dump_freq: Optional[int]


def get_common_params(
    params: List[str],
    parser: Optional[argparse.ArgumentParser] = None,
):
    if parser is None:
        parser = argparse.ArgumentParser()
    ###############
    # on training #
    ###############
    parser.add_argument("--batches_per_epoch", type=int, default=128)
    parser.add_argument("--early_stopping_thr", type=float, default=0.9999)
    parser.add_argument("--sender_entropy_coeff", type=float, default=1e-1)
    parser.add_argument("--receiver_entropy_coeff", type=float, default=1e-1)
    parser.add_argument("--sender_life_span", type=int, default=None)
    parser.add_argument("--receiver_life_span", type=int, default=None)
    parser.add_argument("--length_cost", type=float, default=0.0)
    parser.add_argument("--test_p", type=float, default=0.1)
    parser.add_argument("--embed_size", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--enable_sender_attention", type=s2b, default=True)
    parser.add_argument("--enable_receiver_attention", type=s2b, default=True)
    parser.add_argument("--cell", type=str, choices=("rnn", "gru", "lstm"), default="gru")
    parser.add_argument("--max_n_predicates", type=int, default=0)
    parser.add_argument("--language_dump_freq", type=int, default=None)
    parser.add_argument("--stats_dump_freq", type=int, default=None)

    args = core.init(parser, params)
    typed_args = NamespaceForEmergentLang()
    for k, v in vars(args).items():
        setattr(typed_args, k, v)
    return typed_args
