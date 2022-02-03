from typing import Sequence, Optional, List, Tuple
import sys
import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd

import egg.core as core

from ...io import make_logger
from ..arch import Encoder
from ..arch import Decoder_REINFORCE
from ..arch import Agent
from ..game import SingleGame
from ..game.loss import LossRR
from ..dataset import COMMAND_ID
from ..dataset import enumerate_command
from ..dataset import CommandDataset
from .common_params import get_common_params
from .dump import dump_params
from .intervene import DumpCorpus, Metrics

def get_params(
    params: Sequence[str],
    parser: Optional[argparse.ArgumentParser] = None,
):
    if parser is None:
        parser = argparse.ArgumentParser()
    args = get_common_params(params, parser)
    return args

class PlusOneWrapper(torch.nn.Module):
    def __init__(self, wrapped: torch.nn.Module):
        super().__init__()
        self.wrapped = wrapped

    def forward(
        self,
        *input: Tuple[Optional[torch.Tensor], ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r1, r2, r3 = self.wrapped(*input)
        r1 = r1[:, :-1]
        r2 = r2[:, :-1]
        r3 = r3[:, :-1]
        return r1 + 1, r2, r3

def main(argv: Sequence[str]):
    logger = make_logger('main')
    logger.info('logger initialized')

    opts = get_params(argv)
    dump_params(vars(opts))

    logger.info('Making Datasets...')
    df = enumerate_command(
        opts.max_n_conj,
        random_seed=opts.random_seed,
    )
    assert {"command_tensor", "split"} <= set(df), set(df)
    train_split: pd.DataFrame = df[df["split"] == "train"]
    valid_split: pd.DataFrame = df[df["split"] == "valid"]
    train_data: CommandDataset = CommandDataset(train_split)
    valid_data: CommandDataset = CommandDataset(valid_split)
    logger.info('Making Data Loaders...')
    train_sampler = RandomSampler(
        train_data,
        replacement=True,
        num_samples=(opts.batch_size * opts.batches_per_epoch),
    )
    train_loader = DataLoader(train_data, batch_size=opts.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_data, batch_size=opts.batch_size)

    logger.info('Initializing Agents...')
    #################
    # Define Sender #
    #################
    logger.info('Initializing Sender...')
    encoder = Encoder(
        vocab_size=len(COMMAND_ID),
        embed_size=opts.embed_size,
        hidden_size=opts.hidden_size,
        cell=opts.cell,
    )
    decoder = Decoder_REINFORCE(
        vocab_size=opts.vocab_size,
        embed_size=opts.embed_size,
        max_length=opts.max_len,
        hidden_size=opts.hidden_size,
        cell=opts.cell,
        enable_attention=opts.enable_sender_attention,
    )
    decoder = PlusOneWrapper(decoder)
    sender = Agent(encoder, decoder)

    ###################
    # Define Receiver #
    ###################
    logger.info('Initializing Receiver...')
    encoder = Encoder(
        vocab_size=opts.vocab_size + 1,  # "+1" due to PlusOneWrapper
        embed_size=opts.embed_size,
        hidden_size=opts.hidden_size,
        cell=opts.cell,
    )
    decoder = Decoder_REINFORCE(
        vocab_size=len(COMMAND_ID),
        embed_size=opts.embed_size,
        max_length=max(map(len, df["command_ids"])) - 1,
        hidden_size=opts.hidden_size,
        cell=opts.cell,
        enable_attention=opts.enable_receiver_attention,
    )
    recver = Agent(encoder, decoder)

    ###############
    # Define Game #
    ###############
    logger.info('Initializing Game...')
    game = SingleGame(
        sender=sender,
        recver=recver,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        recver_entropy_coeff=opts.receiver_entropy_coeff,
        loss=LossRR(),
    )
    logger.info(
        f'sender.encoder {sender.encoder.__class__.__name__}, '
        f'sender.decoder {sender.decoder.__class__.__name__}, '
        f'receiver.encoder {recver.encoder.__class__.__name__}, '
        f'receiver.decoder {recver.decoder.__class__.__name__}'
    )

    logger.info('Defining Optimizer...')
    optimizer = core.build_optimizer(game.parameters())

    logger.info('Defining Callbacks...')
    callbacks: List[core.Callback] = [
        Metrics(df, opts.device, freq=0),
        DumpCorpus(df, opts.device, freq=0),
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        core.EarlyStopperAccuracy(opts.early_stopping_thr),
    ]

    if opts.checkpoint_dir:
        checkpoint_name = 'model'
        callbacks.append(core.CheckpointSaver(
            checkpoint_path=opts.checkpoint_dir,
            checkpoint_freq=opts.checkpoint_freq,
            prefix=checkpoint_name,
        ))

    logger.info('Defining Trainer...')
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=valid_loader,
        callbacks=callbacks
    )

    logger.info('Start Training!')
    try:
        trainer.train(n_epochs=opts.n_epochs)
    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt caught!')

    logger.info('Terminating...')
    core.close()


if __name__ == '__main__':
    main(sys.argv[1:])
