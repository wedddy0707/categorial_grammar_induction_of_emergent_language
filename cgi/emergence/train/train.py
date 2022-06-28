from typing import Optional, List, Tuple, Dict, Literal
import sys
import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler

import egg.core as core

from ...semantics.semantics import SemanticsDataset, SEMANTIC_VOCAB_SIZE
from ...io import make_logger
from ..arch import Encoder
from ..arch import Decoder
from ..arch import Decoder_REINFORCE
from ..arch import Agent
from ..game import SingleGame
from ..game.loss import LossRR, LossRS
from .common_params import get_common_params
from .dump import dump_params
from .intervene import DumpCorpus, Metrics, Evaluator, PeriodicAgentResetter


def get_params(
    params: List[str],
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


def main(argv: List[str]):
    logger = make_logger("main")
    logger.info("logger initialized")

    opts = get_params(argv)

    dump_params(vars(opts))

    logger.info("Making Datasets...")
    train_data, test_data, full_data = SemanticsDataset.create(opts.max_n_predicates, opts.random_seed, opts.test_p)
    logger.info("Making Data Loaders...")
    train_sampler = RandomSampler(
        train_data,
        replacement=True,
        num_samples=(opts.batch_size * opts.batches_per_epoch),
    )
    train_loader = DataLoader(train_data, batch_size=opts.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=opts.batch_size)
    # test_loader = DataLoader(test_data, batch_size=opts.batch_size)

    logger.info("Initializing Agents...")
    #################
    # Define Sender #
    #################
    logger.info("Initializing Sender...")
    encoder = Encoder(
        vocab_size=SEMANTIC_VOCAB_SIZE,
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
    logger.info("Initializing Receiver...")
    encoder = Encoder(
        vocab_size=opts.vocab_size + 1,  # "+1" due to PlusOneWrapper
        embed_size=opts.embed_size,
        hidden_size=opts.hidden_size,
        cell=opts.cell,
    )
    decoder = Decoder_REINFORCE(
        vocab_size=SEMANTIC_VOCAB_SIZE - 1,  # "-1" due to PlusOneWrapper
        embed_size=opts.embed_size,
        max_length=full_data.max_len,
        hidden_size=opts.hidden_size,
        cell=opts.cell,
        enable_attention=opts.enable_receiver_attention,
    )
    decoder = PlusOneWrapper(decoder)
    recver = Agent(encoder, decoder)

    ###############
    # Define Game #
    ###############
    logger.info("Initializing Game...")
    game = SingleGame(
        sender=sender,
        recver=recver,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        recver_entropy_coeff=opts.receiver_entropy_coeff,
        loss=LossRR(),
    )
    logger.info(
        f"sender.encoder {sender.encoder.__class__.__name__}, "
        f"sender.decoder {sender.decoder.__class__.__name__}, "
        f"receiver.encoder {recver.encoder.__class__.__name__}, "
        f"receiver.decoder {recver.decoder.__class__.__name__}"
    )

    logger.info("Defining Optimizer...")
    optimizer = core.build_optimizer(game.parameters())

    logger.info("Defining Callbacks...")
    split_to_dataset: Dict[Literal["train", "valid", "test"], SemanticsDataset] = {
        "train": train_data,
        "test": test_data,
    }
    callbacks: List[core.Callback] = [
        Metrics(split_to_dataset, opts.device, freq=0),
        DumpCorpus(split_to_dataset, opts.device, freq=0),
        Evaluator(split_to_dataset, device=opts.device, freq=0),
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        core.EarlyStopperAccuracy(opts.early_stopping_thr),
        PeriodicAgentResetter(opts.sender_life_span, opts.receiver_life_span),
    ]

    if opts.checkpoint_dir:
        checkpoint_name = "model"
        callbacks.append(core.CheckpointSaver(
            checkpoint_path=opts.checkpoint_dir,
            checkpoint_freq=opts.checkpoint_freq,
            prefix=checkpoint_name,
        ))

    logger.info("Defining Trainer...")
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=valid_loader,
        callbacks=callbacks
    )

    logger.info("Start Training!")
    try:
        trainer.train(n_epochs=opts.n_epochs)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught!")

    logger.info("Terminating...")
    core.close()


if __name__ == "__main__":
    main(sys.argv[1:])
