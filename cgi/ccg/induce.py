import argparse
import json
import random
import sys
from typing import Optional, Sequence

import numpy.random as nprandom
import torch

from ..io import LogFile, make_logger
from .metrics import metrics_of_induced_categorial_grammar

logger = make_logger('main')


def s2b(s: str) -> bool:
    if s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if s.lower() not in ('false', 'f', 'no', 'n', '0'):
        logger.warning(
            f'Unknown choice {s} for some boolean option. '
            'Regard it as false.')
    return False


def get_params(
    params: Sequence[str],
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.Namespace:
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, required=True)
    parser.add_argument('--learning_target', type=str, default='emergent',
                        choices=(
                            'input',
                            'emergent',
                            'shuffled',
                            'random',
                            'adjacent_swapped',
                        ))
    parser.add_argument('--swap_count', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--generate_sentence', type=s2b, default=False)
    parser.add_argument('--use_tqdm', type=s2b, default=False)
    parser.add_argument('--show_lexicon', type=s2b, default=False)
    parser.add_argument('--show_parses', type=s2b, default=False)
    parser.add_argument('--overwrite', type=s2b, default=False)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--n_epochs_for_induction', type=int, default=10)
    parser.add_argument('--n_trains_for_induction', type=int, default=1)
    parser.add_argument('--min_epoch_of_corpus', type=int, default=1)
    parser.add_argument('--max_epoch_of_corpus', type=int, default=1000)
    opts = parser.parse_args(params)

    return opts


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    nprandom.seed(seed)


def main(params: Sequence[str]) -> None:
    logger = make_logger('main')
    opts = get_params(params)
    logger.info(json.dumps(vars(opts), indent=4))

    set_random_seed(opts.random_seed)

    logger.info('reading log file...')
    log_file = LogFile(opts.corpus_path)

    if opts.min_epoch_of_corpus > log_file.max_epoch:
        logger.warning(
            'opts.min_epoch_of_corpus > log_file.max_epoch. '
            'Automatically set opts.min_epoch_of_corpus = log_file.max_epoch.'
        )
        opts.min_epoch_of_corpus = log_file.max_epoch

    for epoch in range(
        opts.min_epoch_of_corpus,
        1 + min(opts.max_epoch_of_corpus, log_file.max_epoch),
    ):
        logger.info(f'reading corpus at epoch {epoch}...')
        corpus = log_file.extract_corpus(epoch)
        logger.info('inducing categorial grammar...')
        m = metrics_of_induced_categorial_grammar(
            corpus,
            learning_target=opts.learning_target,
            swap_count=opts.swap_count,
            n_epochs=opts.n_epochs_for_induction,
            n_trains=opts.n_trains_for_induction,
            vocab_size=log_file.extract_config().vocab_size,
            use_tqdm=opts.use_tqdm,
            show_lexicon=opts.show_lexicon,
            show_parses=opts.show_parses,
        )
        logger.info(json.dumps(m, indent=4))
        log_file.insert_metrics(epoch, m)
    if opts.overwrite:
        log_file.write()


if __name__ == '__main__':
    main(sys.argv[1:])
