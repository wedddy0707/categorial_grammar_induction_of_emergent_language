import argparse
import json
import sys
from collections import defaultdict
from typing import (Any, Callable, Hashable, List, Literal, Optional, Sequence,
                    TypeVar)

import editdistance
import pandas as pd
from scipy.stats import spearmanr

from ..io import LogFile, make_logger
from ..util import basic_preprocess_of_corpus_df, set_random_seed

_T = TypeVar('_T')
logger = make_logger('main')

sequence_distance = editdistance.eval


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
    parser.add_argument('--corpus_path', type=str,   required=True)
    parser.add_argument('--calculating_target', type=str, default='emergent',
                        choices=(
                            'input',
                            'emergent',
                            'shuffled',
                            'random',
                            'adjacent_swapped',
                        ))
    parser.add_argument('--swap_count', type=int, default=1)
    parser.add_argument('--overwrite', type=s2b, default=False)
    parser.add_argument('--random_seed', type=int,   default=1)
    parser.add_argument('--min_epoch_of_corpus', type=int, default=1)
    parser.add_argument('--max_epoch_of_corpus', type=int, default=1000)
    opts = parser.parse_args(params)
    return opts


def calc_topsim(
        dataset_1: Sequence[Sequence[Hashable]],
        dataset_2: Sequence[Sequence[Hashable]],
        distance_1: Callable[[_T, _T], float] = sequence_distance,
        distance_2: Callable[[_T, _T], float] = sequence_distance
) -> float:
    dist_1: List[float] = list()
    dist_2: List[float] = list()
    assert len(dataset_1) == len(dataset_2)
    for i in range(len(dataset_1)):
        for j in range(i + 1, len(dataset_1)):
            dist_1.append(distance_1(dataset_1[i], dataset_1[j]))
            dist_2.append(distance_2(dataset_2[i], dataset_2[j]))
    return spearmanr(dist_1, dist_2).correlation


def metrics_of_topsim(
    corpus: pd.DataFrame,
    vocab_size: int,
    learning_target: Literal[
        'emergent',
        'shuffled',
        'adjacent_swapped',
        'random',
    ] = 'emergent',
    swap_count: int = 1,
):
    corpus = basic_preprocess_of_corpus_df(
        corpus,
        learning_target=learning_target,
        swap_count=swap_count,
        vocab_size=vocab_size,
    )
    corpus = corpus[corpus['split'] == 'train']
    metric: 'defaultdict[str, List[Any]]' = defaultdict(list)
    # keys for metric
    suffix = learning_target
    if learning_target == 'adjacent_swapped':
        suffix += f'_{swap_count}'
    elif learning_target == 'random':
        suffix += f'_{vocab_size}'
    TOPSIM = f'topsim_{suffix}'
    metric[TOPSIM].append(calc_topsim(corpus['sentence'].tolist(), corpus['input'].tolist()))
    return metric


def main(params: Sequence[str]):
    opts = get_params(params)
    logger.info(json.dumps(vars(opts), indent=4))

    set_random_seed(1)

    logger.info('reading log_file...')
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
        logger.info(f'reading corpus at epoch {epoch}')
        corpus = log_file.extract_corpus(epoch)
        vocab_size: int = log_file.extract_config().vocab_size
        logger.info('Calculating TopSim...')
        m = metrics_of_topsim(
            corpus,
            vocab_size=vocab_size,
            swap_count=opts.swap_count,
            learning_target=opts.calculating_target,
        )
        logger.info(json.dumps(m, indent=4))
        log_file.insert_metrics(epoch, m)
    if opts.overwrite:
        log_file.write()


if __name__ == '__main__':
    main(sys.argv[1:])
