import argparse
import json
import sys
from typing import Optional, Sequence

from ..io import LogFile, make_logger
from ..util import set_random_seed
from .tre import metrics_of_tre

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
    parser.add_argument('--corpus_path', type=str,   required=True)
    parser.add_argument('--learning_target', type=str, default='emergent',
                        choices=(
                            'input',
                            'emergent',
                            'shuffled',
                            'random',
                            'adjacent_swapped',
                        ))
    parser.add_argument('--swap_count', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--overwrite', type=s2b, default=False)
    parser.add_argument('--random_seed', type=int,   default=1)
    parser.add_argument('--n_epochs_for_tre', type=int, default=1000)
    parser.add_argument('--n_trains_for_tre', type=int, default=1)
    parser.add_argument('--min_epoch_of_corpus', type=int, default=1)
    parser.add_argument('--max_epoch_of_corpus', type=int, default=1000)
    opts = parser.parse_args(params)
    return opts


def main(
    params: Sequence[str],
):
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
        logger.info('Calculating TRE...')
        m = metrics_of_tre(
            corpus,
            vocab_size=vocab_size,
            learning_target=opts.learning_target,
            swap_count=opts.swap_count,
            n_epochs=opts.n_epochs_for_tre,
            n_trains=opts.n_trains_for_tre,
            lr=opts.lr,
        )
        logger.info(json.dumps(m, indent=4))
        log_file.insert_metrics(epoch, m)
    if opts.overwrite:
        log_file.write()


if __name__ == '__main__':
    main(sys.argv[1:])
