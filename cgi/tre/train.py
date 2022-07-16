import argparse
import json
import sys
from typing import Optional, Sequence

from ..io import make_logger, dump_metrics, NameSpaceForMetrics, get_params
from ..util import set_random_seed
from .tre import metrics_of_tre

logger = make_logger("main")


def get_params_wrapper(
    params: Sequence[str],
    parser: Optional[argparse.ArgumentParser] = None,
) -> NameSpaceForMetrics:
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    opts = get_params(params, parser)
    return opts


def main(
    params: Sequence[str],
):
    opts = get_params_wrapper(params)

    assert opts.min_epoch_in_log_file is not None
    assert opts.max_epoch_in_log_file is not None

    logger.info(json.dumps(vars(opts), indent=4, default=repr))

    set_random_seed(opts.random_seed)

    for epoch in range(opts.min_epoch_in_log_file, opts.max_epoch_in_log_file + 1):
        logger.info(f"reading corpus at epoch {epoch}")
        corpus = opts.log_file.extract_corpus(epoch)
        vocab_size: int = opts.log_file.extract_config().vocab_size
        logger.info("Calculating TRE...")
        m = metrics_of_tre(
            corpus,
            vocab_size=vocab_size,
            target_langs=set(opts.target_language),
            swap_count=opts.swap_count,
            n_epochs=opts.n_epochs_for_metric,
            n_trains=opts.n_trains_for_metric,
            lr=opts.lr_for_tre,
        )
        logger.info(json.dumps(m, indent=4))
        dump_metrics(m, opts.log_file, epoch)


if __name__ == "__main__":
    main(sys.argv[1:])
