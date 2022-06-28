import argparse
import json
import sys
from typing import Optional, Sequence

from ..io import make_logger, dump_metrics, get_params, NameSpaceForMetrics
from ..util import s2b, set_random_seed
from .metrics import metrics_of_induced_categorial_grammar

logger = make_logger("main")


def get_params_wrapper(
    params: Sequence[str],
    parser: Optional[argparse.ArgumentParser] = None,
) -> NameSpaceForMetrics:
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--use_tqdm", type=s2b, default=False)
    parser.add_argument("--show_lexicon", type=s2b, default=False)
    parser.add_argument("--show_parses", type=s2b, default=False)
    opts = get_params(params, parser)

    return opts


def main(params: Sequence[str]) -> None:
    logger = make_logger("main")
    opts = get_params_wrapper(params)

    assert opts.min_epoch_in_log_file is not None
    assert opts.max_epoch_in_log_file is not None

    logger.info(json.dumps(vars(opts), indent=4, default=repr))

    set_random_seed(opts.random_seed)

    logger.info("reading log file...")

    for epoch in range(opts.min_epoch_in_log_file, opts.max_epoch_in_log_file + 1):
        logger.info(f"reading corpus at epoch {epoch}...")
        corpus = opts.log_file.extract_corpus(epoch)
        logger.info("inducing categorial grammar...")
        m = metrics_of_induced_categorial_grammar(
            corpus,
            target_lang=opts.target_language,
            swap_count=opts.swap_count,
            n_epochs=opts.n_epochs_for_metric,
            n_trains=opts.n_trains_for_metric,
            vocab_size=opts.log_file.extract_config().vocab_size,
            use_tqdm=opts.use_tqdm,
            show_lexicon=opts.show_lexicon,
            show_parses=opts.show_parses,
            show_train_progress=True,
        )
        logger.info(json.dumps(m, indent=4))
        dump_metrics(m, opts.log_file, epoch)


if __name__ == "__main__":
    main(sys.argv[1:])
