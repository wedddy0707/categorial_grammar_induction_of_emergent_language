import json
import sys
from typing import Sequence, Dict, List

from .io import make_logger, dump_metrics, get_params
from .util import set_random_seed
from .tre.tre import metrics_of_tre
from .topsim.topsim import metrics_of_topsim
from .ccg.metrics import metrics_of_induced_categorial_grammar

logger = make_logger("main")


def main(params: Sequence[str]):
    opts = get_params(params)

    assert opts.min_epoch_in_log_file is not None
    assert opts.max_epoch_in_log_file is not None

    logger.info(json.dumps(vars(opts), indent=4, default=repr))

    set_random_seed(opts.random_seed)

    for epoch in range(opts.min_epoch_in_log_file, opts.max_epoch_in_log_file + 1):
        logger.info(f"Read corpus at epoch {epoch}")
        try:
            corpus = opts.log_file.extract_corpus(epoch)
        except KeyError as e:
            logger.info(f"Corpus at epoch {e} is not available.")
            continue

        vocab_size: int = opts.log_file.extract_config().vocab_size

        m: Dict[str, Dict[str, List[float]]] = {}

        logger.info("Compute TRE")
        m_tre = metrics_of_tre(
            corpus=corpus,
            vocab_size=vocab_size,
            target_langs=set(opts.target_language),
            swap_count=opts.swap_count,
            n_epochs=opts.n_epochs_for_tre,
            n_trains=opts.n_trains_for_tre,
            lr=opts.lr_for_tre,
        )
        m.update(m_tre)

        logger.info("Compute TopSim")
        m_topsim = metrics_of_topsim(
            corpus=corpus,
            vocab_size=vocab_size,
            target_langs=set(opts.target_language),
            swap_count=opts.swap_count,
        )
        m.update(m_topsim)

        logger.info("Compute CGF & CGL")
        m_cgi = metrics_of_induced_categorial_grammar(
            corpus,
            target_langs=set(opts.target_language),
            swap_count=opts.swap_count,
            vocab_size=vocab_size,
            n_trains=opts.n_trains_for_metric,
            show_lexicon=True,
        )
        m.update(m_cgi)

        logger.info(json.dumps(m, indent=4))
        dump_metrics(m, opts.log_file, epoch)


if __name__ == "__main__":
    main(sys.argv[1:])
