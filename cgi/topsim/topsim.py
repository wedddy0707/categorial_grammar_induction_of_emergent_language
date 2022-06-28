import argparse
import json
import sys
from typing import Callable, List, Optional, Sequence, TypeVar, Set, Dict

import editdistance
import pandas as pd
from scipy.stats import spearmanr

from ..io import make_logger, dump_metrics, NameSpaceForMetrics, get_params
from ..corpus import basic_preprocess_of_corpus_df, TargetLanguage, CorpusKey, Metric
from ..util import set_random_seed

_T = TypeVar("_T")
logger = make_logger("main")


def get_params_wrapper(
    params: Sequence[str],
    parser: Optional[argparse.ArgumentParser] = None,
) -> NameSpaceForMetrics:
    if parser is None:
        parser = argparse.ArgumentParser()
    opts = get_params(params)
    return opts


def compute_topsim(
        dataset_1: Sequence[Sequence[_T]],
        dataset_2: Sequence[Sequence[_T]],
        distance_1: Callable[[Sequence[_T], Sequence[_T]], float] = editdistance.eval,
        distance_2: Callable[[Sequence[_T], Sequence[_T]], float] = editdistance.eval,
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
    target_langs: Set[TargetLanguage],
    swap_count: int = 1,
):
    metric: Dict[str, List[float]] = {}

    for target_lang in sorted(target_langs, key=(lambda x: x.value)):
        preprocessed_corpus = basic_preprocess_of_corpus_df(
            corpus,
            target_lang=target_lang,
            swap_count=swap_count,
            vocab_size=vocab_size,
        )
        preprocessed_corpus = preprocessed_corpus[preprocessed_corpus["split"] == "train"]

        if target_lang == TargetLanguage.adjacent_swapped:
            key = "{}_{}".format(target_lang.value, swap_count)
        else:
            key = target_lang.value

        metric[key] = [compute_topsim(preprocessed_corpus[CorpusKey.sentence].tolist(), preprocessed_corpus[CorpusKey.input].tolist())]
    return {Metric.topsim: metric}


def main(params: Sequence[str]):
    opts = get_params_wrapper(params)

    assert opts.min_epoch_in_log_file is not None
    assert opts.max_epoch_in_log_file is not None

    logger.info(json.dumps(vars(opts), indent=4, default=repr))

    set_random_seed(opts.random_seed)

    for epoch in range(opts.min_epoch_in_log_file, opts.max_epoch_in_log_file + 1):
        logger.info(f"Reading corpus at epoch {epoch}")
        corpus = opts.log_file.extract_corpus(epoch)
        vocab_size: int = opts.log_file.extract_config().vocab_size
        logger.info("Calculating TopSim...")
        m = metrics_of_topsim(
            corpus,
            vocab_size=vocab_size,
            swap_count=opts.swap_count,
            target_langs={opts.target_language},
        )
        logger.info(json.dumps(m, indent=4))
        dump_metrics(m, opts.log_file, epoch)


if __name__ == "__main__":
    main(sys.argv[1:])
