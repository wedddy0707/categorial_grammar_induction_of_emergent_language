import time
from typing import List, Set, Dict, Optional

import pandas as pd

from ..io import make_logger
from ..corpus import basic_preprocess_of_corpus_df, TargetLanguage, CorpusKey, Metric
from .train import Dataset, test, train

logger = make_logger(__name__)


def metrics_of_induced_categorial_grammar(
    corpus: pd.DataFrame,
    target_langs: Set[TargetLanguage],
    swap_count: int = 1,
    n_epochs: int = 100,
    n_trains: int = 1,
    lr: float = 0.1,
    c: float = 0.1,
    vocab_size: int = 1,
    use_tqdm: bool = False,
    show_train_progress: bool = False,
    show_lexicon: bool = False,
    show_parses: bool = False,
):
    metric: Dict[str, Dict[str, List[Optional[float]]]] = {
        Metric.cgf: {},
        Metric.cgl: {},
    }

    for target_lang in sorted(target_langs, key=(lambda x: x.value)):
        start_time = time.time()
        logger.info(f"Target Language: {target_lang}")
        ##############
        # Preprocess #
        ##############
        preprocessed_corpus = basic_preprocess_of_corpus_df(
            corpus,
            target_lang=target_lang,
            swap_count=swap_count,
            vocab_size=vocab_size,
        )
        # dataset
        train_split = preprocessed_corpus[corpus[CorpusKey.split] == "train"]
        valid_split = preprocessed_corpus[corpus[CorpusKey.split] == "test"]
        test_split = preprocessed_corpus[corpus[CorpusKey.split] == "test"]
        train_dataset: Dataset = tuple(
            zip(train_split[CorpusKey.sentence], train_split[CorpusKey.semantics], train_split[CorpusKey.input])
        )
        valid_dataset: Dataset = tuple(
            zip(valid_split[CorpusKey.sentence], valid_split[CorpusKey.semantics], valid_split[CorpusKey.input])
        )
        test_dataset: Dataset = tuple(
            zip(test_split[CorpusKey.sentence], test_split[CorpusKey.semantics], test_split[CorpusKey.input])
        )

        if target_lang == TargetLanguage.adjacent_swapped:
            key = "{}_{}".format(target_lang.value, swap_count)
        else:
            key = target_lang.value

        metric[Metric.cgf][key] = []
        metric[Metric.cgl][key] = []

        for _ in range(n_trains):
            #########
            # Train #
            #########
            parser = train(
                train_dataset,
                valid_dataset,
                n_epochs=n_epochs,
                lr=lr,
                c=c,
                show_progress=show_train_progress,
            )
            ########
            # Test #
            ########
            train_precision, train_recall, train_f1, train_parses = test(parser, train_dataset)
            valid_precision, valid_recall, valid_f1, valid_parses = test(parser, valid_dataset)
            test_precision, test_recall, test_f1, test_parses = test(parser, test_dataset)
            logger.info(
                "\n"
                f"For train data:      precision={train_precision}, recall={train_recall}, F1={train_f1}.\n"
                f"For validation data: precision={valid_precision}, recall={valid_recall}, F1={valid_f1}.\n"
                f"For train data:      precision={test_precision},  recall={test_recall},  F1={test_f1}."
            )
            ######################
            # Some visualization #
            ######################
            if show_lexicon:
                for item in sorted(parser.lexicon, key=lambda x: parser.params[x], reverse=True):
                    print(f"{str(item): <60} SCORE={parser.params[item]}")
            if show_parses:
                print("parses for train data:")
                for p in train_parses:
                    print(p, "\n")
                print("parses for validation data:")
                for p in valid_parses:
                    print(p, "\n")
                print("parses for test data:")
                for p in test_parses:
                    print(p, "\n")
            ###########
            # Metrics #
            ###########
            metric[Metric.cgf.value][key].append(test_f1)
            metric[Metric.cgl.value][key].append(len(parser.lexicon) / len(train_dataset))
        end_time = time.time()
        logger.info(f"processing time: {end_time - start_time}")
    return metric
