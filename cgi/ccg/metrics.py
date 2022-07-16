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
    show_train_progress: bool = False,
    show_lexicon: bool = False,
    show_parses: bool = False,
):
    metric: Dict[str, Dict[str, List[Optional[float]]]] = {
        Metric.cgf.value: {},
        Metric.cgl.value: {},
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
        split: pd.Series[str] = corpus[CorpusKey.split]
        trn_split = preprocessed_corpus[split == "train"]
        dev_split = preprocessed_corpus[split == "test"]
        tst_split = preprocessed_corpus[split == "test"]
        trn_dataset: Dataset = tuple(zip(
            trn_split[CorpusKey.sentence],   # type: ignore
            trn_split[CorpusKey.semantics],  # type: ignore
            trn_split[CorpusKey.input],      # type: ignore
        ))
        dev_dataset: Dataset = tuple(zip(
            dev_split[CorpusKey.sentence],   # type: ignore
            dev_split[CorpusKey.semantics],  # type: ignore
            dev_split[CorpusKey.input],      # type: ignore
        ))
        tst_dataset: Dataset = tuple(zip(
            tst_split[CorpusKey.sentence],   # type: ignore
            tst_split[CorpusKey.semantics],  # type: ignore
            tst_split[CorpusKey.input],      # type: ignore
        ))

        if target_lang == TargetLanguage.adjacent_swapped:
            key = "{}_{}".format(target_lang.value, swap_count)
        else:
            key = target_lang.value

        metric[Metric.cgf.value][key] = []
        metric[Metric.cgl.value][key] = []

        for _ in range(n_trains):
            #########
            # Train #
            #########
            parser = train(
                trn_dataset,
                dev_dataset,
                n_epochs=n_epochs,
                lr=lr,
                c=c,
                show_progress=show_train_progress,
            )
            ########
            # Test #
            ########
            trn_precision, trn_recall, trn_f1, trn_parses = test(parser, trn_dataset)
            dev_precision, dev_recall, dev_f1, dev_parses = test(parser, dev_dataset)
            tst_precision, tst_recall, tst_f1, tst_parses = test(parser, tst_dataset)
            logger.info(
                "\n"
                f"For train data: precision={trn_precision}, recall={trn_recall}, F1={trn_f1}.\n"
                f"For dev data:   precision={dev_precision}, recall={dev_recall}, F1={dev_f1}.\n"
                f"For test data:  precision={tst_precision}, recall={tst_recall}, F1={tst_f1}."
            )
            ######################
            # Some visualization #
            ######################
            if show_lexicon:
                logger.info(
                    "Show Lexicon\n" + "\n".join([
                        f"{str(item): <60} SCORE={parser.params[item]}"
                        for item in sorted(
                            parser.lexicon,
                            reverse=True,
                            key=lambda x: parser.params[x],
                        )
                    ])
                )
            if show_parses:
                logger.info("Parses for train data:\n" + "\n\n".join(trn_parses))
                logger.info("Parses for dev data:\n" + "\n\n".join(dev_parses))
                logger.info("Parses for test data:\n" + "\n\n".join(tst_parses))
            ###########
            # Metrics #
            ###########
            metric[Metric.cgf.value][key].append(tst_f1)
            metric[Metric.cgl.value][key].append(len(parser.lexicon) / len(trn_dataset))
        end_time = time.time()
        logger.info(f"Processing time: {end_time - start_time}s")
    return metric
