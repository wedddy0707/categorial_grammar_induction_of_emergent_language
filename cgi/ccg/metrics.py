import time
from typing import List, Set, Dict, Optional
from collections import Counter

import pandas as pd
import itertools

from ..io import make_logger
from ..corpus import basic_preprocess_of_corpus_df, TargetLanguage, CorpusKey, Metric
from ..topsim.topsim import compute_topsim
from .train import Dataset, test, train
from .lexitem import CategorialGrammarRule
# from .train import surface_realization

logger = make_logger(__name__)


def metrics_of_induced_categorial_grammar(
    corpus: pd.DataFrame,
    target_langs: Set[TargetLanguage],
    swap_count: int = 1,
    n_epochs: int = 20,
    n_trains: int = 1,
    lr: float = 0.1,
    c: float = 1e-5,
    beam_size: Optional[int] = 10,
    vocab_size: int = 1,
    show_train_progress: bool = False,
    show_lexicon: bool = False,
    show_parses: bool = False,
):
    metric: Dict[str, Dict[str, List[Optional[float]]]] = {
        Metric.cgf.value: {},
        Metric.cgl.value: {},
        Metric.cgt.value: {},
        Metric.cgs.value: {},
        Metric.ibm_model_1.value: {},
        Metric.fb_ratio.value: {},
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
        trn_dev_split = preprocessed_corpus[split == "train"]
        dev_size = int(len(trn_dev_split) / 9)
        trn_split = trn_dev_split.head(len(trn_dev_split) - dev_size)
        dev_split = trn_dev_split.tail(dev_size)
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
        metric[Metric.cgt.value][key] = []
        metric[Metric.cgs.value][key] = []
        metric[Metric.ibm_model_1.value][key] = []
        metric[Metric.fb_ratio.value][key] = []

        for _ in range(n_trains):
            #########
            # Train #
            #########
            train_info = train(
                trn_dataset,
                dev_dataset,
                n_epochs=n_epochs,
                lr=lr,
                c=c,
                beam_size=beam_size,
                show_progress=show_train_progress,
            )
            parser = train_info.best_parser
            logger.info(
                f"Best Parser at epoch {train_info.best_epoch}"
            )
            ########
            # Test #
            ########
            trn_eval = test(parser, trn_dataset, beam_size=beam_size)
            dev_eval = test(parser, dev_dataset, beam_size=beam_size)
            tst_eval = test(parser, tst_dataset, beam_size=beam_size)
            logger.info(
                "Precition, Recall, & F1-score\n"
                f"For train data: precision={trn_eval.precision}, recall={trn_eval.recall}, F1={trn_eval.f1score}.\n"
                f"For dev data:   precision={dev_eval.precision}, recall={dev_eval.recall}, F1={dev_eval.f1score}.\n"
                f"For test data:  precision={tst_eval.precision}, recall={tst_eval.recall}, F1={tst_eval.f1score}."
            )
            # tst_realization = surface_realization(parser, tst_dataset, beam_size=beam_size)
            # logger.info(
            #     "\n".join(
            #         ("Surface Realization Results for Test Data",) + tst_realization.logging_infos
            #     )
            # )
            ######################
            # Some visualization #
            ######################
            if show_lexicon:
                logger.info(
                    "Show Lexicon\n" + "\n".join([
                        "{} & {}".format(
                            item.to_latex(bracket_with_dallers=True),
                            parser.params[item],
                        )
                        for item in sorted(
                            parser.lexicon,
                            reverse=True,
                            key=lambda x: parser.params[x],
                        )
                    ])
                )
            if show_parses:
                logger.info("Parses for train data:\n" + "\n\n".join(trn_eval.visualized_top_score_parses))
                logger.info("Parses for dev data:\n" + "\n\n".join(dev_eval.visualized_top_score_parses))
                logger.info("Parses for test data:\n" + "\n\n".join(tst_eval.visualized_top_score_parses))
            ###########
            # Metrics #
            ###########
            metric[Metric.cgf.value][key].append(tst_eval.f1score)
            metric[Metric.cgl.value][key].append(len(parser.lexicon) / len(trn_dataset))
            metric[Metric.cgt.value][key].append(
                compute_topsim(trn_eval.word_sequences, [i for _, _, i in trn_dataset])
            )
            # metric[Metric.cgs.value][key].append(
            #     sum(tst_realization.edit_distances) / len(tst_realization.edit_distances)
            # )

            translation_scores = [
                parser.init_param_factory.model.score(msg, lgc.constant_nodes())
                for msg, lgc, _ in trn_dataset
            ]
            metric[Metric.ibm_model_1.value][key].append(
                sum(translation_scores) / len(translation_scores)
            )

            count_of_applied_rules = Counter(
                itertools.chain.from_iterable(
                    parse.applied_rules
                    for parse in trn_eval.top_score_parses
                    if parse is not None
                )
            )
            count_fa = count_of_applied_rules[CategorialGrammarRule.forward_application_rule]
            count_ba = count_of_applied_rules[CategorialGrammarRule.backward_application_rule]
            try:
                fb_ratio = (count_fa - count_ba) / (count_fa + count_ba)
            except ZeroDivisionError:
                fb_ratio = 0
            metric[Metric.fb_ratio.value][key].append(fb_ratio)

        end_time = time.time()
        logger.info(f"Processing time: {end_time - start_time}s")
    return metric
