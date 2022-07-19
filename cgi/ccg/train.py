import itertools
import json
import logging
import random
import time
import copy
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..io import make_logger
from .genlex import newlex
from .lexitem import BasicCat, LexItem, Sem
from .loglinear_ccg import LogLinearCCG

logger = make_logger(__name__)
empty_lexicon: Set[LexItem] = set()

Lexicon = Set[LexItem]
Sentence = Tuple[Hashable, ...]
Dataset = Sequence[Tuple[Sentence, Sem, Sequence[int]]]


class InitParamFactory:
    def __init__(
        self,
        dataset: Dataset,
        scale: float,
        default: float,
    ):
        self.scale = scale
        self.default = default

        ngram_len_to_cooccurences: Dict[int, List[Tuple[Sentence, Sem]]] = {}
        for msg, lgc, _ in dataset:
            for node in lgc.constant():
                for n in range(1, len(msg) + 1):
                    for i in range(0, len(msg) - n + 1):
                        if n not in ngram_len_to_cooccurences:
                            ngram_len_to_cooccurences[n] = []
                        ngram_len_to_cooccurences[n].append((msg[i:i + n], node))
        self.ngram_len_to_cooccurences = {
            k: Counter(v) for k, v in ngram_len_to_cooccurences.items()
        }
        self.ngram_len_to_log_of_total_count_of_cooccurences = {
            k: np.log2(sum(v.values())) for k, v
            in self.ngram_len_to_cooccurences.items()
        }
        self.ngram_to_log_of_total_count: Dict[Tuple[Hashable, ...], int] = dict()

    def __call__(self, key: LexItem) -> float:
        if not key.sem.constant():
            return self.default
        ngram = key.pho

        cooccurences = self.ngram_len_to_cooccurences[len(ngram)]
        log_all_count = self.ngram_len_to_log_of_total_count_of_cooccurences[len(ngram)]

        if ngram in self.ngram_to_log_of_total_count:
            log_of_ngram_count = self.ngram_to_log_of_total_count[ngram]
        else:
            log_of_ngram_count = np.log2(sum(count for (x, _), count in cooccurences.items() if x == ngram))
            self.ngram_to_log_of_total_count[ngram] = log_of_ngram_count

        pmis: List[float] = []
        for node in key.sem.constant():
            log_of_node_count = np.log2(sum(count for (_, y), count in cooccurences.items() if y == node))
            pmis.append(
                np.log2(cooccurences[ngram, node]) - log_of_ngram_count - log_of_node_count + log_all_count
            )
        return float(np.average(pmis)) * self.scale


def compute_f1_score(p: Optional[float], r: float):
    if p is None:
        f = None
    elif p + r == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    return f


def train(
    trn_dataset: Dataset,
    dev_dataset: Dataset,
    n_epochs: int = 100,
    lr: float = 0.1,
    c: float = 0,
    beam_size: Optional[int] = 10,
    show_progress: bool = False,
):
    logging_level = logger.level
    if not show_progress:
        logger.setLevel(logging.FATAL)

    ####################
    # initialize model #
    ####################
    logger.info("initializing model")
    parser = LogLinearCCG(
        lr=lr,
        c=c,
        init_param_factory=InitParamFactory(trn_dataset, scale=1, default=1),
    )
    ######################
    # initialize lexicon #
    ######################
    logger.info("initializing lexicon")
    for msg, lgc, _ in trn_dataset:
        parser.update_lexicon(LexItem(cat=BasicCat.S, sem=lgc, pho=tuple(msg)))
    ############
    # Training #
    ############
    logger.info("start training")
    best_dev_fscore = 0
    best_parser = copy.deepcopy(parser)
    for epoch in range(1, n_epochs + 1):
        logger.info(
            f"Lexicon size is {len(parser.lexicon)} "
            f"at the begging of epoch {epoch}"
        )
        ########################
        # Parameter Estimation #
        ########################
        times = [time.time()]
        old_lexicon = parser.lexicon.copy()
        new_lexicon: Set[LexItem] = set()
        for msg, lgc, _ in random.sample(trn_dataset, len(trn_dataset)):
            first_parses = parser(
                msg,
                logical_form=lgc,
                beam_size=beam_size,
                return_only_top_score=True)
            # Splitting lexical entries
            parser.update_lexicon(
                empty_lexicon.union(*(newlex(x) for x in first_parses)))
            # Parameter Estimation
            second_parses = parser(msg, beam_size=beam_size)
            parser.zero_grad()
            parser.calc_grad(second_parses, lgc)
            parser.update_params()
            # Lexicon Pruning
            new_lexicon |= empty_lexicon.union(
                *(x.lexitems for x in first_parses))
        parser.lexicon = new_lexicon
        ########################
        # Accuracy Calculation #
        ########################
        times.append(time.time())
        trn_p, trn_r, trn_f = test(parser, trn_dataset, beam_size=beam_size)[:3]
        dev_p, dev_r, dev_f = test(parser, dev_dataset, beam_size=beam_size)[:3]
        times.append(time.time())
        if dev_f is not None and dev_f > best_dev_fscore:
            best_dev_fscore = dev_f
            best_parser = copy.deepcopy(parser)
        logger.info(json.dumps({
            "mode": "train",
            "epoch": epoch,
            "trn-p": trn_p,
            "trn-r": trn_r,
            "trn-f": trn_f,
            "dev-p": dev_p,
            "dev-r": dev_r,
            "dev-f": dev_f,
            "size": len(parser.lexicon),
            "#new lexicon": len(parser.lexicon - old_lexicon),
            "#old lexicon": len(old_lexicon - parser.lexicon),
            "times": [round(y - x, 4) for x, y in zip(times[:-1], times[1:])],
        }, indent=4))
    ##########################################
    # make prior distribution over sentences #
    ##########################################
    logger.info("make prior distribution over sentences")
    best_parser.unigram = Counter(
        itertools.chain.from_iterable(msg for msg, _, _ in trn_dataset))

    logger.setLevel(logging_level)
    return best_parser


def test(
    parser: LogLinearCCG,
    dataset: Dataset,
    beam_size: Optional[int] = 10,
):
    are_parsed: List[int] = []
    are_correct: List[int] = []
    visualized_top_score_parses: List[str] = []
    for msg, lgc, _ in dataset:
        parses = parser(msg, beam_size=beam_size)

        is_parsed = is_correct = False
        if len(parses) > 0:
            sem_score: "defaultdict[Sem, float]" = defaultdict(float)
            for parse in parses:
                sem_score[parse.item.sem] += parse.score
            top_score_sem = max(sem_score.items(), key=lambda x: x[1])[0]
            is_parsed = True
            is_correct = (top_score_sem == lgc)
        are_parsed.append(int(is_parsed))
        are_correct.append(int(is_correct))

        if len(parses) > 0:
            top_score_parse = max(parses, key=lambda x: x.score)
            dump = "({}, {}) is parsed {}.\n{}".format(
                msg,
                lgc,
                "correctly" if top_score_parse.item.sem == lgc else "wrongly",
                max(parses, key=lambda x: x.score).visualize(),
            )
            visualized_top_score_parses.append(dump)
        else:
            dump = "({}, {}) is not parsed.".format(msg, lgc)
            visualized_top_score_parses.append(dump)
    try:
        precision = sum(are_correct) / sum(are_parsed)
    except ZeroDivisionError:
        precision = None
    recall = sum(are_correct) / len(are_correct)

    return (
        precision,
        recall,
        compute_f1_score(precision, recall),
        visualized_top_score_parses,
    )
