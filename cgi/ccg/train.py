import itertools
import json
import logging
import time
import copy
import editdistance
from collections import Counter, defaultdict
from typing import Hashable, List, Optional, Sequence, Set, Tuple, NamedTuple

import numpy as np
from numpy.random import RandomState

from ..io import make_logger
from .genlex import newlex
from .lexitem import BasicCat, LexItem, Sem
from .loglinear_ccg import LogLinearCCG

logger = make_logger(__name__)

Lexicon = Set[LexItem]
Sentence = Tuple[Hashable, ...]
Dataset = Sequence[Tuple[Sentence, Sem, Sequence[int]]]


class TrainReturnInfo(NamedTuple):
    final_parser: LogLinearCCG
    best_parser: LogLinearCCG
    best_epoch: int


class TestReturnInfo(NamedTuple):
    precision: Optional[float]
    recall: float
    f1score: Optional[float]
    visualized_top_score_parses: Tuple[str, ...]
    word_sequences: Tuple[Tuple[Tuple[Hashable, ...], ...], ...]


class SurfaceRealizationReturnInfo(NamedTuple):
    realized_sentences: Tuple[Sentence, ...]
    edit_distances: Tuple[int, ...]
    logging_infos: Tuple[str, ...]


class InitParamFactory:
    def __init__(
        self,
        dataset: Dataset,
        scale: float,
        default: float,
    ):
        self.scale = scale
        self.default = default

        self.ngram_to_log_frequency = {
            k: float(np.log2(v))
            for k, v in Counter(
                msg[i:i + n]
                for msg, _, _ in dataset
                for n in range(1, len(msg) + 1)
                for i in range(0, len(msg) - n + 1)
            ).items()
        }
        self.const_to_log_frequency = {
            k: float(np.log2(v))
            for k, v in Counter(
                const
                for _, lgc, _ in dataset
                for const in lgc.constant()
            ).items()
        }
        self.ngram_const_pair_to_log_cooccurring_frequency = {
            k: float(np.log2(v))
            for k, v in Counter(
                (msg[i:i + n], const)
                for msg, lgc, _ in dataset
                for const in lgc.constant()
                for n in range(1, len(msg) + 1)
                for i in range(0, len(msg) - n + 1)
            ).items()
        }
        self.ngram_len_to_log_total_frequency = {
            k: float(np.log2(v))
            for k, v in Counter(
                n for msg, _, _ in dataset
                for n in range(1, len(msg) + 1)
            ).items()
        }

    def __call__(self, key: LexItem) -> float:
        if not key.sem.constant():
            return self.default
        ngram = key.pho
        log_total_freq = self.ngram_len_to_log_total_frequency[len(ngram)]
        log_ngram_freq = self.ngram_to_log_frequency[ngram]
        pmis: List[float] = [
            sum([
                + self.ngram_const_pair_to_log_cooccurring_frequency[ngram, const],
                - self.const_to_log_frequency[const],
                - log_ngram_freq,
                + log_total_freq,
            ]) for const in key.sem.constant()
        ]
        return float(np.average(pmis)) * self.scale


def compute_f1_score(p: Optional[float], r: float):
    if p is None:
        f = None
    elif p + r == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)
    return f


def train(
    trn_dataset: Dataset,
    dev_dataset: Dataset,
    n_epochs: int,
    lr: float,
    c: float,
    beam_size: Optional[int],
    show_progress: bool = False,
    random_seed: int = 0,
):
    logging_level = logger.level
    if not show_progress:
        logger.setLevel(logging.FATAL)

    random_state = RandomState(random_seed)

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
    parser.lexicon = {
        LexItem(cat=BasicCat.S, sem=lgc, pho=tuple(msg))
        for msg, lgc, _ in trn_dataset
    }
    ############
    # Training #
    ############
    logger.info("start training")
    best_dev_fscore: float = 0
    best_epoch: int = 0
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
        empty: Set[LexItem] = set()  # This is just for type hinting
        # Splitting lexical entries
        tmp_lexicon: Set[LexItem] = set()
        for msg, lgc, _ in trn_dataset:
            first_parses = parser.parse(msg, logical_form=lgc, beam_size=beam_size)
            if len(first_parses) > 0:
                first_parses_top_score = max(p.score for p in first_parses)
                top_score_first_parses = filter(lambda x: x.score == first_parses_top_score, first_parses)
                tmp_lexicon |= empty.union(*(newlex(x) for x in top_score_first_parses))
        parser.lexicon = tmp_lexicon
        # Parameter Estimation
        for msg, lgc, _ in map(
            lambda i: trn_dataset[i],
            random_state.choice(  # type: ignore
                len(trn_dataset),
                len(trn_dataset),
                replace=False,
            ),
        ):
            parser.zero_grad()
            second_parses = parser.parse(msg, beam_size=beam_size)
            parser.calc_grad(second_parses, lgc)
            parser.update_params()
        # Lexicon Pruning
        new_lexicon: Set[LexItem] = set()
        for msg, lgc, _ in trn_dataset:
            third_parses = parser.parse(msg, logical_form=lgc, beam_size=beam_size)
            if len(third_parses) > 0:
                third_parses_top_score = max(p.score for p in third_parses)
                top_score_third_parses = filter(lambda x: x.score == third_parses_top_score, third_parses)
                new_lexicon |= empty.union(*(x.lexitems for x in top_score_third_parses))
        parser.lexicon = new_lexicon
        ########################
        # Accuracy Calculation #
        ########################
        times.append(time.time())
        trn_p, trn_r, trn_f = test(parser, trn_dataset, beam_size=beam_size)[:3]
        dev_p, dev_r, dev_f = test(parser, dev_dataset, beam_size=beam_size)[:3]
        times.append(time.time())
        if dev_f is None:
            pass
        elif ((dev_f > best_dev_fscore) or (dev_f == best_dev_fscore and len(parser.lexicon) < len(best_parser.lexicon))):
            best_dev_fscore = dev_f
            best_epoch = epoch
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
    return TrainReturnInfo(
        final_parser=parser,
        best_parser=best_parser,
        best_epoch=best_epoch,
    )


def test(
    parser: LogLinearCCG,
    dataset: Dataset,
    beam_size: Optional[int],
):
    are_parsed: List[int] = []
    are_correct: List[int] = []

    visualized_top_score_parses: List[str] = []
    word_sequences: List[Tuple[Tuple[Hashable, ...], ...]] = []

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
            word_sequences.append(top_score_parse.word_sequence())
        else:
            dump = "({}, {}) is not parsed.".format(msg, lgc)
            visualized_top_score_parses.append(dump)
            word_sequences.append((msg,))
    try:
        precision = sum(are_correct) / sum(are_parsed)
    except ZeroDivisionError:
        precision = None
    recall = sum(are_correct) / len(are_correct)

    return TestReturnInfo(
        precision=precision,
        recall=recall,
        f1score=compute_f1_score(precision, recall),
        visualized_top_score_parses=tuple(visualized_top_score_parses),
        word_sequences=tuple(word_sequences),
    )


def surface_realization(
    parser: LogLinearCCG,
    dataset: Dataset,
    beam_size: Optional[int],
    give_up_normalization: bool = True,
):
    best_sentences: List[Sentence] = []
    edit_distances: List[int] = []
    logging_infos: List[str] = []

    for gold_sentence, meaning_representation, _ in dataset:
        sentences, histories = parser.apply_transduction(meaning_representation)
        if len(sentences) == 0:
            best_sentences.append(())
            edit_distances.append(len(gold_sentence))
            logging_infos.append(
                f"The surface of logical Form {meaning_representation} is not realized. "
                f"The gold surface is {gold_sentence}."
            )
            continue
        sentence_probs = [
            parser.unigram_model(x) for x in sentences
        ]
        if give_up_normalization:
            feature_scores = [
                sum(parser.params[lexitem] for lexitem in h) for h in histories
            ]
            best_sentence = max(
                zip(sentences, feature_scores, sentence_probs),
                key=(lambda x: np.exp(x[1]) * x[2]),
            )[0]
        else:
            parse_lists = [
                parser.parse(x, beam_size=beam_size)
                for x in sentences
            ]
            sentence_conditional_probs = [
                parser.prob_of_target_logical_form(parse_list, meaning_representation)
                for parse_list in parse_lists
            ]
            best_sentence = max(
                zip(sentences, sentence_conditional_probs, sentence_probs),
                key=(lambda x: x[1] * x[2]),
            )[0]
        best_sentences.append(best_sentence)
        edit_distances.append(editdistance.eval(best_sentence, gold_sentence))
        correctness = "correct" if best_sentence == gold_sentence else "wrong"
        logging_infos.append(
            f"The realized surface of logical Form {meaning_representation} is {best_sentence}. "
            f"It is {correctness}. "
            f"The gold surface is {gold_sentence}."
        )
    return SurfaceRealizationReturnInfo(
        realized_sentences=tuple(best_sentences),
        edit_distances=tuple(edit_distances),
        logging_infos=tuple(logging_infos),
    )
