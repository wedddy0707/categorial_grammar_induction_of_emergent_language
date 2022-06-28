import itertools
import json
import logging
import random
import time
import copy
from collections import Counter, defaultdict
from typing import (Any, Dict, Hashable, Iterable, List, Optional, Sequence,
                    Set, Tuple)

import numpy as np
from tqdm import tqdm

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
        scale: float = 1,
        default: float = 0,
    ):
        self.scale = scale
        self.default = default

        cooccur_list_per_ngram: 'defaultdict[int, List[Tuple[Sentence, Sem]]]' = defaultdict(list)  # noqa: E501
        for msg, lgc, _ in dataset:
            for cnst in lgc.constant():
                for n in range(1, len(msg) + 1):
                    for i in range(0, len(msg) - n + 1):
                        cooccur_list_per_ngram[n].append((msg[i:i + n], cnst))
        self.cooccur_per_ngram = {
            k: Counter(v) for k, v
            in cooccur_list_per_ngram.items()}
        self.log_all_count_per_ngram = {
            k: np.log2(sum(v.values())) for k, v
            in self.cooccur_per_ngram.items()}
        self.log_pho_count: Dict[Tuple[Hashable, ...], int] = dict()

    def __call__(self, key: LexItem) -> float:
        if not key.sem.constant():
            return self.default
        pho = key.pho
        n = len(pho)
        cooccur = self.cooccur_per_ngram[n]
        log_all_count = self.log_all_count_per_ngram[n]

        if pho in self.log_pho_count:  # noqa: E501
            log_pho_count = self.log_pho_count[pho]
        else:
            log_pho_count = np.log2(sum(
                cooccur[p, c] for p, c in cooccur.keys() if p == pho))
            self.log_pho_count[pho] = log_pho_count

        pmis: List[float] = []
        for cnst in key.sem.constant():
            log_sem_count = np.log2(sum(
                cooccur[p, c] for p, c in cooccur.keys() if c == cnst))
            pmis.append(
                np.log2(cooccur[pho, cnst]) - log_pho_count - log_sem_count + log_all_count
            )
        return float(np.average(pmis)) * self.scale


def make_init_param_factory(
    dataset: Dataset,
    scale: float = 1,
    default: float = 0,
):
    cooccur_list_per_ngram: 'defaultdict[int, List[Tuple[Sentence, Sem]]]' = defaultdict(list)  # noqa: E501
    for msg, lgc, _ in dataset:
        for cnst in lgc.constant():
            for n in range(1, len(msg) + 1):
                for i in range(0, len(msg) - n + 1):
                    cooccur_list_per_ngram[n].append((msg[i:i + n], cnst))

    cooccur_per_ngram = {k: Counter(v) for k, v in cooccur_list_per_ngram.items()}  # noqa: E501

    def init_param_factory(
        key: LexItem,
        cooccur_per_ngram: 'dict[int, Counter[Tuple[Sentence, Sem]]]' = cooccur_per_ngram,  # noqa: E501
        scale: float = scale,
        default: float = default,
    ):
        if not key.sem.constant():
            return default
        pho = key.pho
        cooccur = cooccur_per_ngram[len(pho)]
        log_all_count = np.log2(sum(cooccur.values()))
        log_pho_count = np.log2(sum(
            cooccur[p, c] for p, c in cooccur.keys() if p == pho))

        pmis: List[float] = []
        for cnst in key.sem.constant():
            log_sem_count = np.log2(sum(
                cooccur[p, c] for p, c in cooccur.keys() if c == cnst))
            pmis.append(
                np.log2(cooccur[pho, cnst]) - log_pho_count - log_sem_count + log_all_count
            )
        return float(np.average(pmis)) * scale
    return init_param_factory


def compute_f1_score(p: Optional[float], r: float):
    if p is None:
        f = None
    elif p + r == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    return f


def train(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    n_epochs: int = 100,
    lr: float = 0.1,
    c: float = 0,
    beam_size: Optional[int] = 10,
    use_tqdm: bool = False,
    show_progress: bool = False,
):
    logging_level = logger.level
    if not show_progress:
        logger.setLevel(logging.FATAL)

    ####################
    # initialize model #
    ####################
    logger.info('initializing model')
    parser = LogLinearCCG(
        lr=lr,
        c=c,
        init_param_factory=InitParamFactory(train_dataset),
    )
    ######################
    # initialize lexicon #
    ######################
    logger.info('initializing lexicon')
    pbar: Iterable[Tuple[Sentence, Sem, Any]] = tqdm(
        train_dataset,
        desc='Lexicon Initialization',
        disable=not use_tqdm,
    )
    for msg, lgc, _ in pbar:
        parser.update_lexicon(LexItem(cat=BasicCat.S, sem=lgc, pho=tuple(msg)))
    ############
    # Training #
    ############
    logger.info('start training')
    best_validation_fscore = 0
    best_parser = copy.deepcopy(parser)
    for epoch in range(1, n_epochs + 1):
        logger.info(
            f'Lexicon size is {len(parser.lexicon)} '
            f'at the begging of epoch {epoch}'
        )
        ########################
        # Parameter Estimation #
        ########################
        times = [time.time()]
        old_lexicon = parser.lexicon.copy()
        new_lexicon: Set[LexItem] = set()
        pbar: Iterable[Tuple[Sentence, Sem, Any]] = tqdm(
            random.sample(train_dataset, len(train_dataset)),
            desc=f'Parameter Estimation at epoch {epoch}',
            disable=not use_tqdm,
        )
        for msg, lgc, _ in pbar:
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
            # TQDM
            pbar.set_postfix(lexicon_size=len(parser.lexicon))
        parser.lexicon = new_lexicon
        ########################
        # Accuracy Calculation #
        ########################
        times.append(time.time())
        trn_p, trn_r, trn_f = test(parser, train_dataset, beam_size=beam_size)[:3]
        vld_p, vld_r, vld_f = test(parser, valid_dataset, beam_size=beam_size)[:3]
        times.append(time.time())
        if vld_f is not None and vld_f > best_validation_fscore:
            best_validation_fscore = vld_f
            best_parser = copy.deepcopy(parser)
        logger.info(json.dumps({
            'mode': 'train',
            'epoch': epoch,
            'trn-p': trn_p,
            'trn-r': trn_r,
            'trn-f': trn_f,
            'vld-p': vld_p,
            'vld-r': vld_r,
            'vld-f': vld_f,
            'size': len(parser.lexicon),
            '#new lexicon': len(parser.lexicon - old_lexicon),
            '#old lexicon': len(old_lexicon - parser.lexicon),
            'times': [round(y - x, 4) for x, y in zip(times[:-1], times[1:])],  # noqa: E501
        }, indent=4))
    ##########################################
    # make prior distribution over sentences #
    ##########################################
    logger.info('make prior distribution over sentences')
    best_parser.unigram = Counter(
        itertools.chain.from_iterable(msg for msg, _, _ in train_dataset))

    logger.setLevel(logging_level)
    return best_parser


def test(
    parser: LogLinearCCG,
    dataset: Dataset,
    beam_size: Optional[int] = 25,
    use_tqdm: bool = False,
):
    are_parsed: List[int] = []
    are_correct: List[int] = []
    visualized_top_score_parses: List[str] = []
    pbar: Iterable[Tuple[Sentence, Sem, Any]] = tqdm(
        dataset,
        desc='Precision-Recall Calculation',
        disable=not use_tqdm,
    )
    for msg, lgc, _ in pbar:
        parses = parser(msg, beam_size=beam_size)

        is_parsed = is_correct = False
        if len(parses) > 0:
            sem_score: 'defaultdict[Sem, float]' = defaultdict(float)
            for parse in parses:
                sem_score[parse.item.sem] += parse.score
            top_score_sem = max(sem_score.items(), key=lambda x: x[1])[0]
            is_parsed = True
            is_correct = (top_score_sem == lgc)
        are_parsed.append(int(is_parsed))
        are_correct.append(int(is_correct))

        if len(parses) > 0:
            top_score_parse = max(parses, key=lambda x: x.score)
            visualized_top_score_parses.append(
                f'{msg} is parsed ' + ('correctly' if top_score_parse.item.sem == lgc else 'wrongly') + '.\n' +  # noqa: E501
                max(parses, key=lambda x: x.score).visualize())
        else:
            visualized_top_score_parses.append(
                f'{msg} is not parserd.')
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
