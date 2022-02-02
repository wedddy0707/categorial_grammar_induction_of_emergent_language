import time
from collections import defaultdict
from typing import Any, List, Literal, Optional

import pandas as pd

from ..io import make_logger
from ..util import basic_preprocess_of_corpus_df
from .lexitem import sem_of_command
from .train import Dataset, test, train

logger = make_logger(__name__)


def calc_f_score(p: Optional[float], r: float):
    if p is None:
        f = None
    elif p + r == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    return f


def metrics_of_induced_categorial_grammar(
    corpus: pd.DataFrame,
    learning_target: Literal['input', 'emergent', 'shuffled', 'adjacent_swapped', 'random'] = 'emergent',  # noqa: E501
    swap_count:            int = 1,
    n_epochs:              int = 100,
    n_trains:              int = 1,
    lr:                  float = 0.1,
    c:                   float = 0.0,
    vocab_size:            int = 1,
    use_tqdm:             bool = False,
    show_train_progress:  bool = False,
    show_lexicon:         bool = False,
    show_parses:          bool = False,
):
    start_time = time.time()
    ##############
    # Preprocess #
    ##############
    corpus = basic_preprocess_of_corpus_df(
        corpus,
        learning_target=learning_target,
        swap_count=swap_count,
        vocab_size=vocab_size,
    )
    # convert meanings to logical_forms
    corpus['logical_form'] = corpus['meaning'].map(sem_of_command)
    # dataset
    train_split = corpus[corpus['split'] == 'train']
    valid_split = corpus[corpus['split'] == 'valid']
    train_dataset: Dataset = tuple(zip(train_split['sentence'], train_split['logical_form'], train_split['input']))
    valid_dataset: Dataset = tuple(zip(valid_split['sentence'], valid_split['logical_form'], valid_split['input']))

    metric: 'defaultdict[str, List[Any]]' = defaultdict(list)
    # keys for metric
    suffix = learning_target
    if learning_target == 'adjacent_swapped':
        suffix += f'_{swap_count}'
    elif learning_target == 'random':
        suffix += f'_{vocab_size}'
    TRAIN_PRECISION = f'train_precision_icg_{suffix}'
    TRAIN_RECALL = f'train_recall_icg_{suffix}'
    TRAIN_FSCORE = f'train_fscore_icg_{suffix}'
    TEST_PRECISION = f'test_precision_icg_{suffix}'
    TEST_RECALL = f'test_recall_icg_{suffix}'
    TEST_FSCORE = f'test_fscore_icg_{suffix}'
    LEXICON_SIZE = f'lexicon_size_icg_{suffix}'
    CGF = f"CGF_{suffix}"
    CGL = f"CGL_{suffix}"
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
            use_tqdm=use_tqdm
        )
        ########
        # Test #
        ########
        train_precision, train_recall, train_parses = test(
            parser, train_dataset)
        valid_precision, valid_recall, valid_parses = test(
            parser, valid_dataset)
        ######################
        # Some visualization #
        ######################
        if show_lexicon:
            for item in sorted(
                parser.lexicon,
                key=lambda x: parser.params[x], reverse=True
            ):
                print(f'{str(item): <60} SCORE={parser.params[item]}')
        if show_parses:
            print('parses for train data:')
            for p in train_parses:
                print(p, '\n')
            print('parses for test data:')
            for p in valid_parses:
                print(p, '\n')
        ###########
        # Metrics #
        ###########
        train_f = calc_f_score(train_precision, train_recall)
        valid_f = calc_f_score(valid_precision, valid_recall)
        lexicon_size = len(parser.lexicon)
        metric[TRAIN_PRECISION].append(train_precision)
        metric[TRAIN_RECALL].append(train_recall)
        metric[TRAIN_FSCORE].append(train_f)
        metric[TEST_PRECISION].append(valid_precision)
        metric[TEST_RECALL].append(valid_recall)
        metric[TEST_FSCORE].append(valid_f)
        metric[LEXICON_SIZE].append(lexicon_size)
        metric[CGF].append(valid_f)
        metric[CGL].append(lexicon_size)
    end_time = time.time()
    logger.info(f'processing time: {end_time - start_time}')
    return metric
