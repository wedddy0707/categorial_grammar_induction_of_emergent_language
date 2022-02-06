import time
from collections import defaultdict
from typing import Any, List, Literal

import pandas as pd

from ..io import make_logger
from ..util import basic_preprocess_of_corpus_df
from .lexitem import sem_of_command
from .train import Dataset, test, train

logger = make_logger(__name__)


def metrics_of_induced_categorial_grammar(
    corpus: pd.DataFrame,
    learning_target: Literal['input', 'emergent', 'shuffled', 'adjacent_swapped', 'random'] = 'emergent',  # noqa: E501
    swap_count: int = 1,
    n_epochs: int = 100,
    n_trains: int = 1,
    lr: float = 0.1,
    c: float = 0.0,
    vocab_size: int = 1,
    use_tqdm: bool = False,
    show_train_progress: bool = False,
    show_lexicon: bool = False,
    show_parses: bool = False,
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
    test_split = corpus[corpus['split'] == 'test']
    train_dataset: Dataset = tuple(zip(train_split['sentence'], train_split['logical_form'], train_split['input']))
    valid_dataset: Dataset = tuple(zip(valid_split['sentence'], valid_split['logical_form'], valid_split['input']))
    test_dataset: Dataset = tuple(zip(test_split['sentence'], test_split['logical_form'], test_split['input']))

    metric: 'defaultdict[str, List[Any]]' = defaultdict(list)
    # keys for metric
    suffix = learning_target
    if learning_target == 'adjacent_swapped':
        suffix += f'_{swap_count}'
    elif learning_target == 'random':
        suffix += f'_{vocab_size}'
    TRAIN_PRECISION = f'train_precision_icg_{suffix}'
    TRAIN_RECALL = f'train_recall_icg_{suffix}'
    TRAIN_F1SCORE = f'train_f1score_icg_{suffix}'
    VALID_PRECISION = f'valid_precision_icg_{suffix}'
    VALID_RECALL = f'valid_recall_icg_{suffix}'
    VALID_F1SCORE = f'valid_f1score_icg_{suffix}'
    TEST_PRECISION = f'test_precision_icg_{suffix}'
    TEST_RECALL = f'test_recall_icg_{suffix}'
    TEST_F1SCORE = f'test_f1score_icg_{suffix}'
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
        train_precision, train_recall, train_f1, train_parses = test(
            parser, train_dataset)
        valid_precision, valid_recall, valid_f1, valid_parses = test(
            parser, valid_dataset)
        test_precision, test_recall, test_f1, train_parses = test(
            parser, test_dataset)
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
        lexicon_size = len(parser.lexicon)
        # TRAIN SCORE
        metric[TRAIN_PRECISION].append(train_precision)
        metric[TRAIN_RECALL].append(train_recall)
        metric[TRAIN_F1SCORE].append(train_f1)
        # VALIDATION SCORE
        metric[VALID_PRECISION].append(valid_precision)
        metric[VALID_RECALL].append(valid_recall)
        metric[VALID_F1SCORE].append(valid_f1)
        # TEST SCORE
        metric[TEST_PRECISION].append(test_precision)
        metric[TEST_RECALL].append(test_recall)
        metric[TEST_F1SCORE].append(test_f1)
        # LEXICON SIZE
        metric[LEXICON_SIZE].append(lexicon_size)
        # CGF & CGL
        metric[CGF].append(test_f1)
        metric[CGL].append(lexicon_size)
    end_time = time.time()
    logger.info(f'processing time: {end_time - start_time}')
    return metric
