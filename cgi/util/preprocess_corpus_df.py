from typing import (
    Literal,
    Sequence,
)
import pandas as pd
import random


def eos_remover(x: Sequence[int]):
    return tuple(x[:x.index(0)])


def shuffler(x: Sequence[int]):
    return tuple(random.sample(x, len(x)))


def basic_preprocess_of_corpus_df(
    corpus: pd.DataFrame,
    learning_target: Literal[
        'input',
        'emergent',
        'shuffled',
        'adjacent_swapped',
        'random'
    ] = 'emergent',
    swap_count:  int = 1,
    vocab_size:  int = 1,
) -> pd.DataFrame:
    assert 'message' in corpus
    assert 'meaning' in corpus
    assert 'input' in corpus
    assert 'split' in corpus
    assert learning_target != 'random' or vocab_size > 1

    def adjacent_swapper(
        x: Sequence[int],
        swap_count: int = swap_count,
    ):
        if len(x) < 2:
            return tuple(x)
        y = list(x)
        for _ in range(swap_count):
            i = random.choice(range(0, len(y) - 1))
            y[i], y[i + 1] = y[i + 1], y[i]
        return tuple(y)

    def random_sampler(
        x: Sequence[int],
        vocab_size: int = vocab_size,
    ):
        return tuple(random.sample(range(1, vocab_size), len(x)))

    if learning_target == 'input':
        corpus['sentence'] = corpus['input'].map(eos_remover)
    elif learning_target == 'emergent':
        corpus['sentence'] = corpus['message'].map(eos_remover)
    elif learning_target == 'shuffled':
        corpus['sentence'] = corpus['message'].map(eos_remover).map(shuffler)
    elif learning_target == 'adjacent_swapped':
        corpus['sentence'] = corpus['message'].map(eos_remover).map(adjacent_swapper)  # noqa: E501
    elif learning_target == 'random':
        corpus['sentence'] = corpus['message'].map(random_sampler)
    else:
        raise ValueError(f'Unknown learning_target {learning_target}')
    return corpus
