import random
from typing import Sequence

import pandas as pd

from ..semantics.semantics import eval_sem_repr
from .params import CorpusKey, TargetLanguage


def eos_remover(x: Sequence[int]):
    return tuple(x[:x.index(0) if (0 in x) else None])


def shuffler(x: Sequence[int]):
    return tuple(random.sample(x, len(x)))


def basic_preprocess_of_corpus_df(
    corpus: pd.DataFrame,
    target_lang: TargetLanguage = TargetLanguage.emergent,
    swap_count: int = 1,
    vocab_size: int = 1,
) -> pd.DataFrame:
    corpus = corpus.copy()

    if target_lang == TargetLanguage.random:
        assert vocab_size > 1

    def adjacent_swapper(x: Sequence[int], swap_count: int = swap_count):
        if len(x) < 2:
            return tuple(x)
        y = list(x)
        for _ in range(swap_count):
            i = random.choice(range(0, len(y) - 1))
            y[i], y[i + 1] = y[i + 1], y[i]
        return tuple(y)

    def random_sampler(x: Sequence[int], vocab_size: int = vocab_size):
        return tuple(random.sample(range(1, vocab_size), len(x)))

    if target_lang == TargetLanguage.input:
        corpus[CorpusKey.sentence] = corpus[CorpusKey.input].map(eos_remover)
    elif target_lang == TargetLanguage.emergent:
        corpus[CorpusKey.sentence] = corpus[CorpusKey.message].map(eos_remover)
    elif target_lang == TargetLanguage.shuffled:
        corpus[CorpusKey.sentence] = corpus[CorpusKey.message].map(eos_remover).map(shuffler)
    elif target_lang == TargetLanguage.adjacent_swapped:
        corpus[CorpusKey.sentence] = corpus[CorpusKey.message].map(eos_remover).map(adjacent_swapper)
    elif target_lang == TargetLanguage.random:
        corpus[CorpusKey.sentence] = corpus[CorpusKey.message].map(random_sampler)
    else:
        raise ValueError(f"Unknown target language {target_lang}")

    corpus[CorpusKey.semantics] = corpus[CorpusKey.semantics].map(eval_sem_repr)
    return corpus
