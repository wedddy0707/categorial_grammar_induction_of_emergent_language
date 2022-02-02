from collections import defaultdict
from typing import Any, List, Literal, Sequence, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..emergence.dataset import COMMAND_ID
from ..util import basic_preprocess_of_corpus_df
from .model import Composer, Objective

Meaning = Union[Sequence['Meaning'], str]
MeaningTensor = Union[Sequence['MeaningTensor'], torch.LongTensor]


def recursive_tensorization(x: Meaning) -> MeaningTensor:
    if isinstance(x, str):
        return torch.LongTensor((COMMAND_ID[x],))
    else:
        return tuple(recursive_tensorization(e) for e in x)


def metrics_of_tre(
    corpus: pd.DataFrame,
    vocab_size: int,
    learning_target: Literal[
        'emergent',
        'shuffled',
        'adjacent_swapped',
        'random',
    ] = 'emergent',
    swap_count: int = 1,
    n_epochs:   int = 1000,
    n_trains:   int = 1,
    lr:       float = 0.01,
):
    corpus = basic_preprocess_of_corpus_df(
        corpus,
        learning_target=learning_target,
        swap_count=swap_count,
        vocab_size=vocab_size,
    )
    corpus = corpus[corpus['split'] == 'train']
    stcs: Sequence[Tuple[int, ...]] = corpus['sentence'].tolist()
    mngs:         Sequence[Meaning] = corpus['meaning'].tolist()

    total_len: int = max(len(s) for s in stcs)

    tensor_stcs: Sequence[torch.LongTensor] = [
        F.one_hot(
            torch.LongTensor(tuple(x) + (0,) * (total_len - len(x))),
            num_classes=vocab_size).permute(1, 0) for x in stcs]
    tensor_mngs: Sequence[MeaningTensor] = [
        recursive_tensorization(m) for m in mngs],
    dataset = list(zip(tensor_stcs, tensor_mngs))
    data_size = len(dataset)

    metric: 'defaultdict[str, List[Any]]' = defaultdict(list)
    # keys for metric
    suffix = learning_target
    if learning_target == 'adjacent_swapped':
        suffix += f'_{swap_count}'
    elif learning_target == 'random':
        suffix += f'_{vocab_size}'
    TRE = f'tre_{suffix}'

    for _ in range(n_trains):
        composer = Composer(
            len(COMMAND_ID),
            vocab_size,
            total_len,
        )
        objective = Objective(composer)
        optimizer = optim.RMSprop(objective.parameters(), lr=lr)
        objective.train()
        for _ in range(1, 1 + n_epochs):
            optimizer.zero_grad()
            loss = torch.as_tensor(0, dtype=torch.float)
            for stc, mng in dataset:
                loss = loss + objective(mng, stc)
            loss = loss / data_size
            loss.backward()
            optimizer.step()

        tre: torch.Tensor = torch.as_tensor(0, dtype=torch.float)
        objective.eval()
        with torch.no_grad():
            for stc, mng in dataset:
                tre += objective(mng, stc)
        tre /= data_size
        metric[TRE].append(tre.item())
    return metric
