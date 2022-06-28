from typing import List, Tuple, Callable, Set, Dict

import pandas as pd
import torch
from torch.nn.functional import one_hot
import torch.optim as optim

from ..semantics.semantics import Sem, convert_semantics_to_tensors_for_tre_model, SEMANTIC_PIECE_TO_ID
from ..corpus import basic_preprocess_of_corpus_df, TargetLanguage, CorpusKey, Metric
from .model import Composer, Objective

one_hot: Callable[..., torch.Tensor]


def metrics_of_tre(
    corpus: pd.DataFrame,
    vocab_size: int,
    target_langs: Set[TargetLanguage],
    swap_count: int = 1,
    n_epochs: int = 1000,
    n_trains: int = 1,
    lr: float = 0.01,
):
    metric: Dict[str, List[float]] = {}

    for target_lang in sorted(target_langs, key=(lambda x: x.value)):
        preprocessed_corpus = basic_preprocess_of_corpus_df(
            corpus,
            target_lang=target_lang,
            swap_count=swap_count,
            vocab_size=vocab_size,
        )

        preprocessed_corpus = preprocessed_corpus[preprocessed_corpus[CorpusKey.split] == "train"]
        msgs: List[Tuple[int, ...]] = preprocessed_corpus[CorpusKey.sentence].tolist()
        sems: List[Sem] = preprocessed_corpus[CorpusKey.semantics].tolist()

        total_len = max(len(s) for s in msgs)
        tensor_msgs: List[torch.Tensor] = [
            one_hot(
                torch.as_tensor(tuple(x) + (0,) * (total_len - len(x)), dtype=torch.long),
                num_classes=vocab_size + 1,
            ).permute(1, 0) for x in msgs
        ]
        tensor_sems = [convert_semantics_to_tensors_for_tre_model(m) for m in sems]
        dataset = list(zip(tensor_msgs, tensor_sems))
        data_size = len(dataset)

        if target_lang == TargetLanguage.adjacent_swapped:
            key = "{}_{}".format(target_lang.value, swap_count)
        else:
            key = target_lang.value

        metric[key] = []

        for _ in range(n_trains):
            composer = Composer(max(SEMANTIC_PIECE_TO_ID.values()), vocab_size + 1, total_len)
            objective = Objective(composer)
            optimizer = optim.RMSprop(objective.parameters(), lr=lr)
            objective.train()

            for _ in range(1, 1 + n_epochs):
                optimizer.zero_grad()
                loss = torch.as_tensor(0, dtype=torch.float)
                for msg, sem in dataset:
                    loss = loss + objective.forward(sem, msg)
                loss = loss / data_size
                loss.backward()
                optimizer.step()

            tre: torch.Tensor = torch.as_tensor(0, dtype=torch.float)
            objective.eval()
            with torch.no_grad():
                for stc, mng in dataset:
                    tre += objective(mng, stc)
            tre /= data_size

            metric[key].append(tre.item())
    return {Metric.tre: metric}
