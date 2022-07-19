from typing import List, Callable, Set, Dict

import pandas as pd
import torch
from torch.nn.functional import one_hot  # type: ignore
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
        msgs: List[List[int, ...]] = preprocessed_corpus[CorpusKey.sentence].tolist()  # type: ignore
        sems: List[Sem] = preprocessed_corpus[CorpusKey.semantics].tolist()  # type: ignore

        total_len = max(len(s) for s in msgs)
        tensor_msgs: List[torch.Tensor] = [
            one_hot(
                torch.as_tensor(x + [0] * (total_len - len(x)), dtype=torch.long),
                num_classes=vocab_size + 1,
            ).permute(1, 0) for x in msgs
        ]
        tensor_sems = [convert_semantics_to_tensors_for_tre_model(m) for m in sems]
        dataset = list(zip(tensor_msgs, tensor_sems))

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

            for _ in range(n_epochs):
                optimizer.zero_grad()
                loss = sum(
                    (objective.forward(mng, msg) for msg, mng in dataset),
                    start=torch.as_tensor(0, dtype=torch.float),
                ) / len(dataset)
                loss.backward()  # type: ignore
                optimizer.step()

            objective.eval()
            with torch.no_grad():
                tre = sum(
                    (objective.forward(mng, msg) for msg, mng in dataset),
                    start=torch.as_tensor(0, dtype=torch.float),
                ) / sum(len(m) for m in msgs)

            metric[key].append(tre.item())
    return {Metric.tre.value: metric}
