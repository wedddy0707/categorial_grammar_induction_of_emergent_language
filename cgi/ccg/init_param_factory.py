from typing import Sequence, Hashable, Tuple, TypeVar, Generic
from collections import Counter, defaultdict
from itertools import product

from .lexitem import LexItem
from ..semantics.semantics import Sem


T_src = TypeVar("T_src", bound=Hashable)
T_trg = TypeVar("T_trg", bound=Hashable)


class IBMModelOne(Generic[T_src, T_trg]):
    dataset: Sequence[Tuple[Sequence[T_src], Sequence[T_trg]]]
    table: "defaultdict[Tuple[T_src, T_trg], float]"

    def __init__(
        self,
        dataset: Sequence[Tuple[Sequence[T_src], Sequence[T_trg]]],
        initial_param: float = 1,
        epsilon: float = 1e-5,
    ) -> None:
        self.dataset = dataset
        self.table = defaultdict(lambda: initial_param)
        self.train(epsilon=epsilon)

    def train(self, epsilon: float):
        diff = epsilon + 1
        while diff >= epsilon:
            counters = [
                {
                    (s, t): cnt * self.table[s, t] / sum(self.table[s, t_] for t_ in trg_sent)
                    for (s, t), cnt in Counter(product(src_sent, trg_sent)).items()
                }
                for src_sent, trg_sent in self.dataset
            ]
            trg_token_to_lagrange_multiplier: "defaultdict[T_trg, float]" = defaultdict(float)
            for counter in counters:
                for (_, t), v in counter.items():
                    trg_token_to_lagrange_multiplier[t] += v
            new_table: "defaultdict[Tuple[T_src, T_trg], float]" = defaultdict(lambda: 0)
            for counter in counters:
                for k, v in counter.items():
                    new_table[k] += v
            for s, t in new_table.keys():
                new_table[s, t] /= trg_token_to_lagrange_multiplier[t]
            diff = sum(abs(new_table[k] - self.table[k]) for k in new_table.keys())
            self.table.update(new_table)

    def score(self, src_sent: Sequence[T_src], trg_sent: Sequence[T_trg]):
        scores = [self.table[k] for k in product(src_sent, trg_sent)]
        mean_score = sum(scores) / len(scores)
        return mean_score

    def __call__(self, src_sent: Sequence[T_src], trg_sent: Sequence[T_trg]):
        return self.score(src_sent, trg_sent)


class InitParamFactory:
    def __init__(
        self,
        dataset: Sequence[Tuple[Sequence[Hashable], Sem, Sequence[int]]],
        scale: float,
        default: float,
    ):
        self.scale = scale
        self.default = default

        src_sentences = list(map(lambda x: x[0], dataset))
        trg_sentences = list(map(lambda x: x[1].constant_nodes(), dataset))

        self.model = IBMModelOne(
            dataset=list(zip(src_sentences, trg_sentences)),
        )

    def __call__(self, key: LexItem) -> float:
        pho = key.pho
        const = key.sem.constant_nodes()
        if len(const) > 0:
            return self.scale * self.model.score(pho, const)
        else:
            return 0
