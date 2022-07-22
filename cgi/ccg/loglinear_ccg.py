import heapq
import itertools
from collections import Counter, defaultdict
from typing import Callable, Dict, Hashable, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt

from .cell import Derivation
from .lexitem import BasicCat, LexItem, Sem, TransductionRule, Var


class NoDerivationObtained(Exception):
    pass


class NoCorrectDerivationObtained(Exception):
    pass


def np_softmax(
    x: npt.NDArray[np.float_],
    return_empty_if_empty: bool = True
) -> npt.NDArray[np.float_]:
    if x.size == 0 and return_empty_if_empty:
        return x
    c: npt.NDArray[np.float_] = np.max(x)
    e: npt.NDArray[np.float_] = np.exp(x - c)
    s: npt.NDArray[np.float_] = np.sum(e)
    return e / s


Sentence = Tuple[Hashable, ...]
Lexicon = Set[LexItem]


class defaultdict_for_param(Dict[LexItem, float]):
    def __init__(self, default_factory: Callable[[LexItem], float]):
        self.default_factory = default_factory

    def __getitem__(self, item: LexItem) -> float:
        if item not in self:
            self[item] = self.default_factory(item)
        return super().__getitem__(item)


class LogLinearCCG:
    lr: float
    c: float
    init_param: float
    step: int
    max_word_len: int
    params: defaultdict_for_param
    grad: "defaultdict[LexItem, float]"
    __lexicon: Lexicon
    __pho_to_lexicon: "defaultdict[Tuple[Hashable, ...], Set[Derivation]]"
    __unigram: "defaultdict[Hashable, float]"
    __lexitem_to_transduction_rule: Dict[LexItem, TransductionRule]
    derivations: List[Derivation]
    parseable_sentences: List[Sentence]

    def __init__(
        self,
        lr: float,
        c: float,
        init_param_factory: Callable[[LexItem], float],
    ):
        ##################
        # Initialization #
        ##################
        self.lr = lr
        self.c = c
        self.params = defaultdict_for_param(init_param_factory)
        self.__lexicon = set()
        self.__pho_to_lexicon = defaultdict(set)
        self.__unigram = defaultdict(float)
        self.__lexitem_to_transduction_rule = dict()
        self.step = 0

    @property
    def lexicon(self):
        return self.__lexicon

    @lexicon.setter
    def lexicon(self, x: Lexicon):
        assert isinstance(x, set), x
        assert all(isinstance(e, LexItem) for e in x), x
        self.__lexicon = x
        self.__pho_to_lexicon = defaultdict(set)
        for e in x:
            self.__pho_to_lexicon[e.pho].add(
                Derivation(
                    item=e,
                    score=self.params[e],
                ),
            )

    def update_lexicon(self, x: Union[Lexicon, LexItem]):
        if isinstance(x, LexItem):
            self.__lexicon.add(x)
            self.__pho_to_lexicon[x.pho].add(
                Derivation(
                    item=x,
                    score=self.params[x],
                ),
            )
        else:
            for e in x - self.__lexicon:
                self.__lexicon.add(e)
                self.__pho_to_lexicon[e.pho].add(
                    Derivation(
                        item=e,
                        score=self.params[e],
                    ),
                )

    @property
    def unigram(self):
        return self.__unigram

    @unigram.setter
    def unigram(self, x: 'Counter[Hashable]'):
        assert isinstance(x, Counter), x
        normalizer = sum(x.values())
        assert normalizer > 0
        for k, v in x.items():
            self.__unigram[k] = v / normalizer

    def unigram_model(self, x: Sentence) -> float:
        return float(np.prod([self.unigram[e] for e in x]))

    def zero_grad(self):
        self.derivations = []
        self.grad = defaultdict(float)

    def prob_of_target_logical_form(
        self,
        derivations: List[Derivation],
        target_logical_form: Sem,
        probs: Optional[npt.NDArray[np.float_]] = None,
    ):
        logical_forms = np.array([x.item.sem for x in derivations])
        if probs is None:
            probs = np_softmax(np.array([x.score for x in derivations]))
        return np.sum(probs[logical_forms == target_logical_form])

    def calc_grad(
        self,
        derivations: List[Derivation],
        target_logical_form: Sem,
    ):
        features = [Counter(x.lexitems) for x in derivations]
        probs: npt.NDArray[np.float_] = np_softmax(
            np.array([x.score for x in derivations], dtype=np.float_)
        )
        prob_of_target_logical_form = self.prob_of_target_logical_form(
            derivations, target_logical_form, probs=probs,
        )
        for p, f, derivation in zip(probs, features, derivations):
            for k, v in f.items():
                self.grad[k] -= v * p
                if derivation.item.sem == target_logical_form:
                    self.grad[k] += v * p / prob_of_target_logical_form

    def update_params(self):
        lr_here = self.lr / (1 + self.c * self.step)
        for k, v in self.grad.items():
            self.params[k] += lr_here * v
        self.step += 1

    def __call__(
        self,
        sentence: Sentence,
        lexicon: Optional[Lexicon] = None,
        logical_form: Optional[Sem] = None,
        beam_size: Optional[int] = None,
    ):
        return self.parse(
            sentence,
            lexicon,
            logical_form=logical_form,
            beam_size=beam_size,
        )

    def parse(
        self,
        sentence: Sentence,
        lexicon: Optional[Lexicon] = None,
        logical_form: Optional[Sem] = None,
        beam_size: Optional[int] = None,
    ):
        assert lexicon or self.lexicon, lexicon

        if lexicon is None:
            pho_to_lexicon = self.__pho_to_lexicon
        else:
            pho_to_lexicon: Dict[Tuple[Hashable, ...], Set[Derivation]] = {}
            for e in sorted(lexicon):
                if e.pho not in pho_to_lexicon:
                    pho_to_lexicon[e.pho] = set()
                pho_to_lexicon[e.pho].add(
                    Derivation(
                        item=e,
                        score=self.params[e],
                    )
                )

        def prune_fn(x: List[Derivation], b: Optional[int] = beam_size) -> List[Derivation]:
            if b is None or len(x) < b:
                return x
            else:
                return heapq.nsmallest(b, x)
                # return sorted(x)[:b]

        cell: "defaultdict[Tuple[int, int], List[Derivation]]" = defaultdict(list)

        for width in range(len(sentence) + 1):
            for i in range(len(sentence) - width + 1):
                j = i + width

                cell[i, j].extend(pho_to_lexicon[sentence[i:j]])

                for k in range(i + 1, j):
                    derivs_1 = prune_fn(cell[i, k])
                    derivs_2 = prune_fn(cell[k, j])
                    for deriv_1 in derivs_1:
                        for deriv_2 in derivs_2:
                            if deriv_1.item.can_take_as_right_arg(deriv_2.item):
                                item = deriv_1.item.takes_as_right_arg(deriv_2.item)
                                cell[i, j].append(
                                    Derivation(
                                        item=item,
                                        score=(deriv_1.score + deriv_2.score),
                                        backptrs=(deriv_1, deriv_2),
                                    )
                                )
                            elif deriv_2.item.can_take_as_left_arg(deriv_1.item):
                                item = deriv_2.item.takes_as_left_arg(deriv_1.item)
                                cell[i, j].append(
                                    Derivation(
                                        item=item,
                                        score=(deriv_1.score + deriv_2.score),
                                        backptrs=(deriv_1, deriv_2),
                                    )
                                )
        derivs = cell[0, len(sentence)]
        derivs = filter(lambda x: x.item.cat == BasicCat.S, derivs)
        if logical_form is not None:
            derivs = filter(lambda x: x.item.sem == logical_form, derivs)
        return list(derivs)

    def apply_transduction(
        self,
        sem: Sem,
    ) -> Tuple[List[Sentence], List[List[LexItem]]]:
        sentences: List[Sentence] = []
        histories: List[List[LexItem]] = []
        for lexitem in self.lexicon:
            try:
                rule = self.__lexitem_to_transduction_rule[lexitem]
            except KeyError:
                rule = self.__lexitem_to_transduction_rule[lexitem] = lexitem.to_transduction_rule()

            var_to_sub_sem = rule.lhs.get_map_from_var_to_subsumed_sem(sem)
            if var_to_sub_sem is not None:
                variable_substituted_rhs = (
                    var_to_sub_sem[x] if isinstance(x, Var) else x
                    for x in rule.rhs
                )
                sub_sentences_and_histories: List[Tuple[List[Sentence], List[List[LexItem]]]] = [
                    self.apply_transduction(x)
                    if isinstance(x, Sem) else ([x], [[]])
                    for x in variable_substituted_rhs
                ]
                sub_sentences = map(lambda x: x[0], sub_sentences_and_histories)
                sub_histories = map(lambda x: x[1], sub_sentences_and_histories)
                all_sentences: List[Sentence] = [
                    sum(x, start=()) for x in itertools.product(*sub_sentences)
                ]
                all_histories: List[List[LexItem]] = [
                    sum(x, start=[lexitem]) for x in itertools.product(*sub_histories)
                ]
                sentences.extend(all_sentences)
                histories.extend(all_histories)
        return sentences, histories
