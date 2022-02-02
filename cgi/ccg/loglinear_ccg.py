import itertools
import random
from collections import Counter, defaultdict
from typing import (Callable, Dict, Hashable, List, Optional, Sequence, Set,
                    Tuple, Union)

import numpy as np

from .cell import Derivation
from .lexitem import BasicCat, LexItem, Sem


class NoDerivationObtained(Exception):
    pass


class NoCorrectDerivationObtained(Exception):
    pass


def np_softmax(
    x: np.ndarray,
    return_empty_if_empty: bool = True
) -> np.ndarray:
    if x.size == 0 and return_empty_if_empty:
        return np.array([])
    c = np.max(x)
    e = np.exp(x - c)
    s = np.sum(e)
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
    lr:                  float
    c:                   float
    init_param:          float
    step:                int
    max_word_len:        int
    params:              'defaultdict_for_param'
    grad:                'defaultdict[LexItem, float]'
    __lexicon:           Lexicon
    __pho_to_lexicon:    'defaultdict[Tuple[Hashable, ...], Set[Derivation]]'  # noqa: E501
    __unigram:           'defaultdict[Hashable, float]'
    derivations:         List[Derivation]
    parseable_sentences: List[Sentence]

    def __init__(
        self,
        lr: float,
        c:  float,
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
                    score=self.params[e]))

    def update_lexicon(self, x: Union[Lexicon, LexItem]):
        if isinstance(x, LexItem):
            self.__lexicon.add(x)
            self.__pho_to_lexicon[x.pho].add(
                Derivation(
                    item=x,
                    score=self.params[x]))
        else:
            for e in x - self.__lexicon:
                self.__lexicon.add(e)
                self.__pho_to_lexicon[e.pho].add(
                    Derivation(
                        item=e,
                        score=self.params[e]))

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

    def unigram_model(self, x: Union[int, Sequence[int]]) -> float:
        if isinstance(x, int):
            x = [x]
        return float(np.prod([self.unigram[e] for e in x]))

    def zero_grad(self):
        self.derivations = []
        self.grad = defaultdict(float)

    def calc_grad(
        self,
        derivations:         List[Derivation],
        target_logical_form: Sem,
    ):
        logical_forms = np.array([x.item.sem for x in derivations], dtype=Sem)
        features = [Counter(x.lexitems) for x in derivations]
        ps: np.ndarray = np_softmax(
            np.array([x.score for x in derivations], dtype=float))

        are_correct: np.ndarray = logical_forms == target_logical_form

        p_to_be_correct: float = np.sum(ps[are_correct])

        for p, f, is_correct in zip(ps, features, are_correct):
            for k, v in f.items():
                self.grad[k] -= v * p
                if is_correct:
                    self.grad[k] += v * p / p_to_be_correct

    def update_params(self):
        lr_here = self.lr / (1 + self.c * self.step)
        for k, v in self.grad.items():
            self.params[k] += lr_here * v
        self.step += 1

    def __call__(
        self,
        sentence:     Sentence,
        lexicon:      Optional[Lexicon] = None,
        logical_form: Optional[Sem] = None,
        beam_size:    Optional[int] = None,
        return_only_top_score: bool = False,
    ):
        return self.parse(
            sentence,
            lexicon,
            logical_form=logical_form,
            beam_size=beam_size,
            return_only_top_score=return_only_top_score,
        )

    def parse(
        self,
        sentence:     Sentence,
        lexicon:      Optional[Lexicon] = None,
        logical_form: Optional[Sem] = None,
        beam_size:    Optional[int] = None,
        return_only_top_score: bool = False,
    ):
        assert lexicon or self.lexicon, lexicon

        if not lexicon:
            lexicon = self.lexicon
            pho_to_lexicon = self.__pho_to_lexicon
        else:
            pho_to_lexicon: 'defaultdict[Tuple[Hashable, ...], Set[Derivation]]' = defaultdict(set)  # noqa: E501
            for e in lexicon:
                pho_to_lexicon[e.pho].add(
                    Derivation(
                        item=e,
                        score=self.params[e]))

        def prune_fn(
            x: List[Derivation],
            b: Optional[int] = beam_size,
        ) -> List[Derivation]:
            if b is None or len(x) < b:
                return x
            else:
                return sorted(
                    x, key=(lambda k: k.score), reverse=True)[:b]

        cell: 'defaultdict[Tuple[int, int], List[Derivation]]' = defaultdict(list)  # noqa: E501

        for width in range(len(sentence) + 1):
            for i in range(len(sentence) - width + 1):
                j = i + width

                items = sorted(
                    pho_to_lexicon[sentence[i:j]], key=(lambda x: str(x)))
                items = random.sample(items, len(items))
                cell[i, j].extend(items)

                for k in range(i + 1, j):
                    derivs_1 = prune_fn(cell[i, k])
                    derivs_2 = prune_fn(cell[k, j])
                    for deriv_1 in derivs_1:
                        for deriv_2 in derivs_2:
                            if deriv_1.item.can_take_as_right_arg(deriv_2.item):  # noqa: E501
                                item = deriv_1.item.takes_as_right_arg(deriv_2.item)  # noqa: E501
                                cell[i, j].append(
                                    Derivation(
                                        item=item,
                                        score=(deriv_1.score + deriv_2.score),
                                        backptrs=(deriv_1, deriv_2)))
                            elif deriv_2.item.can_take_as_left_arg(deriv_1.item):  # noqa: E501
                                item = deriv_2.item.takes_as_left_arg(deriv_1.item)  # noqa: E501
                                cell[i, j].append(
                                    Derivation(
                                        item=item,
                                        score=(deriv_1.score + deriv_2.score),
                                        backptrs=(deriv_1, deriv_2)))
        derivs = cell[0, len(sentence)]
        derivs = filter(lambda x: x.item.cat == BasicCat.S, derivs)
        if logical_form is not None:
            derivs = filter(lambda x: x.item.sem == logical_form, derivs)
        if return_only_top_score:
            derivs = list(derivs)  # this operation is necessary.
            if len(derivs) > 0:
                top_score = max(p.score for p in derivs)
                derivs = filter(lambda x: x.score == top_score, derivs)
        return list(derivs)

    def set_parseable_sentences(
        self,
        vocab_size:   int,
        sentence_len: int,
        lexicon:      Optional[Lexicon] = None,
    ):
        assert isinstance(vocab_size,   int) and vocab_size > 0, vocab_size
        assert isinstance(sentence_len, int) and sentence_len > 0, sentence_len
        self.parseable_sentences = []
        for sentence in itertools.product(range(1, vocab_size),
                                          repeat=sentence_len):
            parses = self(sentence, lexicon)
            if len(parses) > 0:
                self.parseable_sentences.append(sentence)
        return self.parseable_sentences

    def generate(
        self,
        logical_form: Sem,
        vocab_size:   int,
        sentence_len: int,
        n_samples:    int = 10000,
        lexicon:      Optional[Lexicon] = None,
        append_eos:   bool = True,
    ):
        # assert hasattr(self, 'parseable_sentences')
        assert isinstance(logical_form, Sem), logical_form
        assert isinstance(append_eos,  bool), append_eos

        vocab = tuple(range(1, vocab_size))
        probs = tuple(self.__unigram[e] for e in vocab)

        sentence = (0,) * sentence_len
        max_prob = 0.0

        for _ in range(n_samples):
            candidate = tuple(
                random.choices(vocab, k=sentence_len, weights=probs))
            prior_prob = self.unigram_model(candidate)
            if prior_prob < max_prob:
                continue
            parses = self(candidate, lexicon)

            sems = np.array([x.item.sem for x in parses], dtype=Sem)
            likelihoods = np_softmax(np.array([x.score for x in parses]))
            likelihood = np.sum(likelihoods[sems == logical_form])

            joint_prob = likelihood * prior_prob
            if joint_prob > max_prob:
                sentence = candidate

        if append_eos:
            sentence = sentence + (0,)

        return sentence
