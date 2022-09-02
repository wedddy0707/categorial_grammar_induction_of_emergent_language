import dataclasses
import enum
from typing import Hashable, Literal, Optional, Tuple, Union

from ..semantics.semantics import Sem, Lambda, Var


class Cat:
    def n_slashes(self, dir: Optional[Literal['/', '\\']] = None) -> int:
        return 0

    def to_latex(self) -> str:
        raise NotImplementedError


class BasicCat(Cat, enum.Enum):
    N = enum.auto()
    S = enum.auto()

    def __str__(self):
        return self.name

    def to_latex(self):
        return "\\texttt{{{}}}".format(self.name)


@dataclasses.dataclass(frozen=True)
class FunctCat(Cat):
    cod: Cat
    slash: Literal["/", "\\"]
    dom: Cat

    def __post_init__(self):
        assert isinstance(self.cod, Cat)
        assert isinstance(self.dom, Cat)
        assert self.slash in ('/', '\\')

    def __str__(self):
        if isinstance(self.dom, FunctCat):
            return f'{self.cod}{self.slash}({self.dom})'
        else:
            return f'{self.cod}{self.slash}{self.dom}'

    def n_slashes(self, dir: Optional[Literal['/', '\\']] = None) -> int:
        assert dir in {None, '\\', '/'}
        n = 0
        if dir is None or dir == self.slash:
            n = 1
        n += self.cod.n_slashes(dir=dir)
        n += self.dom.n_slashes(dir=dir)
        return n

    def to_latex(self):
        latex_slash = "\\backslash" if self.slash == "\\" else "/"
        latex_cod = self.cod.to_latex()
        latex_dom = self.dom.to_latex()
        if isinstance(self.dom, FunctCat):
            latex_dom = "({})".format(latex_dom)
        return " ".join([latex_cod, latex_slash, latex_dom])


class CategorialGrammarRule(enum.Enum):
    forward_application_rule = enum.auto()
    backward_application_rule = enum.auto()


@dataclasses.dataclass(frozen=True)
class TransductionRule:
    lhs: Sem
    rhs: Tuple[Union[Tuple[Hashable, ...], Var], ...]


@dataclasses.dataclass(frozen=True)
class LexItem:
    cat: Cat
    sem: Sem
    pho: Tuple[Hashable, ...] = ()

    def __post_init__(self):
        assert isinstance(self.cat, Cat)
        assert isinstance(self.sem, Sem)
        assert isinstance(self.pho, tuple)

    def __str__(self):
        return f"{self.pho} |- {self.cat}: {self.sem}"

    def __lt__(self, other: "LexItem"):
        return repr(self) < repr(other)

    def __le__(self, other: "LexItem"):
        return repr(self) <= repr(other)

    def __gt__(self, other: "LexItem"):
        return repr(self) > repr(other)

    def __ge__(self, other: "LexItem"):
        return repr(self) >= repr(other)

    def to_latex(self, bracket_with_dallers: bool = True) -> str:
        r"""
        \newcommand{\lexitem}[3]{#1\!\vdash\!#2\!:\!#3}
        """
        latex_cat = self.cat.to_latex()
        latex_sem = self.sem.to_latex()
        latex_pho = "\\text{{{}}}".format(" ".join(str(x) for x in self.pho))
        latex_lexitem = "\\lexitem{{{}}}{{{}}}{{{}}}".format(
            latex_pho,
            latex_cat,
            latex_sem,
        )
        if bracket_with_dallers:
            latex_lexitem = "${}$".format(latex_lexitem)
        return latex_lexitem

    def can_take_as_right_arg(self, other: "LexItem"):
        return isinstance(self.cat, FunctCat) and self.cat.slash == "/" and self.cat.dom == other.cat

    def can_take_as_left_arg(self, other: "LexItem"):
        return isinstance(self.cat, FunctCat) and self.cat.slash == "\\" and self.cat.dom == other.cat

    def takes_as_right_arg(self, other: "LexItem"):
        assert isinstance(self.cat, FunctCat)
        return self.__class__(
            cat=self.cat.cod,
            sem=self.sem.takes(other.sem),
            pho=self.pho + other.pho)

    def takes_as_left_arg(self, other: "LexItem"):
        assert isinstance(self.cat, FunctCat)
        return self.__class__(
            cat=self.cat.cod,
            sem=self.sem.takes(other.sem),
            pho=other.pho + self.pho,
        )

    def with_pho(self, pho: Tuple[Hashable, ...]):
        return self.__class__(
            cat=self.cat,
            sem=self.sem,
            pho=pho,
        )

    def to_transduction_rule(self):
        rhs = (self.pho,)
        cat: Cat = self.cat
        sem: Sem = self.sem
        while isinstance(cat, FunctCat):
            assert isinstance(sem, Lambda)
            rhs = rhs + (sem.arg,) if cat.slash == "/" else (sem.arg,) + rhs
            cat = cat.cod
            sem = sem.body
        lhs = sem
        return TransductionRule(lhs=lhs, rhs=rhs)
