import dataclasses
import enum
from typing import Hashable, Literal, Optional, Set, Tuple, Union

MAX_ITER_COUNT = 4


class Sem:
    def subst(self, old: 'Sem', new: 'Sem'):
        return new if self == old else self

    def normal_form(self):
        return self

    def __call__(self, arg: 'Sem'):
        return self.takes(arg)

    def takes(self, arg: 'Sem') -> 'Sem':
        arg
        raise NotImplementedError

    def arity(self) -> int:
        return 0

    def n_nodes(self) -> int:
        return 1

    def fv(self) -> Set['Variable']:
        return set()

    def constant(self) -> Tuple[Union['Action', 'Number'], ...]:
        return ()


class Action(Sem, enum.Enum):
    LOOK = enum.auto()
    JUMP = enum.auto()
    WALK = enum.auto()
    RUN = enum.auto()
    TURN_UP = enum.auto()
    TURN_DOWN = enum.auto()
    TURN_LEFT = enum.auto()
    TURN_RIGHT = enum.auto()

    def __str__(self):
        return self.name

    def constant(self) -> Tuple[Union['Action', 'Number'], ...]:
        return (self,)


class Number(int, Sem):
    def constant(self) -> Tuple[Union['Action', 'Number'], ...]:
        return (self,)


class Variable(str, Sem):
    def n_nodes(self) -> int:
        return 0

    def fv(self) -> Set['Variable']:
        return {self}


@dataclasses.dataclass(frozen=True)
class Lambda(Sem):
    arg:  Variable
    body: Sem

    def __post_init__(self):
        assert isinstance(self.arg, Variable), self.arg
        assert isinstance(self.body,     Sem), self.body

    def __str__(self):
        return f'\\{self.arg}.{self.body}'

    def __iter__(self):
        return iter((self.arg, self.body))

    def subst(self, old: 'Sem', new: 'Sem'):
        if self.arg == old:
            return self
        else:
            return self.__class__(
                arg=self.arg,
                body=self.body.subst(old, new))

    def takes(self, arg: Sem):
        return self.body.subst(self.arg, arg)

    def arity(self):
        return 1 + self.body.arity()

    def n_nodes(self):
        return self.body.n_nodes()

    def fv(self) -> Set['Variable']:
        return self.body.fv() - {self.arg}

    def constant(self) -> Tuple[Union['Action', 'Number'], ...]:
        return self.body.constant()


@dataclasses.dataclass(frozen=True)
class AndThen(Sem):
    fst: Sem
    snd: Sem

    def __post_init__(self):
        assert isinstance(self.fst, Sem)
        assert isinstance(self.snd, Sem)

    def __str__(self):
        return f'and-then({self.fst}, {self.snd})'

    def __iter__(self):
        return iter((self.fst, self.snd))

    def subst(self, old: 'Sem', new: 'Sem'):
        return self.__class__(
            fst=self.fst.subst(old, new),
            snd=self.snd.subst(old, new))

    def n_nodes(self):
        return 1 + self.fst.n_nodes() + self.snd.n_nodes()

    def fv(self) -> Set['Variable']:
        empty: Set['Variable'] = set()
        return empty.union(*(x.fv() for x in self))

    def constant(self) -> Tuple[Union['Action', 'Number'], ...]:
        return self.fst.constant() + self.snd.constant()


@dataclasses.dataclass(frozen=True)
class Iter(Sem):
    val: Sem
    cnt: Union[Number, Variable]

    def __post_init__(self):
        assert isinstance(self.val, Sem), self.val
        assert isinstance(self.cnt, Sem), self.cnt

    def __str__(self):
        return f'iter({self.val}, {self.cnt})'

    @property
    def fst(self):
        return self.val

    @property
    def snd(self):
        return self.cnt

    def __iter__(self):
        return iter((self.val, self.cnt))

    def subst(self, old: 'Sem', new: 'Sem'):
        return self.__class__(
            val=self.val.subst(old, new),
            cnt=self.cnt)

    def n_nodes(self):
        return 1 + self.val.n_nodes() + self.cnt.n_nodes()

    def fv(self) -> Set['Variable']:
        empty: Set['Variable'] = set()
        return empty.union(*map(lambda x: x.fv(), self))

    def constant(self) -> Tuple[Union['Action', 'Number'], ...]:
        return self.val.constant() + self.cnt.constant()


class Cat:
    def n_slashes(self, dir: Optional[Literal['/', '\\']] = None) -> int:
        return 0


class BasicCat(Cat, enum.Enum):
    V = enum.auto()
    N = enum.auto()
    S = enum.auto()

    def __str__(self):
        return self.name


@dataclasses.dataclass(frozen=True)
class FunctCat(Cat):
    cod:   Cat
    slash: str
    dom:   Cat

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
        assert dir in (None, '\\', '/')
        n = 0
        if dir is None or dir == self.slash:
            n = 1
        n += self.cod.n_slashes(dir=dir)
        n += self.dom.n_slashes(dir=dir)
        return n


@dataclasses.dataclass(frozen=True)
class LexItem:  # 語彙項目
    cat: Cat
    sem: Sem
    pho: Tuple[Hashable, ...] = tuple()

    def __post_init__(self):
        assert isinstance(self.cat, Cat)
        assert isinstance(self.sem, Sem)
        assert isinstance(self.pho, tuple)

    def __str__(self):
        return f"{self.pho} |- {self.cat}: {self.sem}"

    def to_latex_equation(self) -> str:
        return f"{self.pho} \\vdash {self.cat}: {self.sem}"

    def can_take_as_right_arg(self, other: 'LexItem'):
        return (
            isinstance(self.cat, FunctCat) and
            self.cat.slash == '/' and
            self.cat.dom == other.cat)

    def can_take_as_left_arg(self, other: 'LexItem'):
        return (
            isinstance(self.cat, FunctCat) and
            self.cat.slash == '\\' and
            self.cat.dom == other.cat)

    def takes_as_right_arg(self, other: 'LexItem'):
        assert isinstance(self.cat, FunctCat)
        return self.__class__(
            cat=self.cat.cod,
            sem=self.sem.takes(other.sem),
            pho=self.pho + other.pho)

    def takes_as_left_arg(self, other: 'LexItem'):
        assert isinstance(self.cat, FunctCat)
        return self.__class__(
            cat=self.cat.cod,
            sem=self.sem.takes(other.sem),
            pho=other.pho + self.pho)

    def with_pho(self, pho: Tuple[Hashable, ...]):
        return self.__class__(
            cat=self.cat,
            sem=self.sem,
            pho=pho)


x0 = Variable('x0')
x1 = Variable('x1')

SEM_ENTRY = {
    'look':       Action.LOOK,
    'jump':       Action.JUMP,
    'walk':       Action.WALK,
    'run':        Action.RUN,
    'turn-up':    Action.TURN_UP,
    'turn-down':  Action.TURN_DOWN,
    'turn-left':  Action.TURN_LEFT,
    'turn-right': Action.TURN_RIGHT,
    'up':         Lambda(x0, AndThen(Action.TURN_UP,    x0)),
    'down':       Lambda(x0, AndThen(Action.TURN_DOWN,  x0)),
    'left':       Lambda(x0, AndThen(Action.TURN_LEFT,  x0)),
    'right':      Lambda(x0, AndThen(Action.TURN_RIGHT, x0)),
    'and':        Lambda(x0, Lambda(x1, AndThen(x0, x1))),
    'filler':     Lambda(x0, x0),
}
for i in range(MAX_ITER_COUNT):
    SEM_ENTRY[str(i + 1)] = Lambda(x0, Iter(x0, Number(i + 1)))

_V = BasicCat.V
_S = BasicCat.S
_F = FunctCat

silent_lexitems = {
    # V
    LexItem(_V, SEM_ENTRY['look']),
    LexItem(_V, SEM_ENTRY['jump']),
    LexItem(_V, SEM_ENTRY['walk']),
    LexItem(_V, SEM_ENTRY['run']),
    LexItem(_V, SEM_ENTRY['turn-up']),
    LexItem(_V, SEM_ENTRY['turn-down']),
    LexItem(_V, SEM_ENTRY['turn-left']),
    LexItem(_V, SEM_ENTRY['turn-right']),
    # up
    LexItem(_F(_S,  '/', _V), SEM_ENTRY['up']),
    LexItem(_F(_S, '\\', _V), SEM_ENTRY['up']),
    # down
    LexItem(_F(_S,  '/', _V), SEM_ENTRY['down']),
    LexItem(_F(_S, '\\', _V), SEM_ENTRY['down']),
    # left
    LexItem(_F(_S,  '/', _V), SEM_ENTRY['left']),
    LexItem(_F(_S, '\\', _V), SEM_ENTRY['left']),
    # right
    LexItem(_F(_S,  '/', _V), SEM_ENTRY['right']),
    LexItem(_F(_S, '\\', _V), SEM_ENTRY['right']),
    # filler
    LexItem(_F(_V,  '/', _V), SEM_ENTRY['filler']),
    LexItem(_F(_V, '\\', _V), SEM_ENTRY['filler']),
    LexItem(_F(_S,  '/', _S), SEM_ENTRY['filler']),
    LexItem(_F(_S, '\\', _S), SEM_ENTRY['filler']),
    # and
    LexItem(_F(_F(_S,  '/', _S),  '/', _S), SEM_ENTRY['and']),
    LexItem(_F(_F(_S,  '/', _S), '\\', _S), SEM_ENTRY['and']),
    LexItem(_F(_F(_S, '\\', _S),  '/', _S), SEM_ENTRY['and']),
    LexItem(_F(_F(_S, '\\', _S), '\\', _S), SEM_ENTRY['and']),
}
for i in range(MAX_ITER_COUNT):
    silent_lexitems |= {
        LexItem(_F(_S,  '/', _S), SEM_ENTRY[str(i + 1)]),
        LexItem(_F(_S, '\\', _S), SEM_ENTRY[str(i + 1)])}

SILENT_LEX_ITEMS: Set[LexItem] = silent_lexitems
SILENT_LEX_ITEMS_ENTRY = {
    key: set(x for x in SILENT_LEX_ITEMS if x.sem == SEM_ENTRY[key])
    for key in SEM_ENTRY.keys()}


Command = Union[Tuple['Command', 'Command'], str]


def sem_of_command(command: Command) -> Sem:
    if isinstance(command, str):
        return SEM_ENTRY[command]
    else:
        sem_l, sem_r = map(sem_of_command, command)
        if next(iter(command)) == 'and':
            return Lambda(x0, SEM_ENTRY['and'](x0)(sem_r))
        else:
            return sem_r(sem_l)


def words_of_command(command: Command) -> Set[str]:
    if isinstance(command, tuple):
        empty: Set[str] = set()
        return empty.union(*[words_of_command(x) for x in command])
    else:
        return {command}


def lexitems_of_command(command: Command) -> Set[LexItem]:
    words = words_of_command(command)
    # ad-hoc
    words.add('filler')
    if 'up' in words:
        words |= {'turn-up', 'and'}
    if 'down' in words:
        words |= {'turn-down', 'and'}
    if 'left' in words:
        words |= {'turn-left', 'and'}
    if 'right' in words:
        words |= {'turn-right', 'and'}

    items: Set[LexItem] = set()
    for key in words:
        items |= SILENT_LEX_ITEMS_ENTRY[key]
    return items
