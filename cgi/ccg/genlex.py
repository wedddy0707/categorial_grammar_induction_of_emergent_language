from typing import Hashable, Literal, Optional, Sequence, Set, Tuple, TypeVar

from .cell import Derivation
from ..semantics.semantics import (
    And,
    BinaryPredicate,
    Const,
    Lambda,
    Sem,
    Var,
    eval_sem_repr,
)
from .lexitem import BasicCat, FunctCat, LexItem

_T = TypeVar("_T")


def genlex(
    message: Sequence[Hashable],
    meaning: str,
    max_len: Optional[int] = None,
    style: Literal["trigger", "split"] = "split",
) -> Set[LexItem]:
    if style == "trigger":
        raise NotImplementedError
    elif style == "split":
        return genlex_split_style(message, meaning, max_len)
    else:
        raise ValueError(f"Unknown style {style}")


def genlex_split_style(
    message: Sequence[Hashable],
    meaning: str,
    max_len: Optional[int] = None,
):
    message = tuple(message)

    top_lexitem = LexItem(
        cat=BasicCat.S,
        sem=eval_sem_repr(meaning),
        pho=message)
    lexitems = split(top_lexitem)

    if max_len is not None:
        lexitems = {x for x in lexitems if len(x.pho) <= max_len}

    return lexitems


def newlex(parse: Derivation):
    items: Set[LexItem] = set(parse.lexitems)
    # Merge Lexical Items
    queue = [parse]
    while queue:
        e = queue.pop(-1)
        queue.extend(e.backptrs)
        if e.is_preleaf():
            items.add(e.item)
    # Split Lexical Items
    items = items.union(*(split(e, recursive=False, enable_filler=False) for e in parse.lexitems))
    return items


def partial_strs_of(
    x: Sequence[_T],
    max_len: Optional[int] = None,
) -> Set[Sequence[_T]]:
    strs: Set[Sequence[_T]] = set()
    max_len = len(x) if max_len is None else min(len(x), max_len)
    for idx in range(max_len):
        strs.add(x[:idx + 1])
        strs |= partial_strs_of(x[idx + 1:], max_len=max_len)
    return strs


def split(
    lexitem: LexItem,
    recursive: bool = True,
    enable_filler: bool = False,
):
    items: Set[LexItem] = set()
    queue = [lexitem]

    while queue:
        e = queue.pop(-1)  # depth first search
        cat = e.cat
        sem = e.sem
        pho = e.pho

        if isinstance(sem, Const):
            items.add(e)
        elif isinstance(sem, (BinaryPredicate, Lambda)):
            prefixes = {pho[:i] for i in range(1, len(pho))}
            suffixes = {pho[i:] for i in range(1, len(pho))}
            affixes = prefixes | suffixes

            pairs = extract_fun_arg_pairs(sem, enable_filler=enable_filler)

            args_lexitems = {
                LexItem(cat=basic_cat_of(arg), sem=arg, pho=pho)
                for _, arg in pairs for pho in affixes
            }

            funs_lexitems = {
                LexItem(cat=FunctCat(cat, "/", basic_cat_of(arg)), sem=fun, pho=pho)
                for fun, arg in pairs for pho in prefixes
            } | {
                LexItem(cat=FunctCat(cat, "\\", basic_cat_of(arg)), sem=fun, pho=pho)
                for fun, arg in pairs for pho in suffixes
            }

            items |= args_lexitems
            items |= funs_lexitems

            if recursive:
                queue.extend(args_lexitems)
                queue.extend(funs_lexitems)

    return items


def extract_fun_arg_pairs(
    sem: Sem,
    max_arity: int = 6,
    enable_filler: bool = False,
) -> Set[Tuple[Lambda, Sem]]:

    pairs: Set[Tuple[Lambda, Sem]] = set()

    if enable_filler and not (isinstance(sem, Lambda) and sem.arg == sem.body) and not sem.fv():
        x0 = Var("x0")
        pairs.add((Lambda(x0, x0), sem))

    if isinstance(sem, (And, BinaryPredicate)):
        sem_class = sem.__class__
        argl = sem.fst
        argr = sem.snd
        x = Var(f"x{max_arity - 1}")
        if len(argl.fv()) == 0:
            pairs.add((Lambda(x, sem_class(x, argr)), argl))
            pairs |= {
                (Lambda(x, sem_class(f(x), argr)), a)
                for f, a in
                extract_fun_arg_pairs(argl, max_arity=max_arity, enable_filler=enable_filler)
            }
        if len(argr.fv()) == 0:
            pairs.add((Lambda(x, sem_class(argl, x)), argr))
            pairs |= {
                (Lambda(x, sem_class(argl, f(x))), a)
                for f, a in
                extract_fun_arg_pairs(argr, max_arity=max_arity, enable_filler=enable_filler)
            }
        return pairs
    elif isinstance(sem, Lambda):
        x, body = sem.arg, sem.body
        if sem.arity() >= max_arity:
            return set()
        for fun, arg in extract_fun_arg_pairs(body, max_arity - 1, enable_filler=enable_filler):
            y, new_body = fun.arg, fun.body
            pairs.add((Lambda(y, Lambda(x, new_body)), arg))
        return pairs
    elif isinstance(sem, Var):
        return set()
    else:
        return pairs


def basic_cat_of(x: Sem):
    assert isinstance(x, Sem), x
    if isinstance(x, Const):
        return BasicCat.N
    elif isinstance(x, (Lambda, Var)):
        raise ValueError(f'category of {x} is not basic.')
    else:
        return BasicCat.S
