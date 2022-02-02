from typing import Hashable, Literal, Optional, Sequence, Set, Tuple, TypeVar

from .cell import Derivation
from .lexitem import (Action, AndThen, BasicCat, Command, FunctCat, Iter,
                      Lambda, LexItem, Number, Sem, Variable,
                      lexitems_of_command, sem_of_command)

_T = TypeVar('_T')


def genlex(
    message: Sequence[Hashable],
    command: Command,
    max_len: Optional[int] = None,
    style:   Literal['trigger', 'split'] = 'trigger',
) -> Set[LexItem]:
    if style == 'trigger':
        return genlex_trigger_style(message, command, max_len)
    elif style == 'split':
        return genlex_split_style(message, command, max_len)
    else:
        raise ValueError(f'Unknown style {style}')


def genlex_trigger_style(
    message: Sequence[Hashable],
    command: Command,
    max_len: Optional[int] = None,
):
    message = tuple(message)
    pho_set = partial_strs_of(message, max_len=max_len)
    lex_set = lexitems_of_command(command)
    return {x.with_pho(tuple(y)) for x in lex_set for y in pho_set}


def genlex_split_style(
    message: Sequence[Hashable],
    command: Command,
    max_len: Optional[int] = None,
):
    message = tuple(message)

    top_lexitem = LexItem(
        cat=BasicCat.S,
        sem=sem_of_command(command),
        pho=message)
    lexitems = split(top_lexitem)

    if max_len is not None:
        lexitems = {x for x in lexitems if len(x.pho) <= max_len}

    return lexitems


def newlex(parse: Derivation):
    items: Set[LexItem] = set()
    queue = [parse]
    while queue:
        e = queue.pop(-1)
        backptrs = e.backptrs
        # if not e.is_leaf() and all(b.is_leaf() for b in backptrs):
        if not e.is_leaf():
            items.add(e.item)
        else:
            queue.extend(backptrs)
    items = items.union(*(split(
        e, recursive=False, consider_filler=False) for e in parse.lexitems))
    return items


def partial_strs_of(
    x: Sequence[_T],
    max_len: Optional[int] = None
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
    consider_filler: bool = False,
):
    items = {lexitem}
    queue = [lexitem]
    while queue:
        e = queue.pop(-1)  # depth first search
        cat = e.cat
        sem = e.sem
        pho = e.pho

        if isinstance(sem, (Action, Number)):
            items.add(e)
        elif isinstance(sem, (AndThen, Iter, Lambda)):
            prefixes = {pho[:i] for i in range(1, len(pho))}
            suffixes = {pho[i:] for i in range(1, len(pho))}
            affixes = prefixes | suffixes

            pairs = extract_fun_arg_pairs(
                sem,
                consider_filler=consider_filler,
            )

            args_lexitems = {
                LexItem(cat=basic_cat_of(arg), sem=arg, pho=pho)
                for _, arg in pairs for pho in affixes}
            funs_lexitems = {
                LexItem(cat=FunctCat(cat, '/',  basic_cat_of(arg)), sem=fun, pho=pho)  # noqa: E501
                for fun, arg in pairs for pho in prefixes} | {
                LexItem(cat=FunctCat(cat, '\\', basic_cat_of(arg)), sem=fun, pho=pho)  # noqa: E501
                for fun, arg in pairs for pho in suffixes}

            items |= args_lexitems
            items |= funs_lexitems
            if recursive:
                queue.extend(args_lexitems)
                queue.extend(funs_lexitems)
    return items


def extract_fun_arg_pairs(
    sem: Sem,
    max_arity: int = 6,
    consider_filler: bool = False,
) -> Set[Tuple[Lambda, Sem]]:
    pairs: Set[Tuple[Lambda, Sem]] = set()

    if (
        consider_filler and
        not (isinstance(sem, Lambda) and sem.arg == sem.body) and
        not sem.fv()
    ):
        x0 = Variable('x0')
        pairs.add((Lambda(x0, x0), sem))

    if isinstance(sem, AndThen):
        argl = sem.fst
        argr = sem.snd
        x = Variable(f'x{max_arity - 1}')
        if not argl.fv():
            pairs.add((Lambda(x, AndThen(x, argr)), argl))
            for f, a in extract_fun_arg_pairs(
                argl,
                max_arity=max_arity,
                consider_filler=consider_filler,
            ):
                pairs.add((Lambda(x, AndThen(f(x), argr)), a))
        if not argr.fv():
            pairs.add((Lambda(x, sem.__class__(argl, x)), argr))
            for f, a in extract_fun_arg_pairs(
                argr,
                max_arity=max_arity,
                consider_filler=consider_filler,
            ):
                pairs.add((Lambda(x, AndThen(argl, f(x))), a))
        return pairs
    elif isinstance(sem, Iter):
        # almost identical code as above, but for ease of type-checking
        argl = sem.fst
        argr = sem.snd
        x = Variable(f'x{max_arity - 1}')
        if not argl.fv():
            pairs.add((Lambda(x, sem.__class__(x, argr)), argl))
            for f, a in extract_fun_arg_pairs(
                argl,
                max_arity=max_arity,
                consider_filler=consider_filler,
            ):
                pairs.add((Lambda(x, Iter(f(x), argr)), a))
        if not argr.fv():
            pairs.add((Lambda(x, sem.__class__(argl, x)), argr))
        return pairs
    elif isinstance(sem, Lambda):
        x, body = sem.arg, sem.body
        if sem.arity() >= max_arity:
            return set()
        for fun, arg in extract_fun_arg_pairs(
            body,
            max_arity - 1,
            consider_filler=consider_filler,
        ):
            y, new_body = fun.arg, fun.body
            pairs.add((Lambda(y, Lambda(x, new_body)), arg))
        return pairs
    elif isinstance(sem, Variable):
        return set()
    else:
        return pairs


def basic_cat_of(x: Sem):
    assert isinstance(x, Sem), x
    if isinstance(x, Action):
        return BasicCat.V
    elif isinstance(x, Number):
        return BasicCat.N
    elif isinstance(x, (Lambda, Variable)):
        raise ValueError(f'category of {x} is not basic.')
    else:
        return BasicCat.S
