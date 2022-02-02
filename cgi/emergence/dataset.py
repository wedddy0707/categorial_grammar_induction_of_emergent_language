from typing import Sequence, Tuple, Any, Union
import pandas as pd
import itertools
import torch

VERB_ATTR = ("look", "jump", "walk", "run")
DIRE_ATTR = ("up", "down", "left", "right")
ITER_ATTR = ("1", "2", "3", "4")
CONJ_ATTR = ("and",)

COMMAND_ID = dict((y, x) for x, y in enumerate(
    ("eos",) +
    VERB_ATTR +
    DIRE_ATTR +
    ITER_ATTR +
    CONJ_ATTR
))
assert COMMAND_ID['eos'] == 0

def command_to_command_ids(x: Sequence[str]):
    return tuple(COMMAND_ID[e] for e in x)

def flatten_nested_tuple(x: Union[Tuple[Any, ...], str]) -> Tuple[str, ...]:
    if isinstance(x, tuple):
        return tuple(itertools.chain.from_iterable(map(flatten_nested_tuple, x)))
    else:
        return (x,)

Phrase = Tuple[Tuple[str, str], str]
TreeCommand = Union[
    Phrase,
    Tuple[Phrase, "TreeCommand"],
]

def enumerate_command(
    max_n_conj: int,
    random_seed: int = 1,
    valid_p: float = 0.5,
    train_split_label: str = "train",
    valid_split_label: str = "valid",
) -> pd.DataFrame:
    ##############################################
    # begin with making tree structured commands #
    ##############################################
    tree_phrases: Tuple[Phrase, ...] = tuple(itertools.product(
        tuple(itertools.product(VERB_ATTR, DIRE_ATTR)), ITER_ATTR
    ))
    tree_commands: Tuple[TreeCommand, ...] = tree_phrases
    for _ in range(max_n_conj):
        tree_commands = tuple(itertools.product(
            tree_phrases, tuple(itertools.product(CONJ_ATTR, tree_commands)),
        ))
    #################################
    # make flat structured commands #
    #################################
    flat_commands: Tuple[Tuple[str, ...], ...] = tuple(map(flatten_nested_tuple, tree_commands))
    #######################################################
    # convert str-typed commands to int-typed command ids #
    #######################################################
    maxlen_com_ids = max(map(len, flat_commands)) + 1  # I mean by "+1" that every command ends with EOS.
    command_ids: Tuple[Tuple[int, ...], ...] = tuple(map(command_to_command_ids, flat_commands))
    command_ids: Tuple[Tuple[int, ...], ...] = tuple(map(
        lambda x: (x + (0,) * maxlen_com_ids)[:maxlen_com_ids], command_ids
    ))
    command_tensors: Tuple[torch.Tensor, ...] = tuple(map(
        lambda x: torch.as_tensor(x, dtype=torch.long), command_ids
    ))
    ##################
    # make dataframe #
    ##################
    df = pd.DataFrame(data={
        "tree_command":   tree_commands,
        "flat_command":   flat_commands,
        "command_ids":    command_ids,
        "command_tensor": command_tensors,
    })
    #####################
    # shuffle dataframe #
    #####################
    df: pd.DataFrame = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    n_valid_samples = int(len(df) * valid_p)
    n_train_samples = len(df) - n_valid_samples
    df["split"] = [train_split_label] * n_train_samples + [valid_split_label] * n_valid_samples
    return df
