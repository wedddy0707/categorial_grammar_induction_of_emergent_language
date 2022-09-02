import dataclasses
import enum
from torch.utils.data import Dataset
from typing import Set, Tuple, Dict, Type, List, Union, Optional
import torch
from numpy.random import RandomState


class Sem:
    def subst(self, old: "Sem", new: "Sem"):
        return new if self == old else self

    def normal_form(self):
        return self

    def __call__(self, arg: "Sem"):
        return self.takes(arg)

    def takes(self, arg: "Sem") -> "Sem":
        raise NotImplementedError

    def arity(self) -> int:
        return 0

    def n_nodes(self) -> int:
        return 1

    def fv(self) -> Set["Var"]:
        return set()

    def constant_nodes(self) -> Tuple[Union["Const", Type["BinaryPredicate"]], ...]:
        return ()

    def subtrees(self) -> Tuple["Sem", ...]:
        return (self,)

    def polish(self) -> Tuple[Union["Sem", Type["Sem"]], ...]:
        raise NotImplementedError

    def subsumes(self, arg: "Sem") -> bool:
        raise NotImplementedError

    def get_map_from_var_to_subsumed_sem(self, arg: "Sem") -> Optional[Dict["Var", "Sem"]]:
        raise NotImplementedError

    def to_latex(self) -> str:
        raise NotImplementedError


class Const(Sem, enum.Enum):
    CIRCLE = enum.auto()
    TRIANGLE = enum.auto()
    SQUARE = enum.auto()
    STAR = enum.auto()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

    def constant_nodes(self):
        return (self,)

    def polish(self):
        return (self,)

    def subsumes(self, arg: Sem) -> bool:
        return self == arg

    def get_map_from_var_to_subsumed_sem(self, arg: Sem) -> Optional[Dict["Var", Sem]]:
        return {} if self == arg else None

    def to_latex(self) -> str:
        return "\\texttt{{{}}}".format(self.name)


class Var(str, Sem):
    def n_nodes(self) -> int:
        return 0

    def fv(self) -> Set["Var"]:
        return {self}

    def subsumes(self, arg: Sem) -> bool:
        return True

    def polish(self):
        return (self,)

    def get_map_from_var_to_subsumed_sem(self, arg: Sem) -> Dict["Var", Sem]:
        return {self: arg}

    def to_latex(self) -> str:
        return str(self)


@dataclasses.dataclass(frozen=True)
class Lambda(Sem):
    arg: Var
    body: Sem

    def __post_init__(self):
        assert isinstance(self.arg, Var), self.arg
        assert isinstance(self.body, Sem), self.body

    def __str__(self):
        return f"\\{self.arg}.{self.body}"

    def __iter__(self):
        return iter((self.arg, self.body))

    def subst(self, old: Sem, new: Sem):
        if self.arg == old:
            return self
        else:
            return self.__class__(
                arg=self.arg,
                body=self.body.subst(old, new),
            )

    def takes(self, arg: Sem):
        return self.body.subst(self.arg, arg)

    def arity(self):
        return 1 + self.body.arity()

    def n_nodes(self):
        return self.body.n_nodes()

    def subtrees(self) -> Tuple["Sem", ...]:
        return (self,) + self.body.subtrees()

    def fv(self) -> Set[Var]:
        return self.body.fv() - {self.arg}

    def constant_nodes(self):
        return self.body.constant_nodes()

    def polish(self):
        return (self.__class__, self.arg) + self.body.polish()

    def subsumes(self, arg: "Sem") -> bool:
        return isinstance(arg, Lambda) and self.body.subsumes(arg.body)

    def get_map_from_var_to_subsumed_sem(self, arg: Sem) -> Optional[Dict["Var", Sem]]:
        if isinstance(arg, Lambda):
            m = self.body.get_map_from_var_to_subsumed_sem(arg)
            if isinstance(m, dict):
                m.pop(self.arg, None)
            return m
        return None

    def to_latex(self) -> str:
        return "\\lambda {}.{}".format(self.arg.to_latex(), self.body.to_latex())


@dataclasses.dataclass(frozen=True)
class BinaryPredicate(Sem):
    fst: Sem
    snd: Sem

    def __post_init__(self):
        assert isinstance(self.fst, Sem)
        assert isinstance(self.snd, Sem)

    def __str__(self):
        return f"{self.__class__.__name__}({self.fst}, {self.snd})"

    def __iter__(self):
        return iter((self.fst, self.snd))

    def subst(self, old: Sem, new: Sem):
        return self.__class__(
            fst=self.fst.subst(old, new),
            snd=self.snd.subst(old, new),
        )

    def n_nodes(self):
        return 1 + self.fst.n_nodes() + self.snd.n_nodes()

    def fv(self) -> Set[Var]:
        empty: Set[Var] = set()
        return empty.union(*(x.fv() for x in self))

    def constant_nodes(self):
        return (self.__class__,) + self.fst.constant_nodes() + self.snd.constant_nodes()

    def subtrees(self) -> Tuple["Sem", ...]:
        return (self,) + self.fst.subtrees() + self.snd.subtrees()

    def polish(self):
        return (self.__class__,) + self.fst.polish() + self.snd.polish()

    def subsumes(self, arg: Sem) -> bool:
        return \
            isinstance(arg, BinaryPredicate) \
            and self.__class__ == arg.__class__ \
            and self.fst.subsumes(arg.fst) \
            and self.snd.subsumes(arg.snd)

    def get_map_from_var_to_subsumed_sem(self, arg: Sem) -> Optional[Dict["Var", Sem]]:
        if isinstance(arg, BinaryPredicate) and self.__class__ == arg.__class__:
            m_fst = self.fst.get_map_from_var_to_subsumed_sem(arg.fst)
            if m_fst is None:
                return None
            m_snd = self.snd.get_map_from_var_to_subsumed_sem(arg.snd)
            if m_snd is None:
                return None
            if any(m_fst[k] != m_snd[k] for k in set(m_fst) & set(m_snd)):
                return None
            m_fst.update(m_snd)
            return m_fst
        return None

    def to_latex(self) -> str:
        return "\\texttt{{{}}}({}, {})".format(
            self.__class__.__name__,
            self.fst.to_latex(),
            self.snd.to_latex(),
        )


@dataclasses.dataclass(frozen=True)
class And(BinaryPredicate):
    pass


BIN_PREDICATES: List[Type[BinaryPredicate]] = [And]

SEMANTIC_PIECE_TO_ID: Dict[Union[Const, Type[BinaryPredicate], str], int] = {".": 0}
SEMANTIC_PIECE_TO_ID.update({x: i for i, x in enumerate(Const, start=(max(SEMANTIC_PIECE_TO_ID.values()) + 1))})
SEMANTIC_PIECE_TO_ID.update({x: i for i, x in enumerate(BIN_PREDICATES, start=(max(SEMANTIC_PIECE_TO_ID.values()) + 1))})
SEMANTIC_VOCAB_SIZE = max(len(SEMANTIC_PIECE_TO_ID), max(SEMANTIC_PIECE_TO_ID.values()))

print("Const: {}".format(set(Const)))
print("Binary Predicates: {}".format(BIN_PREDICATES))
print("SEMANTIC_PIECE_TO_ID: {}".format(SEMANTIC_PIECE_TO_ID))
print("SEMANTIC_VOCAB_SIZE: {}".format(SEMANTIC_VOCAB_SIZE))


def eval_sem_repr(x: str):
    assert "__" not in x
    return eval(x, {"__builtins__": {}}, {"Const": Const, "Var": Var, "Lambda": Lambda, "And": And})


def enumerate_semantics(n_predicates: int):
    assert n_predicates >= 0
    global BIN_PREDICATES

    dataset_of_n_predicates: Dict[int, Set[Sem]] = {0: set(Const)}

    for n in range(1, n_predicates + 1):
        dataset_of_n_predicates[n] = {
            bin_pred(fst_arg, snd_arg)
            for bin_pred in BIN_PREDICATES
            for i in range(n)
            for fst_arg in dataset_of_n_predicates[i]
            for snd_arg in dataset_of_n_predicates[n - i - 1]
        }

    dataset = dataset_of_n_predicates[n_predicates]

    return sorted(dataset, key=(lambda x: repr(x)))


def convert_semantics_to_ids_for_seq2seq_model(x: Sem) -> List[int]:
    global SEMANTIC_PIECE_TO_ID

    if isinstance(x, Const):
        return [SEMANTIC_PIECE_TO_ID[x]]
    elif isinstance(x, BinaryPredicate):
        fst_ids = convert_semantics_to_ids_for_seq2seq_model(x.fst)
        snd_ids = convert_semantics_to_ids_for_seq2seq_model(x.snd)
        return [SEMANTIC_PIECE_TO_ID[x.__class__]] + fst_ids + snd_ids
    else:
        raise NotImplementedError


TreeTensor = Union[torch.Tensor, Tuple["TreeTensor", "TreeTensor"]]


def convert_semantics_to_tensors_for_tre_model(x: Sem) -> TreeTensor:
    global SEMANTIC_PIECE_TO_ID

    if isinstance(x, Const):
        return torch.as_tensor((SEMANTIC_PIECE_TO_ID[x],), dtype=torch.long)
    elif isinstance(x, BinaryPredicate):
        fst_ids = convert_semantics_to_tensors_for_tre_model(x.fst)
        snd_ids = convert_semantics_to_tensors_for_tre_model(x.snd)
        return (fst_ids, snd_ids)
    else:
        raise ValueError(f"Unexpected Argument {x}")


class SemanticsDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    data: List[Sem]
    data_ids: List[torch.Tensor]
    max_len: int

    def __init__(
        self,
        data: List[Sem],
    ) -> None:
        super().__init__()
        self.data = data

        non_padded_ids = [
            convert_semantics_to_ids_for_seq2seq_model(x)
            for x in data
        ]
        self.max_len = max(len(x) for x in non_padded_ids)
        self.data_ids = [
            torch.as_tensor(x + [0] * (self.max_len - len(x)))
            for x in non_padded_ids
        ]

    @classmethod
    def create(
        cls,
        n_predicates: int,
        random_seed: int,
        test_p: float,
    ):
        assert n_predicates >= 0
        assert 0 <= test_p < 1

        full_data_list = enumerate_semantics(n_predicates)

        RandomState(random_seed).shuffle(full_data_list)  # type: ignore

        split_index = int((1 - test_p) * len(full_data_list))

        train_data_list = full_data_list[:split_index]
        test_data_list = full_data_list[split_index:]

        train_dataset = cls(train_data_list)
        test_dataset = cls(test_data_list)
        full_dataset = cls(full_data_list)

        return train_dataset, test_dataset, full_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        dummy_label = torch.zeros(1)
        return (self.data_ids[index], dummy_label, self.data_ids[index])

    def iterator_for_evaluation(self):
        return zip(self.data_ids, self.data)


if __name__ == "__main__":
    print("Dataset sizes:")
    for i in range(7):
        print("\tlen(enumerate_semantics({}))={}".format(i, len(enumerate_semantics(i))))
