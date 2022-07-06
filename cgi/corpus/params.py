from typing import Literal
import enum


class CorpusKey:
    input: Literal["input"] = "input"
    message: Literal["message"] = "message"
    semantics: Literal["semantics"] = "semantics"
    output: Literal["output"] = "output"
    split: Literal["split"] = "split"
    acc: Literal["acc"] = "acc"
    loss: Literal["loss"] = "loss"
    sentence: Literal["sentence"] = "sentence"


class TargetLanguage(enum.Enum):
    emergent = "emergent"
    input = "input"
    shuffled = "shuffled"
    random = "random"
    adjacent_swapped = "adjacent_swapped"
    adjacent_swapped_1 = "adjacent_swapped_1"
    adjacent_swapped_2 = "adjacent_swapped_2"
    adjacent_swapped_3 = "adjacent_swapped_3"


class Metric(enum.Enum):
    emecom = "emecom"
    topsim = "TopSim"
    tre = "TRE"
    cgf = "CGF"
    cgl = "CGL"
    cgi = "CGI-general-info"
