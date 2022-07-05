import json
from typing import Sequence, Any, List, Literal, Dict, Optional
import editdistance
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr

import egg.core as core

from ...semantics.semantics import SemanticsDataset
from ..msg import get_mask


_ACC = "acc"
_LOSS = "loss"
_INPUT = "input"
_SEMANTICS = "semantics"
_MESSAGE = "message"
_OUTPUT = "output"
_SPLIT = "split"


class AskSender(core.Callback):
    split_to_dataset: Dict[Literal["train", "valid", "test"], SemanticsDataset]
    device: torch.device
    freq: Optional[int]

    def __init__(
        self,
        split_to_dataset: Dict[Literal["train", "valid", "test"], SemanticsDataset],
        device: torch.device,
        freq: Optional[int] = 1,
    ):
        self.split_to_dataset = split_to_dataset
        self.device = device
        self.freq = freq

    def on_train_begin(self, trainer_instance: Any):
        self.trainer = trainer_instance
        self.epoch_counter = self.trainer.start_epoch

    def on_train_end(self):
        self.dump()

    def on_epoch_end(self, *stuff: Any):
        self.epoch_counter += 1
        if self.freq is not None and self.epoch_counter % self.freq == 0:
            self.dump()

    def cut_eos(self, x: Sequence[int]):
        return x[:(x.index(0) if 0 in x else None)]

    def ask_sender(self):
        data: defaultdict[str, List[Any]] = defaultdict(list)
        game = self.trainer.game
        game.eval()
        with torch.no_grad():
            for split, dataset in self.split_to_dataset.items():
                for sample, sem in dataset.iterator_for_evaluation():
                    input_s = torch.stack([sample.to(self.device)])
                    output_s = game.sender(input_s)[0]
                    output_r = game.recver(output_s)[0]
                    loss, rest = game.loss(
                        input_s, output_s, None, output_r, None
                    )
                    acc: torch.Tensor = rest["acc"]
                    output_s = output_s * get_mask(output_s)
                    data[_ACC].append(acc.item())
                    data[_LOSS].append(loss.item())
                    data[_INPUT].append(sample.tolist())
                    data[_MESSAGE].append(output_s.tolist()[0])
                    data[_OUTPUT].append(output_r.tolist()[0])
                    data[_SEMANTICS].append(sem)
                    data[_SPLIT].append(split)
        game.train()
        return dict(data)

    def dump(self):
        pass


class Metrics(AskSender):
    def dump(self):
        data: pd.DataFrame = pd.DataFrame(self.ask_sender())
        data: pd.DataFrame = data[data[_SPLIT] == "train"]
        msg: List[List[int]] = data[_MESSAGE].tolist()
        ipt: List[List[int]] = data[_INPUT].tolist()
        msg_dist: List[int] = []
        ipt_dist: List[int] = []
        for i in range(len(msg)):
            for j in range(i + 1, len(msg)):
                msg_dist.append(editdistance.eval(self.cut_eos(msg[i]), self.cut_eos(msg[j])))
                ipt_dist.append(editdistance.eval(self.cut_eos(ipt[i]), self.cut_eos(ipt[j])))
        topsim: float = spearmanr(msg_dist, ipt_dist).correlation
        output = {
            "mode": "metric",
            "epoch": self.epoch_counter,
            "topsim": topsim,
        }
        print(json.dumps(output, default=repr), flush=True)


class DumpCorpus(AskSender):
    def on_train_begin(self, trainer_instance: Any):
        super().on_train_begin(trainer_instance)
        self.dump(mode="dataset")

    def dump(
        self,
        mode: Literal["dataset", "language"] = "language",
    ):
        assert mode in {"dataset", "language"}
        data = self.ask_sender()
        if mode == "dataset":
            output = {
                "mode": mode,
                "data": {
                    k: v for k, v in data.items()
                    if k in {_INPUT, _SEMANTICS, _SPLIT}
                },
            }
        else:
            output = {
                "mode": mode,
                "epoch": self.epoch_counter,
                "data": {
                    k: v for k, v in data.items()
                    if k in {_MESSAGE, _OUTPUT, _ACC, _LOSS}
                },
            }
        print(json.dumps(output, default=repr).replace(" ", ""), flush=True)


class Evaluator(AskSender):
    def dump(self):
        data: pd.DataFrame = pd.DataFrame(self.ask_sender())
        output = {
            "mode": "evaluation",
            "epoch": self.epoch_counter,
            "train_acc": data[data[_SPLIT] == "train"][_ACC].mean(),
            "train_loss": data[data[_SPLIT] == "train"][_LOSS].mean(),
            "valid_acc": data[data[_SPLIT] == "valid"][_ACC].mean(),
            "valid_loss": data[data[_SPLIT] == "valid"][_LOSS].mean(),
            "test_acc": data[data[_SPLIT] == "test"][_ACC].mean(),
            "test_loss": data[data[_SPLIT] == "test"][_LOSS].mean(),
        }
        print(json.dumps(output, default=repr), flush=True)


class PeriodicAgentResetter(core.Callback):
    sender_life_span: Optional[int]
    recver_life_span: Optional[int]
    sender: nn.Module
    recver: nn.Module

    def __init__(
        self,
        sender_life_span: Optional[int],
        recver_life_span: Optional[int],
    ) -> None:
        super().__init__()
        self.sender_life_span = sender_life_span
        self.recver_life_span = recver_life_span

    def on_train_begin(self, trainer_instance: Any) -> None:
        self.trainer = trainer_instance
        self.epoch = self.trainer.start_epoch
        self.sender = self.trainer.game.sender
        self.recver = self.trainer.game.recver

    def on_epoch_begin(self, *stuff) -> None:
        # Assume that epoch begins with 1, not 0.
        if self.sender_life_span is not None and (self.epoch - 1) % self.sender_life_span == 0:
            self.__reset_module_parameters(self.sender)
        if self.recver_life_span is not None and (self.epoch - 1) % self.recver_life_span == 0:
            self.__reset_module_parameters(self.recver)

    def on_epoch_end(self, *stuff: Any):
        self.epoch += 1

    def __reset_module_parameters(self, m: nn.Module):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
        for child in m.children():
            self.__reset_module_parameters(child)
