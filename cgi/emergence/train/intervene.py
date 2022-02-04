import json
from typing import Sequence, Any, List, Literal
import editdistance
import torch
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr

import egg.core as core

from ..msg import get_mask


class AskSender(core.Callback):
    data: pd.DataFrame
    device: torch.device
    freq: int

    def __init__(
        self,
        data: pd.DataFrame,
        device: torch.device,
        freq: int = 1,
    ):
        assert isinstance(data, pd.DataFrame)
        assert {"split", "command_ids", "command_tensor", "tree_command"} <= set(data)
        self.data = data
        self.device = device
        self.freq = freq

    def on_train_begin(self, trainer_instance: Any):
        self.trainer = trainer_instance
        self.epoch_counter = self.trainer.start_epoch

    def on_train_end(self):
        self.dump()

    def on_epoch_end(self, *stuff):
        self.epoch_counter += 1
        if self.freq > 0 and self.epoch_counter % self.freq == 0:
            self.dump()

    def cut_eos(self, x: Sequence[int]):
        return x[:(x.index(0) if 0 in x else None)]

    def ask_sender(self):
        data: defaultdict[str, List[Any]] = defaultdict(list)
        game = self.trainer.game
        game.eval()
        with torch.no_grad():
            for sample in self.data.itertuples(index=False):
                input_s = torch.stack([sample.command_tensor.to(self.device)])
                output_s = game.sender(input_s)[0]
                output_r = game.recver(output_s)[0]
                loss, rest = game.loss(
                    input_s, output_s, None, output_r, None)
                acc = rest["acc"]
                output_s = output_s * get_mask(output_s)
                data["acc"].append(acc.item())
                data["loss"].append(loss.item())
                data["input"].append(sample.command_ids)
                data["message"].append(output_s.tolist()[0])
                data["meaning"].append(sample.tree_command)
                data["split"].append(sample.split)
        game.train()
        return data

    def dump(self):
        pass


class Metrics(AskSender):
    def dump(self):
        data: pd.DataFrame = pd.DataFrame(self.ask_sender())
        data: pd.DataFrame = data[data["split"] == "train"]
        msg: List[List[int]] = data["message"].tolist()
        ipt: List[List[int]] = data["input"].tolist()
        msg_dist: List[int] = []
        ipt_dist: List[int] = []
        for i in range(len(msg)):
            for j in range(i + 1, len(msg)):
                msg_dist.append(editdistance.eval(self.cut_eos(msg[i]), self.cut_eos(msg[j])))
                ipt_dist.append(editdistance.eval(self.cut_eos(ipt[i]), self.cut_eos(ipt[j])))
        topsim: float = spearmanr(msg_dist, ipt_dist).correlation
        output = {
            'mode': 'metric',
            'epoch': self.epoch_counter,
            'topsim': topsim,
        }
        print(json.dumps(output), flush=True)


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
                'mode': mode,
                'data': {
                    k: v for k, v in data.items()
                    if k in ('input', 'meaning', 'split')},
            }
        else:
            output = {
                'mode': mode,
                'epoch': self.epoch_counter,
                'data': {
                    k: v for k, v in data.items()
                    if k in ('message', 'acc', 'loss')},
            }
        print(json.dumps(output).replace(' ', ''), flush=True)


class Evaluator(AskSender):
    def dump(self):
        data: pd.DataFrame = pd.DataFrame(self.ask_sender())
        output = {
            "mode": "evaluation",
            "epoch": self.epoch_counter,
            "train_acc": data[data["split"] == "train"]["acc"].mean(),
            "train_loss": data[data["split"] == "train"]["loss"].mean(),
            "valid_acc": data[data["split"] == "valid"]["acc"].mean(),
            "valid_loss": data[data["split"] == "valid"]["loss"].mean(),
            "test_acc": data[data["split"] == "test"]["acc"].mean(),
            "test_loss": data[data["split"] == "test"]["loss"].mean(),
        }
        print(json.dumps(output), flush=True)
