import json
import os
from collections import defaultdict
from typing import Any, Dict, Literal, Mapping, NamedTuple, Optional

import pandas as pd


class LogFile:
    def __init__(
        self,
        log_path: str,
    ) -> None:
        assert os.path.isfile(log_path)

        self.log_path = log_path
        self.read()

    def read(self):
        with open(self.log_path, mode='r') as fileobj:
            self.lines = fileobj.readlines()
        self.jsons: 'defaultdict[str, Dict[Any, Any]]' = defaultdict(dict)
        self.line_numbs: 'defaultdict[str, Dict[int, int]]' = defaultdict(dict)
        self.max_epoch = 0
        self.min_train_loss = float('inf')
        for i, line in enumerate(self.lines):
            try:
                json_line = json.loads(line)
            except ValueError:
                continue
            if 'mode' not in json_line:
                continue
            mode = json_line['mode']
            if 'epoch' in json_line:
                self.jsons[mode]  # just touch "mode"
                epoch = json_line['epoch']
                self.line_numbs[mode][epoch] = i
                self.max_epoch = max(self.max_epoch, epoch)
            else:
                self.jsons[mode] = json_line
            if mode == 'train':
                self.min_train_loss = min(
                    self.min_train_loss,
                    json_line['original_loss'])
        return self.lines

    def write(self, path: Optional[str] = None):
        if path is None:
            path = self.log_path
        with open(path, mode='w') as fileobj:
            fileobj.writelines(self.lines)

    def extract_corpus(self, epoch: int) -> pd.DataFrame:
        data: Mapping[str, Any] = dict()
        if 'corpus' in self.jsons:
            mode = 'corpus'
            if epoch not in self.jsons[mode]:
                self.jsons[mode][epoch] = json.loads(
                    self.lines[self.line_numbs[mode][epoch]])
            data.update(self.jsons[mode][epoch]['data'])
        else:
            mode = 'language'
            if epoch not in self.jsons[mode]:
                self.jsons[mode][epoch] = json.loads(
                    self.lines[self.line_numbs[mode][epoch]])
            data.update(self.jsons['dataset']['data'])
            data.update(self.jsons[mode][epoch]['data'])
        return pd.DataFrame(data=data)

    def extract_config(self):
        jsn = self.jsons['config']
        names = [(str(k), type(v)) for k, v in jsn.items()]
        return NamedTuple('Config', names)(**jsn)

    def insert_metrics(
        self,
        epoch: int,
        metrics: Mapping[str, Any],
        reload: bool = True,
    ) -> None:
        assert isinstance(epoch, int) and 0 < epoch <= self.max_epoch, epoch
        if reload:
            self.read()
        try:
            line_numb = self.line_numbs['metric'][epoch]
        except KeyError:
            raise ValueError(f'Metric at Epoch {epoch} Not Found.')
        json_dump = json.loads(self.lines[line_numb])
        json_dump.update(metrics)
        self.lines[line_numb] = json.dumps(json_dump) + '\n'

    def extract_learning_history(
        self,
        mode: Literal['train', 'test', 'metric'],
        min_epoch: int = 1,
        convergence_epsilon: float = float('inf'),
    ) -> pd.DataFrame:
        assert mode in ('train', 'test', 'metric'), mode
        data: Mapping[str, Any] = defaultdict(list)
        for epoch in range(min_epoch, 1 + self.max_epoch):
            if epoch not in self.jsons['train']:
                self.jsons['train'][epoch] = json.loads(
                    self.lines[self.line_numbs['train'][epoch]])
            if epoch not in self.jsons[mode]:
                self.jsons[mode][epoch] = json.loads(
                    self.lines[self.line_numbs[mode][epoch]])
        for epoch, jsn_per_epoch in self.jsons[mode].items():
            if (
                epoch < min_epoch or
                self.jsons['train'][epoch]['original_loss'] > self.min_train_loss + convergence_epsilon  # noqa: E501
            ):
                continue
            for k, v in jsn_per_epoch.items():
                data[k].append(v)
        return pd.DataFrame(data=data)
