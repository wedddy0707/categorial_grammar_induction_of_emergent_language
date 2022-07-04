import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, Literal, Union, NamedTuple, Optional, List, Tuple

import enum
import pandas as pd


class Mode(enum.Enum):
    config = enum.auto()
    dataset = enum.auto()
    language = enum.auto()
    train = enum.auto()
    test = enum.auto()
    metric = enum.auto()
    evaluation = enum.auto()


class LogFile:
    def __init__(
        self,
        log_path: Union[str, pathlib.Path],
    ) -> None:
        if not isinstance(log_path, pathlib.Path):
            log_path = pathlib.Path(log_path)
        assert log_path.is_file()
        self.log_path = log_path
        self.read()

    def read(self):
        with self.log_path.open() as fileobj:
            self.lines = fileobj.readlines()
        self.jsons: "defaultdict[str, Dict[Any, Any]]" = defaultdict(dict)
        self.line_idx: "defaultdict[str, Dict[int, int]]" = defaultdict(dict)
        self.max_epoch = 0
        self.max_acc = 0.0
        self.line_number_of_mode: Dict[Mode, Dict[Optional[int], int]] = {mode: {} for mode in Mode}

        for i, line in enumerate(self.lines):
            try:
                info: Dict[str, Any] = json.loads(line)
            except ValueError:
                continue
            mode: Optional[str] = info.pop("mode", None)
            epoch: Optional[int] = info.pop("epoch", None)
            if mode == "config":
                self.line_number_of_mode[Mode.config][None] = i
                self.line_no_of_config = i
            elif mode == "dataset":
                self.line_number_of_mode[Mode.dataset][None] = i
                self.line_no_of_dataset = i
            elif mode == "language":
                self.line_number_of_mode[Mode.language][epoch] = i
            elif mode == "train":
                self.line_number_of_mode[Mode.train][epoch] = i
            elif mode == "test":
                self.line_number_of_mode[Mode.test][epoch] = i
                self.max_acc = max(self.max_acc, info["acc"])
            elif mode == "metric":
                self.line_number_of_mode[Mode.metric][epoch] = i
            elif mode == "evaluation":
                self.line_number_of_mode[Mode.evaluation][epoch] = i
            if epoch is not None:
                self.max_epoch = max(self.max_epoch, epoch)
        return self.lines

    def get_first_epoch_to_reach_acc(
        self,
        acc: float,
    ) -> Union[int, None]:
        for epoch in range(1, self.max_epoch + 1):
            info = json.loads(self.lines[self.line_number_of_mode[Mode.test][epoch]])
            if info["acc"] >= acc:
                return epoch
        return None

    def extract_corpus(self, epoch: int) -> pd.DataFrame:
        data: Dict[str, Any] = dict()
        data.update(json.loads(self.lines[self.line_number_of_mode[Mode.dataset][None]])["data"])
        data.update(json.loads(self.lines[self.line_number_of_mode[Mode.language][epoch]])["data"])
        return pd.DataFrame(data=data)

    def extract_config(self):
        config: Dict[str, Any] = json.loads(self.lines[self.line_number_of_mode[Mode.config][None]])
        names = [(str(k), type(v)) for k, v in config.items()]
        return NamedTuple("Config", names)(**config)

    def extract_learning_history(
        self,
        mode: Literal["train", "test", "metric", "evaluation"],
        epoch: int,
    ) -> Dict[str, Any]:
        assert mode in {"train", "test", "metric", "evaluation"}, mode
        info: Dict[str, Any] = json.loads(
            self.lines[
                self.line_number_of_mode[Mode.train][epoch] if mode == "train"
                else self.line_number_of_mode[Mode.test][epoch] if mode == "test"
                else self.line_number_of_mode[Mode.metric][epoch] if mode == "metric"
                else self.line_number_of_mode[Mode.evaluation][epoch]
            ]
        )
        return info


def dump_metrics(
    metrics: Dict[str, Any],
    log_file: LogFile,
    epoch: int,
    comment: Optional[str] = None
):
    """
    Print results of a compositionality measure
    such as Categorial Grammar Induction (CGI), TRE, TopSim, etc.

    Example usage:
    ```python
    log_file = LogFile("path/to/logfile")
    epoch = log_file.max_epoch
    corpus = log_file.extract_corpus(epoch)
    m = cgi.ccg.metrics.metrics_of_categorial_grammar_induction(corpus, ...)

    dump_metrics(m, log_file, epoch)
    # {"epoch": ..., "config": ..., ...}
    ```

    Parameter
    ---------
    metrics: Dict[str, List[Any]]
        Results of metrics.
    log_file: LogFile
        A LogFile object to specify from which emergent communication experiment the measure was computed.
    epoch: int
        An epoch number to specify from which emergent language in "log_file" the measure was computed.
    comment: Optional[str]
        An Optional comment to print (if any).

    Return
    ------
    None. This is a void function.
    """
    dump: Dict[str, Any] = {
        "epoch": epoch,
        "config_of_emergence": log_file.extract_config()._asdict(),
    }
    if comment is not None:
        dump["comment"] = comment
    dump.update(metrics)
    print(json.dumps(dump))


def get_logfiles(
    log_dirs: List[str]
) -> Dict[Tuple[int, int], List[LogFile]]:
    log_files: "defaultdict[Tuple[int, ...], List[LogFile]]" = defaultdict(list)
    for log_dir in map(pathlib.Path, log_dirs):
        assert log_dir.is_dir()
        for log_file in map(LogFile, log_dir.glob("*.log")):
            config = log_file.extract_config()
            key = (
                int(config.n_attributes),
                int(config.n_values),
                # int(config.n_guessable_attributes),
            )
            log_files[key].append(log_file)
            print(f"{key}, {config.random_seed}, {log_file.max_epoch}")
    return log_files
