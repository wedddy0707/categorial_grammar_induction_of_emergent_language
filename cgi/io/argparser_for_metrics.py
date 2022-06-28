from typing import Sequence, Optional, Literal, List
import argparse

from ..corpus import TargetLanguage
from .log_file_reader import LogFile
from .logger import make_logger

logger = make_logger("argparser")


class NameSpaceForMetrics:
    # Common Options for CGI, TRE, and TopSim
    log_file_path: str
    min_epoch_in_log_file: Optional[int]
    max_epoch_in_log_file: Optional[int]
    target_language_name: List[Literal["input", "emergent", "shuffled", "random", "adjacent_swapped"]]
    swap_count: int
    random_seed: int
    # Common Options for CGI and TRE
    n_trains_for_metric: int
    n_epochs_for_metric: int
    # Options for CGI
    use_tqdm: bool
    show_parses: bool
    show_lexicon: bool
    # Options for TRE
    lr_for_tre: float
    n_trains_for_tre: int
    n_epochs_for_tre: int
    # After argparsing
    log_file: LogFile
    target_language: List[TargetLanguage]


def get_params(
    params: Sequence[str],
    parser: Optional[argparse.ArgumentParser] = None,
) -> NameSpaceForMetrics:
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--log_file_path", type=str, required=True)
    parser.add_argument("--min_epoch_in_log_file", type=int, default=None)
    parser.add_argument("--max_epoch_in_log_file", type=int, default=None)
    parser.add_argument("--target_language_name",
                        type=str,
                        nargs="+",
                        default=[
                            "input",
                            "emergent",
                            "shuffled",
                            "random",
                            "adjacent_swapped",
                        ],
                        choices=(
                            "input",
                            "emergent",
                            "shuffled",
                            "random",
                            "adjacent_swapped",
                        ))
    parser.add_argument("--swap_count", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--n_epochs_for_metric", type=int, default=10)
    parser.add_argument("--n_trains_for_metric", type=int, default=1)
    parser.add_argument("--lr_for_tre", type=float, default=0.01)
    parser.add_argument("--n_epochs_for_tre", type=int, default=1000)
    parser.add_argument("--n_trains_for_tre", type=int, default=1)

    opts = parser.parse_args(params, NameSpaceForMetrics())

    opts.log_file = LogFile(opts.log_file_path)
    name_to_enum_object = {x.value: x for x in TargetLanguage}
    opts.target_language = [
        name_to_enum_object[name]
        for name in opts.target_language_name
    ]

    if opts.min_epoch_in_log_file is None:
        opts.min_epoch_in_log_file = 1
    elif opts.min_epoch_in_log_file > opts.log_file.max_epoch:
        logger.warning(
            "opts.min_epoch_in_log_file > opts.log_file.max_epoch. "
            "Automatically set opts.min_epoch_in_log_file = opts.log_file.max_epoch."
        )
        opts.min_epoch_in_log_file = opts.log_file.max_epoch

    if opts.max_epoch_in_log_file is None:
        opts.max_epoch_in_log_file = opts.log_file.max_epoch
    elif opts.min_epoch_in_log_file > opts.log_file.max_epoch:
        logger.warning(
            "opts.max_epoch_in_log_file > opts.log_file.max_epoch. "
            "Automatically set opts.max_epoch_in_log_file = opts.log_file.max_epoch."
        )
        opts.max_epoch_in_log_file = opts.log_file.max_epoch

    return opts
