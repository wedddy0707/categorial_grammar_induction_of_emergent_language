from typing import List, Dict, NamedTuple, Union, Optional, Sequence, Hashable
from scipy.stats import pearsonr  # type: ignore
import argparse
import itertools
import pathlib
import matplotlib.pyplot as plt
import torch
import sys

from ..io import LogFile, make_logger
from ..corpus import Metric, TargetLanguage

logger = make_logger(__name__)


class GameConfig(NamedTuple):
    max_n_predicates: int
    max_len: int
    vocab_size: int

    def __repr__(self) -> str:
        return f"({self.max_n_predicates}, {self.max_len}, {self.vocab_size})"


NestedDict = Dict[str, Union[Hashable, List[Hashable], "NestedDict"]]


class NameSpaceForPlot:
    file_of_exp_dirs: str
    figure_save_dir: str
    exp_dirs: List[pathlib.Path]
    game_config_to_metric_scores: Dict[GameConfig, NestedDict]


def update_nested_dict(d: NestedDict, update: NestedDict):
    for update_k, update_v in update.items():
        if update_k not in d:
            d[update_k] = update_v
            continue

        v = d[update_k]

        if isinstance(v, dict) and isinstance(update_v, dict):
            update_nested_dict(v, update_v)
        elif isinstance(v, dict) or isinstance(update_v, dict):
            raise ValueError("Error")
        elif isinstance(v, list):
            if isinstance(update_v, list):
                v.extend(update_v)
            else:
                v.append(update_v)
        else:
            assert not isinstance(update_v, list), (update_k, update_v)
            d[update_k] = [v, update_v]


def get_params(params: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_of_exp_dirs", type=str, required=True)
    parser.add_argument("--figure_save_dir", type=str, default="./imgs")
    args = parser.parse_args(params, namespace=NameSpaceForPlot())

    file_of_exp_dirs = pathlib.Path(args.file_of_exp_dirs)
    with file_of_exp_dirs.open(mode="r") as f:
        args.exp_dirs = [file_of_exp_dirs.parent.parent / line.strip() for line in f.readlines()]
        print(args.exp_dirs)

    args.game_config_to_metric_scores = {}
    game_config_to_random_seeds: Dict[GameConfig, List[int]] = {}
    for exp_dir in args.exp_dirs:
        log_files = [
            LogFile(p) for p in exp_dir.glob("compo_metrics_log/*.log")
        ]
        all_config = log_files[0].extract_learning_history(
            "metric", log_files[0].max_epoch
        )[
            "config_of_emergence"
        ]
        game_config = GameConfig(
            all_config["max_n_predicates"],
            all_config["max_len"],
            all_config["vocab_size"],
        )

        if game_config not in args.game_config_to_metric_scores:
            args.game_config_to_metric_scores[game_config] = {}
        if game_config not in game_config_to_random_seeds:
            game_config_to_random_seeds[game_config] = []

        for log_file in log_files:
            metric_score_info = log_file.extract_learning_history(
                "metric", log_file.max_epoch
            )
            random_seed = metric_score_info["config_of_emergence"]["random_seed"]
            game_config_to_random_seeds[game_config].append(random_seed)
            metric_score_info.pop("config_of_emergence", None)
            update_nested_dict(
                args.game_config_to_metric_scores[game_config],
                metric_score_info,
            )

    for game_config, seeds in game_config_to_random_seeds.items():
        if len(set(seeds)) < len(seeds):
            raise ValueError(
                "Found multiple log files that contain the same random seed and the same game config. "
                f"Check out game config {game_config}."
            )

    return args


def plot_correlations_between_scores(
    game_config_to_metric_scores: Dict[GameConfig, NestedDict],
    metric_x: Metric,
    metric_y: Metric,
    target_lang: TargetLanguage = TargetLanguage.emergent,
    figname: Optional[str] = None,
    save_dir: pathlib.Path = pathlib.Path("./"),
):
    fig = plt.figure(tight_layout=True)
    ax: plt.Axes = fig.add_subplot(111)
    all_metric_scores_x = torch.as_tensor([], dtype=torch.float)
    all_metric_scores_y = torch.as_tensor([], dtype=torch.float)
    for game_config, metric_scores in game_config_to_metric_scores.items():
        metric_scores_x = metric_scores[metric_x.value][target_lang.value]
        metric_scores_y = metric_scores[metric_y.value][target_lang.value]
        assert isinstance(metric_scores_x, list)
        assert isinstance(metric_scores_y, list)
        metric_scores_x = torch.as_tensor([(x if isinstance(x, float) else 0.0) for x in metric_scores_x])
        metric_scores_y = torch.as_tensor([(y if isinstance(y, float) else 0.0) for y in metric_scores_y])
        ax.scatter(
            metric_scores_x,
            metric_scores_y,
            label=repr(game_config),
        )
        all_metric_scores_x = torch.cat([all_metric_scores_x, metric_scores_x])
        all_metric_scores_y = torch.cat([all_metric_scores_y, metric_scores_y])
    pearson_corr = pearsonr(all_metric_scores_x, all_metric_scores_y)
    ax.legend()
    ax.set_xlabel(metric_x.value)
    ax.set_ylabel(metric_y.value)
    ax.set_title(f"Pearson $r={pearson_corr[0]}$ ($p={pearson_corr[1]}$)")
    if figname is None:
        figname = "correlation_x{}_y{}_lang{}.png".format(
            metric_x.value,
            metric_y.value,
            target_lang.value,
        )
    fig.savefig((save_dir / figname).as_posix(), bbox_inches="tight")


def plot_comparisons_among_target_langs(
    game_config_to_metric_scores: Dict[GameConfig, NestedDict],
    metric: Metric,
    target_langs: Sequence[TargetLanguage],
    figname: Optional[str] = None,
    save_dir: pathlib.Path = pathlib.Path("./"),
):
    fig = plt.figure(tight_layout=True)
    ax: plt.Axes = fig.add_subplot(111)
    for game_config, metric_scores in game_config_to_metric_scores.items():
        scores = [
            metric_scores[metric.value][target_lang.value]
            for target_lang in target_langs
        ]
        assert all(isinstance(v, list) for v in scores)
        scores = [
            [float(e) if isinstance(e, float) else 0.0 for e in v]
            for v in scores
        ]
        y_data = torch.as_tensor(scores, dtype=torch.float)
        x_data = torch.arange(y_data.shape[0])
        y_avg_data = y_data.mean(dim=-1)
        y_sem_data = y_data.std(dim=-1, unbiased=True) / torch.sqrt(torch.as_tensor(y_data.shape[0])).item()
        ax.plot(
            x_data,
            y_avg_data,
            label=repr(game_config),
        )
        ax.fill_between(
            x_data,
            y_avg_data + y_sem_data,
            y_avg_data - y_sem_data,
            color=ax.get_lines()[-1].get_color(),
            alpha=0.1,
        )
    ax.legend()
    ax.set_xlabel("Language")
    ax.set_ylabel(metric.value)
    ax.set_xticks(list(range(len(target_langs))))
    ax.set_xticklabels([lang.value for lang in target_langs])
    if figname is None:
        figname = "comparison_langs_metric{}.png".format(
            metric.value,
        )
    fig.savefig((save_dir / figname).as_posix(), bbox_inches="tight")


def main(params: List[str]):
    args = get_params(params)

    figure_save_dir = pathlib.Path(args.figure_save_dir)
    figure_save_dir.mkdir(parents=True, exist_ok=True)

    for metric_x, metric_y in itertools.combinations(
        [Metric.cgf, Metric.cgl, Metric.topsim, Metric.tre], r=2,
    ):
        plot_correlations_between_scores(
            args.game_config_to_metric_scores,
            metric_x,
            metric_y,
            save_dir=figure_save_dir,
        )
    for metric in [Metric.cgf, Metric.cgl, Metric.topsim, Metric.tre]:
        plot_comparisons_among_target_langs(
            args.game_config_to_metric_scores,
            metric=metric,
            target_langs=[
                TargetLanguage.input,
                TargetLanguage.emergent,
                TargetLanguage.adjacent_swapped_1,
                TargetLanguage.shuffled,
                TargetLanguage.random,
            ],
            save_dir=figure_save_dir,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
