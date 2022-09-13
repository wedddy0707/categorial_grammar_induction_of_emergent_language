from typing import List, Dict, NamedTuple, Union, Optional, Sequence, Hashable, Any, Tuple
from scipy.stats import pearsonr  # type: ignore
from collections import defaultdict
import argparse
import itertools
import pathlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys

from ..io import LogFile, make_logger
from ..corpus import Metric, TargetLanguage

logger = make_logger(__name__)


class GameConfig(NamedTuple):
    n_predicates: int
    max_len: int
    vocab_size: int

    def __repr__(self) -> str:
        return "$(\\mathcal{{D}}_{{{}}}, {}, {})$".format(
            self.n_predicates,
            self.max_len,
            self.vocab_size,
        )


NestedDict = Dict[str, Union[Hashable, List[Hashable], "NestedDict"]]


class NameSpaceForPlot:
    file_of_exp_dirs: Union[str, pathlib.Path]
    figure_save_dir: Union[str, pathlib.Path]
    exp_dirs: List[pathlib.Path]
    game_config_to_metric_scores: Dict[GameConfig, NestedDict]


def is_defined_float(x: Any):
    return \
        (x is not None) and \
        (not np.isnan(x)) and \
        (not np.isinf(x))


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
    args.figure_save_dir = pathlib.Path(args.figure_save_dir) / file_of_exp_dirs.stem

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


def aggregate_metric_scores_when_target_lang_is_input(
    game_config_to_metric_scores: Dict[GameConfig, NestedDict],
):
    n_predicates_to_metric_to_scores: "defaultdict[int, defaultdict[str, List[Hashable]]]" = defaultdict(lambda: defaultdict(list))
    for game_config, metric_scores in game_config_to_metric_scores.items():
        for metric, target_lang_to_scores in metric_scores.items():
            if isinstance(target_lang_to_scores, dict) and TargetLanguage.input.value in target_lang_to_scores:
                scores = target_lang_to_scores[TargetLanguage.input.value]
                assert isinstance(scores, list)
                n_predicates_to_metric_to_scores[game_config.n_predicates][metric].extend(scores)
    return n_predicates_to_metric_to_scores


def extract_generalization_loss(
    game_config_to_metric_scores: Dict[GameConfig, NestedDict],
):
    game_config_to_generalization_losses: Dict[GameConfig, list[Hashable]] = {}
    for game_config, metric_scores in game_config_to_metric_scores.items():
        emecom_performance = metric_scores[Metric.emecom.value]
        assert isinstance(emecom_performance, dict), emecom_performance
        test_loss = emecom_performance["test_loss"]
        assert isinstance(test_loss, list)
        game_config_to_generalization_losses[game_config] = test_loss
    return game_config_to_generalization_losses


def plot_correlations_between_generalization_loss_and_score(
    game_config_to_metric_scores: Dict[GameConfig, NestedDict],
    metrics: Sequence[Metric],
    target_lang: TargetLanguage = TargetLanguage.emergent,
    figname: Optional[str] = None,
    save_dir: pathlib.Path = pathlib.Path("./"),
):
    game_config_to_generalization_loss = extract_generalization_loss(game_config_to_metric_scores)
    fig = plt.figure(tight_layout=True)
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics), i + 1)
        all_metric_scores = torch.as_tensor([], dtype=torch.float)
        all_generalization_losses = torch.as_tensor([], dtype=torch.float)
        for game_config, metric_scores in game_config_to_metric_scores.items():
            generalization_losses_ = game_config_to_generalization_loss[game_config]
            metric_scores_ = metric_scores[metric.value][target_lang.value]
            assert isinstance(metric_scores_, list)
            metric_scores = torch.as_tensor([(x if is_defined_float(x) else 0.0) for x in metric_scores_])
            generalization_losses = torch.as_tensor([(x if is_defined_float(x) else 0.0) for x in generalization_losses_])
            ax.scatter(
                metric_scores,
                generalization_losses,
                label=repr(game_config),
                # marker={8: "o", 16: "D", 32: "*"}[game_config.vocab_size],   # type: ignore
                # color={4: "green", 8: "red"}[game_config.max_len],           # type: ignore
                # edgecolors={2: None, 3: "black"}[game_config.n_predicates],  # type: ignore
            )
            all_metric_scores = torch.cat([all_metric_scores, metric_scores])
            all_generalization_losses = torch.cat([all_generalization_losses, generalization_losses])
        pearson_corr = pearsonr(all_metric_scores, all_generalization_losses)
        # ax.legend()
        ax.set_xlabel(metric.value)
        ax.set_ylabel("Generalization Loss")
        ax.set_title(f"Pearson $r={pearson_corr[0]:.5f}$ ($p={pearson_corr[1]:.5f}$)")
    if figname is None:
        figname = "relationsToGeneralizationLoss_metrics{}_lang{}.png".format(
            "&".join(str(m.value) for m in metrics),
            str(target_lang.value).capitalize(),
        )
        figname = figname.replace("/", "")
    fig.savefig((save_dir / figname).as_posix(), bbox_inches="tight")


def plot_correlations_between_scores(
    game_config_to_metric_scores: Dict[GameConfig, NestedDict],
    metric_pairs: Sequence[Tuple[Metric, Metric]],
    # metric_x: Metric,
    # metric_y: Metric,
    target_lang: TargetLanguage = TargetLanguage.emergent,
    figname: Optional[str] = None,
    save_dir: pathlib.Path = pathlib.Path("./"),
):
    fig = plt.figure(tight_layout=True)
    for i, (metric_x, metric_y) in enumerate(metric_pairs):
        ax = fig.add_subplot(1, len(metric_pairs), i + 1)
        all_metric_scores_x = torch.as_tensor([], dtype=torch.float)
        all_metric_scores_y = torch.as_tensor([], dtype=torch.float)
        for game_config, metric_scores in game_config_to_metric_scores.items():
            metric_scores_x_ = metric_scores[metric_x.value][target_lang.value]
            metric_scores_y_ = metric_scores[metric_y.value][target_lang.value]
            assert isinstance(metric_scores_x_, list)
            assert isinstance(metric_scores_y_, list)
            metric_scores_x = torch.as_tensor([(x if is_defined_float(x) else 0.0) for x in metric_scores_x_])
            metric_scores_y = torch.as_tensor([(y if is_defined_float(y) else 0.0) for y in metric_scores_y_])
            ax.scatter(
                metric_scores_x,
                metric_scores_y,
                label=repr(game_config),
                # marker={8: "o", 16: "D", 32: "*"}[game_config.vocab_size],   # type: ignore
                # color={4: "green", 8: "red"}[game_config.max_len],           # type: ignore
                # edgecolors={2: None, 3: "black"}[game_config.n_predicates],  # type: ignore
            )
            all_metric_scores_x = torch.cat([all_metric_scores_x, metric_scores_x])
            all_metric_scores_y = torch.cat([all_metric_scores_y, metric_scores_y])
        pearson_corr = pearsonr(all_metric_scores_x, all_metric_scores_y)
        # ax.legend()
        ax.set_xlabel(metric_x.value)
        ax.set_ylabel(metric_y.value)
        ax.set_title(f"Pearson $r={pearson_corr[0]:.5f}$ ($p={pearson_corr[1]:.5f}$)")
    if figname is None:
        figname = "correlations_metricPairs{}_lang{}.png".format(
            "&".join("(" + str(pair[0].value) + "," + str(pair[1].value) + ")" for pair in metric_pairs),
            target_lang.value,
        )
        figname = figname.replace("/", "")
    fig.savefig((save_dir / figname).as_posix(), bbox_inches="tight")


def plot_comparisons_among_target_langs(
    game_config_to_metric_scores: Dict[GameConfig, NestedDict],
    metrics: Sequence[Metric],
    target_langs: Sequence[TargetLanguage],
    figname: Optional[str] = None,
    save_dir: pathlib.Path = pathlib.Path("./"),
):
    fig = plt.figure(tight_layout=True)
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics), i + 1)

        lang_to_scores: "defaultdict[TargetLanguage, List[List[float]]]" = defaultdict(list)
        for target_lang in target_langs:
            for _, metric_scores in game_config_to_metric_scores.items():
                scores: List[Hashable] = metric_scores[metric.value][target_lang.value]
                assert isinstance(scores, list)
                lang_to_scores[target_lang].append([float(e) if is_defined_float(e) else 0.0 for e in scores])

        bar_width = 0.2
        n_target_langs = len(target_langs)
        x_data = np.arange(len(game_config_to_metric_scores)) * bar_width * n_target_langs * 1.5

        for j, (k, v) in enumerate(lang_to_scores.items()):
            mean = np.array([np.mean(x) for x in v], dtype=np.float_)
            standard_error = np.array([np.std(x, ddof=1) / np.sqrt(len(x)) for x in v], dtype=np.float_)
            ax.bar(
                x_data + j * bar_width,
                mean,
                yerr=standard_error,
                label=k.value,
                width=bar_width,
            )

        ax.set_xlabel("$(\\mathcal{{I}},L,|\\mathcal{{A}}|)$")
        ax.set_ylabel(str(metric.value).capitalize())
        ax.set_xticks(x_data + (bar_width * n_target_langs / 2))
        ax.set_xticklabels(
            [repr(x) for x in game_config_to_metric_scores.keys()],
            rotation=45,
            ha="right",
        )
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        bbox_anchor=(0.5, -0.1),
        loc="upper center",
    )
    if figname is None:
        figname = "comparison_langs_metrics{}.png".format("&".join(str(m.value) for m in metrics))
        figname = figname.replace("/", "")
    fig.savefig((save_dir / figname).as_posix(), bbox_inches="tight")


def main(params: List[str]):
    args = get_params(params)

    figure_save_dir = pathlib.Path(args.figure_save_dir)
    figure_save_dir.mkdir(parents=True, exist_ok=True)

    plot_correlations_between_scores(
        args.game_config_to_metric_scores,
        metric_pairs=(
            (Metric.topsim, Metric.cgf),
            (Metric.topsim, Metric.cgl),
            (Metric.topsim, Metric.cgt),
        ),
        save_dir=figure_save_dir,
    )
    plot_correlations_between_scores(
        args.game_config_to_metric_scores,
        metric_pairs=(
            (Metric.tre, Metric.cgf),
            (Metric.tre, Metric.cgl),
            (Metric.tre, Metric.cgt),
        ),
        save_dir=figure_save_dir,
    )
    plot_correlations_between_scores(
        args.game_config_to_metric_scores,
        metric_pairs=(
            (Metric.fb_ratio, Metric.cgf),
            (Metric.fb_ratio, Metric.cgl),
            (Metric.fb_ratio, Metric.cgt),
        ),
        save_dir=figure_save_dir,
    )
    plot_correlations_between_generalization_loss_and_score(
        args.game_config_to_metric_scores,
        metrics=(
            Metric.cgf,
            Metric.cgl,
            Metric.cgt,
        ),
        save_dir=figure_save_dir,
    )
    plot_comparisons_among_target_langs(
        args.game_config_to_metric_scores,
        metrics=(
            Metric.cgf,
            Metric.cgl,
            Metric.cgt,
        ),
        target_langs=(
            TargetLanguage.input,
            TargetLanguage.emergent,
            TargetLanguage.shuffled,
            TargetLanguage.random,
        ),
        save_dir=figure_save_dir,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
