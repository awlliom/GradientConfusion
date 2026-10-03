from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.ticker import FuncFormatter, MaxNLocator

from attacks.score.square_attack import SquareAttack
from datasets.dataset import Dataset
from utils.compute import l2_proj_maker
from utils.defense import softmax_gradient_torch_vectorized
from utils.misc import create_dir
from utils.model_loader import load_torch_models, load_torch_models_imagesub


warnings.filterwarnings("ignore", message="delta_grad == 0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append("../..")


NUM_EVAL_EXAMPLES = 10
MAX_LOSS_QUERIES = 1000
BATCH_SIZE = 10
DEFENSES = ["aaa", "rls"]


def is_cifar(dset_name: str) -> bool:
    return dset_name in ("cifar10", "cifar10aug")


def is_imagenet(dset_name: str) -> bool:
    return dset_name in ("imagenet", "imagenet_sub")


def prepare_eval_tensor(xs, dset_name: str) -> torch.Tensor:
    if isinstance(xs, torch.Tensor):
        x_eval = xs.permute(0, 3, 1, 2)
    else:
        x_eval = torch.FloatTensor(xs.transpose(0, 3, 1, 2))
    if is_cifar(dset_name):
        x_eval = x_eval / 255.0
    return x_eval


def get_epsilon(config, dset_name: str) -> float:
    epsilon = config["attack_config"]["epsilon"]
    if is_cifar(dset_name):
        epsilon = epsilon / 255.0
    return epsilon


def load_model_for_dataset(model_name: str, dset_name: str) -> torch.nn.Module:
    if is_imagenet(dset_name):
        return load_torch_models_imagesub(model_name)
    return load_torch_models(model_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def compute_batch_jv_chunked_with_components(model, batch_images, chunk_size=10):
    """
    Compute A^T v (ATv) given v derived from model outputs (defended model).
    This does not compute b* for AAA/RLS defenses.
    """
    B = batch_images.shape[0]
    all_probs = []
    all_ATv = []
    all_logits = []

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        images_chunk = batch_images[start:end].clone().detach().requires_grad_(True)

        logits = model(images_chunk)
        all_logits.append(logits.detach())
        probs = F.softmax(logits, dim=1)

        _, sorted_indices = torch.sort(probs, dim=1, descending=True)
        i = sorted_indices[:, 0]
        j = sorted_indices[:, 1]

        batch_indices = torch.arange(end - start, device=batch_images.device)
        probs_i = probs[batch_indices, i]
        probs_j = probs[batch_indices, j]

        s = probs_i - probs_j

        # v: gradient w.r.t. logits, shape (chunk, num_classes)
        v = torch.autograd.grad(torch.sum(s), logits, retain_graph=True)[0]

        # Compute A^T v (VJP) - gradient w.r.t. images
        ATv = torch.autograd.grad(
            outputs=logits,
            inputs=images_chunk,
            grad_outputs=v,
            retain_graph=True,
        )[0]

        all_probs.append(probs.detach())
        all_ATv.append(ATv.detach())

        del logits, probs, i, j, probs_i, probs_j, s, v, ATv
        torch.cuda.empty_cache()

    return (
        torch.cat(all_probs, dim=0),
        torch.cat(all_ATv, dim=0),
        torch.cat(all_logits, dim=0),
    )


def compute_batch_jv_chunked_with_v(model, batch_images, v_batch, chunk_size=10):
    """
    Compute A A^T v and A^T v given an external v (logit-space) batch.
    v_batch: shape (B, num_classes)
    """
    B = batch_images.shape[0]
    all_ATv = []

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        images_chunk = batch_images[start:end].clone().detach().requires_grad_(True)
        v_chunk = v_batch[start:end]

        logits = model(images_chunk)

        ATv = torch.autograd.grad(
            outputs=logits,
            inputs=images_chunk,
            grad_outputs=v_chunk,
            retain_graph=True,
        )[0]

        all_ATv.append(ATv.detach())

        del logits, ATv
        torch.cuda.empty_cache()

    return torch.cat(all_ATv, dim=0)


def calculate_angle(vec1, vec2):
    """
    Calculate angle in degrees between two vectors.
    Both vectors should be flattened.
    """
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()

    vec1_norm = vec1_flat / (torch.norm(vec1_flat) + 1e-10)
    vec2_norm = vec2_flat / (torch.norm(vec2_flat) + 1e-10)

    cos_angle = torch.dot(vec1_norm, vec2_norm)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angle_rad = torch.acos(cos_angle)
    angle_deg = angle_rad * 180.0 / np.pi

    return angle_deg.item()


def softmax_gradient_from_probs(probs, i, j):
    """
    Compute the softmax gradient vector B given probabilities and top-2 indices.
    """
    y = probs.detach()
    # softmax_gradient_torch_vectorized is written for numpy but works with tensors too.
    return softmax_gradient_torch_vectorized(y, i, j)


def create_polar_histogram(
    angles_dict,
    save_path,
    title,
    show_dashed_lines=True,
    dashed_stat="median",
):
    """
    Create a polar histogram similar to the example image.
    angles_dict: dictionary with keys as sample indices and values as lists of angles
    """
    all_angles = []
    for _, angles in angles_dict.items():
        all_angles.extend(angles)

    all_angles = np.array(all_angles)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")

    n_bins = 36  # 10-degree bins
    bins = np.linspace(0, 180, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_centers_rad = np.deg2rad(bin_centers)
    bin_width = np.deg2rad(bins[1] - bins[0])

    sample_items = sorted(angles_dict.items(), key=lambda x: x[0])
    num_series = len(sample_items)
    cmap_name = "tab10" if num_series <= 10 else "tab20"
    cmap = cm.get_cmap(cmap_name, num_series)
    max_count = 0
    stat_lines = []

    min_alpha = 0.25
    max_alpha = 0.7
    alpha_span = max_alpha - min_alpha
    denom = max(1, num_series - 1)

    for idx, (sample_idx, angles) in enumerate(sample_items):
        angles = np.array(angles)
        hist, _ = np.histogram(angles, bins=bins)
        max_count = max(max_count, hist.max() if hist.size else 0)
        color = cmap(idx)
        alpha = max_alpha - (alpha_span * idx / denom)
        ax.bar(
            bin_centers_rad,
            hist,
            width=bin_width,
            bottom=0.0,
            color=color,
            edgecolor=color,
            linewidth=0.6,
            alpha=alpha,
            label=str(sample_idx),
            zorder=2,
        )
        if angles.size:
            if dashed_stat == "mean":
                stat_angle = float(np.mean(angles))
            else:
                stat_angle = float(np.median(angles))
            stat_lines.append((stat_angle, color, sample_idx))

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    r_data_max = max_count * 1.1 if max_count > 0 else 1
    ax.set_ylim(0, r_data_max)
    ax.set_rlabel_position(0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    ax.grid(True, linewidth=0.5, color="gray", alpha=0.3, zorder=1)

    if show_dashed_lines and max_count > 0 and stat_lines:
        line_end = r_data_max * 1.0
        label_base = r_data_max * 1.05
        r_step = r_data_max * 0.03 / max(1, len(stat_lines))
        for idx, (stat_angle, color, sample_idx) in enumerate(stat_lines):
            theta = np.deg2rad(stat_angle)
            ax.plot(
                [theta, theta],
                [0, line_end],
                color=color,
                linestyle="--",
                linewidth=1.2,
                zorder=3,
                clip_on=False,
            )
            ax.text(
                theta,
                label_base + idx * r_step,
                f"{stat_angle:.1f}°",
                color=color,
                fontsize=9,
                ha="center",
                va="bottom",
                zorder=4,
                clip_on=False,
            )

    # plt.title(title, fontsize=14, fontweight="bold", pad=20)

    ax.legend(
        title="Sample",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        bbox_transform=ax.transAxes,
        ncol=min(5, max(1, num_series)),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="#B0B7C3",
        facecolor="white",
        borderpad=0.4,
        handletextpad=0.6,
        labelspacing=0.4,
        borderaxespad=0.0,
    )

    mean_angle = np.mean(all_angles)
    median_angle = np.median(all_angles)
    stats_text = f"Mean: {mean_angle:.1f}°\nMedian: {median_angle:.1f}°\nTotal: {len(all_angles)}"
    # plt.text(
    #     0.02, 0.98, stats_text, transform=fig.transFigure,
    #     fontsize=10, verticalalignment="top",
    #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # )

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved histogram to {save_path}")
    print(
        f"Statistics - Mean: {mean_angle:.2f}°, "
        f"Median: {median_angle:.2f}°, Std: {np.std(all_angles):.2f}°"
    )


def run_angle_analysis_for_defense(
    defense_name,
    config,
    defense_config_path,
    original_defense_text,
    show_dashed_lines,
    dashed_stat,
):
    print("\n" + "=" * 60)
    print(f"Running angle analysis for defense: {defense_name}")
    print("=" * 60)

    set_seed(config["seed"])

    # Ensure CPU default tensors while constructing datasets/loaders.
    torch.set_default_tensor_type("torch.FloatTensor")

    dset_name = config["dset_name"]
    dset = Dataset(dset_name, config)
    model_name = config["modeln"]
    model_defended = load_model_for_dataset(model_name, dset_name)
    model_grad = model_defended

    if defense_name == "aaa":
        # AAA defense is implemented with an inner optimization on logits that
        # assumes no_grad in the outer forward. To compute A/ATv we use the
        # undefended model for gradients and the AAA model for logits/probs.
        base_defense_config = json.loads(original_defense_text)
        base_defense_config["defense"] = "none"
        defense_config_path.write_text(json.dumps(base_defense_config, indent=4))
        model_grad = load_model_for_dataset(model_name, dset_name)
        # Restore current defense setting for any later loads.
        restored_config = json.loads(original_defense_text)
        restored_config["defense"] = defense_name
        defense_config_path.write_text(json.dumps(restored_config, indent=4))

    print(f"Testing on model: {model_name}")
    print(f"Number of samples: {config['num_eval_examples']}")
    print(f"Number of iterations: {config['attack_config']['max_loss_queries']}")

    p_norm = config["attack_config"]["p"]
    epsilon = get_epsilon(config, dset_name)

    if "gpu" in config["device"] and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    def cw_loss(logit, label, target=False):
        if target:
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(label)
            second_max_index = target_is_max.long() * argsort[:, 1] + (
                ~target_is_max
            ).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label)
            second_max_index = gt_is_max.long() * argsort[:, 1] + (
                ~gt_is_max
            ).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    criterion = cw_loss

    attacker = SquareAttack(
        **config["attack_config"],
        lb=dset.min_value,
        ub=dset.max_value,
    )

    target = config["target"]

    angles_Bopt_ATv = {}

    num_eval_examples = config["num_eval_examples"]
    eval_batch_size = config["attack_config"]["batch_size"]
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    print(f"Processing {num_batches} batch(es)")

    with torch.no_grad():
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print(f"\nProcessing batch {ibatch + 1}/{num_batches}: samples {bstart} to {bend - 1}")

            x_batch, y_batch = dset.get_eval_data(bstart, bend)
            y_batch = torch.LongTensor(y_batch).cuda()
            x_ori = prepare_eval_tensor(x_batch.copy(), dset_name).cuda()

            for sample_idx in range(bstart, bend):
                angles_Bopt_ATv[sample_idx] = []

            if p_norm != "inf":
                proj_2 = l2_proj_maker(x_ori, epsilon)

            iteration_counter = [0]

            def loss_fct(xs, es=False):
                x_eval = prepare_eval_tensor(xs, dset_name).cuda()

                if p_norm == "inf":
                    x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                else:
                    x_eval = proj_2(x_eval)
                x_eval = torch.clamp(x_eval, 0, 1)

                x_eval = x_eval.detach().requires_grad_(True)

                with torch.enable_grad():
                    model_grad.eval()
                    if defense_name == "aaa":
                        with torch.no_grad():
                            model_defended.eval()
                            logits_batch = model_defended(x_eval.cuda())
                        probs = F.softmax(logits_batch.detach(), dim=1)
                        v_list = []
                        for b in range(probs.shape[0]):
                            probs_b = probs[b].detach().squeeze()
                            i = int(torch.argmax(probs_b).item())
                            j = int(torch.argsort(probs_b)[-2].item())
                            v_list.append(softmax_gradient_from_probs(probs_b, i, j))
                        v_batch = torch.stack(v_list).to(x_eval.device)
                        ATv_batch = compute_batch_jv_chunked_with_v(
                            model_grad, x_eval.cuda(), v_batch, chunk_size=10
                        )
                    else:
                        probs, ATv_batch, logits_batch = compute_batch_jv_chunked_with_components(
                            model_grad, x_eval.cuda(), chunk_size=10
                        )

                B_opt_list = []
                for b in range(x_eval.shape[0]):
                    probs_b = probs[b].detach().squeeze()
                    i = int(torch.argmax(probs_b).item())
                    j = int(torch.argsort(probs_b)[-2].item())
                    B_opt_sample = softmax_gradient_from_probs(probs_b, i, j)
                    B_opt_list.append(B_opt_sample.to(x_eval.device))

                B_opt = torch.stack(B_opt_list)

                for b in range(x_eval.shape[0]):
                    sample_global_idx = bstart + b

                    ATv_sample = ATv_batch[b]
                    Bopt_sample = B_opt[b]

                    with torch.enable_grad():
                        x_sample = x_eval[b : b + 1].detach().clone().requires_grad_(True)
                        logits_sample = model_grad(x_sample)

                        A_Bopt = torch.autograd.grad(
                            outputs=logits_sample,
                            inputs=x_sample,
                            grad_outputs=Bopt_sample.unsqueeze(0),
                            retain_graph=False,
                            create_graph=False,
                        )[0]

                    angle2 = calculate_angle(A_Bopt, ATv_sample)

                    angles_Bopt_ATv[sample_global_idx].append(angle2)

                loss = criterion(logits_batch, y_batch, target)

                if es:
                    logits_detached = logits_batch.detach()
                    correct = torch.argmax(logits_detached, axis=1) == y_batch
                    if target:
                        return correct, loss.detach()
                    else:
                        return ~correct, loss.detach()
                else:
                    iteration_counter[0] += 1
                    if iteration_counter[0] % 100 == 0:
                        print(f"  Iteration {iteration_counter[0]}/{MAX_LOSS_QUERIES}")
                    return loss.detach()

            def early_stop_crit_fct(xs):
                x_eval = prepare_eval_tensor(xs, dset_name)
                x_eval = torch.clamp(x_eval, 0, 1)

                with torch.no_grad():
                    model_defended.eval()
                    logits = model_defended(x_eval.cuda())
                logits = logits.detach()

                correct = torch.argmax(logits, axis=1) == y_batch

                if target:
                    return correct
                else:
                    return ~correct

            attacker.run(x_batch, loss_fct, early_stop_crit_fct)
            print(f"\nBatch {ibatch + 1} complete: {attacker.result()}")

    output_dir = Path("angle_analysis_results") / f"{defense_name}_{dset_name}"
    create_dir(str(output_dir))

    data = {
        "angles_Bopt_ATv": {k: v for k, v in angles_Bopt_ATv.items()},
    }

    with open(output_dir / "angles_data.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved angle data to {output_dir / 'angles_data.json'}")

    print("\nCreating polar histograms...")
    create_polar_histogram(
        angles_Bopt_ATv,
        output_dir / "histogram_A_Bopt_vs_ATv.pdf",
        f"Histogram of Angular Deviations ({defense_name}): A·B_opt vs A^T·v",
        show_dashed_lines=show_dashed_lines,
        dashed_stat=dashed_stat,
    )

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved in: {output_dir}/")
    print("  - angles_data.json: Raw angle measurements")
    print("  - histogram_A_Bopt_vs_ATv.pdf: Angle histogram for A·B_opt vs A^T·v")
    print("=" * 60)

    is_separate_grad_model = model_grad is not model_defended
    del model_defended
    if is_separate_grad_model:
        del model_grad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Angle analysis for AAA/RLS defenses (CIFAR-10 & ImageNet)."
    )
    parser.add_argument(
        "--config",
        default="config-jsons/cifar10_square_linf_config.json",
        help="Path to attack config JSON.",
    )
    parser.add_argument(
        "--defense-config",
        default="config-jsons/defense_config.json",
        help="Path to defense_config.json.",
    )
    parser.add_argument(
        "--num-eval-examples",
        type=int,
        default=None,
        help="Override number of evaluation examples.",
    )
    parser.add_argument(
        "--max-loss-queries",
        type=int,
        default=None,
        help="Override max loss queries (iterations).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override attack batch size.",
    )
    parser.add_argument(
        "--show-dashed-lines",
        action="store_true",
        help="Show dashed median lines on the polar histogram.",
    )
    parser.add_argument(
        "--hide-dashed-lines",
        action="store_true",
        help="Hide dashed median lines on the polar histogram.",
    )
    parser.add_argument(
        "--dashed-stat",
        choices=["mean", "median"],
        default="median",
        help="Statistic for dashed lines (default: median).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    defense_config_path = Path(args.defense_config)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not defense_config_path.is_file():
        raise FileNotFoundError(f"Defense config not found: {defense_config_path}")

    with open(config_path) as config_file:
        config = json.load(config_file)

    config["num_eval_examples"] = args.num_eval_examples or NUM_EVAL_EXAMPLES
    config["attack_config"]["max_loss_queries"] = args.max_loss_queries or MAX_LOSS_QUERIES
    config["attack_config"]["batch_size"] = args.batch_size or BATCH_SIZE
    show_dashed_lines = args.show_dashed_lines or not args.hide_dashed_lines
    dashed_stat = args.dashed_stat

    original_defense_text = defense_config_path.read_text()

    try:
        for defense_name in DEFENSES:
            if defense_name == "aaa" and not torch.cuda.is_available():
                print("AAA defense requires CUDA; skipping.")
                continue

            defense_config = json.loads(original_defense_text)
            defense_config["defense"] = defense_name
            defense_config_path.write_text(json.dumps(defense_config, indent=4))

            run_angle_analysis_for_defense(
                defense_name,
                config,
                defense_config_path,
                original_defense_text,
                show_dashed_lines,
                dashed_stat,
            )
    finally:
        defense_config_path.write_text(original_defense_text)


if __name__ == "__main__":
    main()
