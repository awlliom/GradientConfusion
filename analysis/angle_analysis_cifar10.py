from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from attacks.score.square_attack import SquareAttack
from utils.compute import l2_proj_maker
from utils.model_loader import load_torch_models, load_torch_models_imagesub
from utils.misc import config_path_join, src_path_join, create_dir
from datasets.dataset import Dataset
import torch
import pandas as pd
import numpy as np
import sys
import time
import os
import math
import argparse
import json
import torch.nn.functional as F
from utils.defense import *
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator, FuncFormatter
warnings.filterwarnings("ignore", message="delta_grad == 0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), ".."))

sys.path.append('../..')


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


def compute_batch_jv_chunked_with_components(model, batch_images, chunk_size=10):
    """
    Modified version that returns A, A.T@v, and b* for angle calculation.
    """
    B = batch_images.shape[0]
    all_AATv = []
    all_probs = []
    all_ATv = []
    all_A_rows = []

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        images_chunk = batch_images[start:end].clone().detach().requires_grad_(True)

        logits = model(images_chunk)
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
        
        # Step 1: Compute A^T v (VJP) - gradient w.r.t. images (this is A.T @ v)
        # Shape: (chunk, C, H, W)
        ATv = torch.autograd.grad(
            outputs=logits,
            inputs=images_chunk,
            grad_outputs=v,
            retain_graph=True
        )[0]
        
        # Step 2: Compute A(A^T v) (JVP) - back to logit space (this is A @ A.T @ v = b*)
        # Shape: (chunk, num_classes)
        _, AATv_chunk = torch.autograd.functional.jvp(
            lambda x: model(x), images_chunk, ATv
        )
        
        all_AATv.append(-AATv_chunk.detach())
        all_probs.append(probs.detach())
        all_ATv.append(ATv.detach())
        
        # Store necessary info to compute A later (we'll compute it per sample as needed)
        all_A_rows.append({
            'images_chunk': images_chunk.detach(),
            'start_idx': start,
            'end_idx': end
        })

        del logits, probs, i, j, probs_i, probs_j, s, v, ATv, AATv_chunk
        torch.cuda.empty_cache()
        
    AATC = torch.cat(all_AATv, dim=0)
    
    return AATC, torch.cat(all_probs, dim=0), torch.cat(all_ATv, dim=0), all_A_rows


def compute_A_times_vector(model, image, vector_in_logit_space):
    """
    Computes A^T @ vector where A is Jacobian of logits w.r.t. inputs.
    vector_in_logit_space: shape (num_classes,) or (num_classes, 1)
    Returns: shape matching input image
    """
    image = image.clone().detach().requires_grad_(True)
    logits = model(image.unsqueeze(0))
    
    # Compute A^T @ vector (gradient w.r.t. inputs)
    vector_flat = vector_in_logit_space.squeeze()
    result = torch.autograd.grad(
        outputs=logits,
        inputs=image,
        grad_outputs=vector_flat.unsqueeze(0),
        retain_graph=False
    )[0]
    
    return result.squeeze()


def calculate_angle(vec1, vec2):
    """
    Calculate angle in degrees between two vectors.
    Both vectors should be flattened.
    """
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()
    
    # Normalize
    vec1_norm = vec1_flat / (torch.norm(vec1_flat) + 1e-10)
    vec2_norm = vec2_flat / (torch.norm(vec2_flat) + 1e-10)
    
    # Cosine similarity
    cos_angle = torch.dot(vec1_norm, vec2_norm)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle_rad = torch.acos(cos_angle)
    angle_deg = angle_rad * 180.0 / np.pi
    
    return angle_deg.item()


def create_polar_histogram(angles_dict, save_path, title, show_dashed_lines=True):
    """
    Create a polar histogram similar to the example image.
    angles_dict: dictionary with keys as sample indices and values as lists of angles
    """
    # Flatten all angles (for summary statistics)
    all_angles = []
    for _, angles in angles_dict.items():
        all_angles.extend(angles)

    all_angles = np.array(all_angles)

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')

    # Define bins (in degrees)
    n_bins = 36  # 10-degree bins
    bins = np.linspace(0, 180, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Convert to radians for polar plot
    bin_centers_rad = np.deg2rad(bin_centers)
    bin_width = np.deg2rad(bins[1] - bins[0])

    # Plot each sample's histogram with a distinct color
    sample_items = sorted(angles_dict.items(), key=lambda x: x[0])
    num_series = len(sample_items)
    cmap_name = 'tab10' if num_series <= 10 else 'tab20'
    cmap = cm.get_cmap(cmap_name, num_series)
    max_count = 0
    mean_lines = []

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
            zorder=2
        )
        if angles.size:
            mean_angle = float(np.mean(angles))
            mean_lines.append((mean_angle, color, sample_idx))

    # Styling: rotate semicircle so 0° is on the right
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Set radial ticks (integer counts / iterations per bin)
    r_data_max = max_count * 1.1 if max_count > 0 else 1
    ax.set_ylim(0, r_data_max)
    ax.set_rlabel_position(0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    # Grid
    ax.grid(True, linewidth=0.5, color='gray', alpha=0.3, zorder=1)

    # Mean angle markers per sample
    if show_dashed_lines and max_count > 0 and mean_lines:
        line_end = r_data_max * 1.0
        label_base = r_data_max * 1.05
        r_step = r_data_max * 0.03 / max(1, len(mean_lines))
        for idx, (mean_angle, color, sample_idx) in enumerate(mean_lines):
            theta = np.deg2rad(mean_angle)
            ax.plot(
                [theta, theta],
                [0, line_end],
                color=color,
                linestyle='--',
                linewidth=1.2,
                zorder=3,
                clip_on=False
            )
            ax.text(
                theta,
                label_base + idx * r_step,
                f"{mean_angle:.1f}°",
                color=color,
                fontsize=9,
                ha='center',
                va='bottom',
                zorder=4,
                clip_on=False
            )

    # Title
    # plt.title(title, fontsize=14, fontweight='bold', pad=20)

    # Legend for sample indices
    ax.legend(
        title='Sample',
        loc='lower center',
        bbox_to_anchor=(0.5, 0.1),
        bbox_transform=ax.transAxes,
        ncol=min(5, max(1, num_series)),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor='#B0B7C3',
        facecolor='white',
        borderpad=0.4,
        handletextpad=0.6,
        labelspacing=0.4,
        borderaxespad=0.0
    )

    # Add statistics text
    mean_angle = np.mean(all_angles)
    median_angle = np.median(all_angles)
    stats_text = f'Mean: {mean_angle:.1f}°\nMedian: {median_angle:.1f}°\nTotal: {len(all_angles)}'
    # plt.text(
    #     0.02, 0.98, stats_text, transform=fig.transFigure,
    #     fontsize=10, verticalalignment='top',
    #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # )

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved histogram to {save_path}")
    print(f"Statistics - Mean: {mean_angle:.2f}°, Median: {median_angle:.2f}°, Std: {np.std(all_angles):.2f}°")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Angle analysis for GC defense (CIFAR-10 & ImageNet)."
    )
    parser.add_argument(
        "--config",
        default="config-jsons/cifar10_square_linf_config.json",
        help="Path to attack config JSON.",
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
        help="Show dashed mean lines on the polar histogram.",
    )
    parser.add_argument(
        "--hide-dashed-lines",
        action="store_true",
        help="Hide dashed mean lines on the polar histogram.",
    )
    args = parser.parse_args()

    # Configuration
    config_path = args.config
    
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Override settings for testing
    config['num_eval_examples'] = args.num_eval_examples or 10  # 10 samples
    config['attack_config']['max_loss_queries'] = args.max_loss_queries or 1000  # 1000 iterations
    config['attack_config']['batch_size'] = args.batch_size or 10  # Match batch size to num samples
    show_dashed_lines = args.show_dashed_lines or not args.hide_dashed_lines
    
    # Set seed
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load dataset and model
    # Ensure CPU default tensors while constructing datasets/loaders.
    torch.set_default_tensor_type('torch.FloatTensor')

    dset = Dataset(config['dset_name'], config)
    model_name = config['modeln']
    model = load_model_for_dataset(model_name, config['dset_name'])
    
    print(f'Testing on model: {model_name}')
    print(f'Number of samples: {config["num_eval_examples"]}')
    print(f'Number of iterations: {config["attack_config"]["max_loss_queries"]}')

    p_norm = config['attack_config']['p']
    epsilon = get_epsilon(config, config['dset_name'])

    # Set device
    if 'gpu' in config['device'] and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    def cw_loss(logit, label, target=False):
        if target:
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(label)
            second_max_index = target_is_max.long() * argsort[:, 1] + (~ target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label)
            second_max_index = gt_is_max.long() * argsort[:, 1] + (~gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    criterion = cw_loss

    # Initialize attacker
    attacker = SquareAttack(
        **config['attack_config'],
        lb=dset.min_value,
        ub=dset.max_value
    )

    target = config["target"]
    
    # Storage for angles
    # Format: {sample_idx: [angle at iter 0, angle at iter 1, ...]}
    angles_bstar_ATv = {}  # Angles between A*bstar and A.T@v
    angles_Bopt_ATv = {}   # Angles between A*B_opt and A.T@v
    
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['attack_config']['batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    print(f'Processing {num_batches} batch(es)')
    
    with torch.no_grad():
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print(f'\nProcessing batch {ibatch+1}/{num_batches}: samples {bstart} to {bend-1}')

            x_batch, y_batch = dset.get_eval_data(bstart, bend)
            y_batch = torch.LongTensor(y_batch).cuda()
            x_ori = prepare_eval_tensor(x_batch.copy(), config['dset_name']).cuda()

            batch_size_actual = x_ori.shape[0]
            
            # Initialize angle storage for this batch
            for sample_idx in range(bstart, bend):
                angles_bstar_ATv[sample_idx] = []
                angles_Bopt_ATv[sample_idx] = []

            if p_norm != 'inf':
                proj_2 = l2_proj_maker(x_ori, epsilon)

            # Modify loss function to track angles
            iteration_counter = [0]  # Use list to make it mutable in nested function
            
            def loss_fct(xs, es=False):
                x_eval = prepare_eval_tensor(xs, config['dset_name']).cuda()

                if p_norm == 'inf':
                    x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                else:
                    x_eval = proj_2(x_eval)
                x_eval = torch.clamp(x_eval, 0, 1)
                
                # Ensure x_eval requires grad for angle calculations
                x_eval = x_eval.detach().requires_grad_(True)
                
                # Compute defense components with angle tracking
                with torch.enable_grad():
                    model.eval()
                    b_star, probs, ATv_batch, _ = compute_batch_jv_chunked_with_components(
                        model, x_eval.cuda(), chunk_size=10
                    )
                    b_star_unsqueezed = b_star.unsqueeze(-1)
                
                # Compute B_opt for each sample
                B_opt_list = []
                for b in range(x_eval.shape[0]):
                    bstar_np = b_star[b].detach().squeeze().cpu().numpy()
                    y_prime_np = probs[b].detach().squeeze().cpu().numpy()
                    i = int(np.argmax(y_prime_np))
                    j = int(np.argsort(y_prime_np)[-2])
                    
                    _, B_opt_sample, _ = best_effort_match_sparse(
                        bstar_np, y_prime_np, 0.5, i, j, top_k=5
                    )
                    B_opt_list.append(torch.tensor(B_opt_sample, device=b_star.device))
                
                B_opt = torch.stack(B_opt_list)
                
                # Calculate angles for each sample in batch
                for b in range(x_eval.shape[0]):
                    sample_global_idx = bstart + b
                    
                    # Get components for this sample
                    ATv_sample = ATv_batch[b]  # A.T @ v (gradient w.r.t. input)
                    bstar_sample = b_star[b]   # A @ A.T @ v
                    Bopt_sample = B_opt[b]     # B_opt vector
                    
                    # We need to compute:
                    # 1. A^T @ bstar (input-space vector)
                    # 2. A^T @ B_opt
                    
                    # For A @ bstar: we already have bstar = A @ A.T @ v
                    # So we need to apply A one more time
                    with torch.enable_grad():
                        # Create a fresh tensor for this sample with grad enabled
                        x_sample = x_eval[b:b+1].detach().clone().requires_grad_(True)
                        logits_sample = model(x_sample)
                        
                        # Compute A^T @ bstar (gradient w.r.t. input with bstar as grad_outputs)
                        A_bstar = torch.autograd.grad(
                            outputs=logits_sample,
                            inputs=x_sample,
                            grad_outputs=bstar_sample.unsqueeze(0),
                            retain_graph=True,
                            create_graph=False
                        )[0]
                        
                        # Compute A^T @ B_opt
                        A_Bopt = torch.autograd.grad(
                            outputs=logits_sample,
                            inputs=x_sample,
                            grad_outputs=Bopt_sample.unsqueeze(0),
                            retain_graph=False,
                            create_graph=False
                        )[0]
                    
                    # Calculate angles
                    angle1 = calculate_angle(A_bstar, ATv_sample)
                    angle2 = calculate_angle(A_Bopt, ATv_sample)
                    
                    angles_bstar_ATv[sample_global_idx].append(angle1)
                    angles_Bopt_ATv[sample_global_idx].append(angle2)
                
                # Get predictions from optimized probabilities
                y_logit = parallel_optimization(b_star_unsqueezed, probs, epsi=0.5, top_k=5)
                
                loss = criterion(y_logit, y_batch, target)
                
                if es:
                    y_logit = y_logit.detach()
                    correct = torch.argmax(y_logit, axis=1) == y_batch
                    if target:
                        return correct, loss.detach()
                    else:
                        return ~correct, loss.detach()
                else:
                    iteration_counter[0] += 1
                    if iteration_counter[0] % 100 == 0:
                        print(f"  Iteration {iteration_counter[0]}/1000")
                    return loss.detach()

            def early_stop_crit_fct(xs):
                x_eval = prepare_eval_tensor(xs, config['dset_name'])
                x_eval = torch.clamp(x_eval, 0, 1)
                
                with torch.enable_grad():
                    model.eval()
                    b_star, probs = compute_batch_jv_chunked(model, x_eval.cuda(), chunk_size=10)
                    b_star = b_star.unsqueeze(-1)
                y_logit = parallel_optimization(b_star, probs, epsi=0.5, top_k=5)
                y_logit = y_logit.detach()

                correct = torch.argmax(y_logit, axis=1) == y_batch

                if target:
                    return correct
                else:
                    return ~correct

            # Run attack
            logs_dict = attacker.run(x_batch, loss_fct, early_stop_crit_fct)
            print(f"\nBatch {ibatch+1} complete: {attacker.result()}")

    # Create output directory
    output_dir = f"angle_analysis_results/gc_{config['dset_name']}"
    create_dir(output_dir)
    
    # Save raw data
    data = {
        'angles_bstar_ATv': {k: v for k, v in angles_bstar_ATv.items()},
        'angles_Bopt_ATv': {k: v for k, v in angles_Bopt_ATv.items()}
    }
    
    with open(f'{output_dir}/angles_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved angle data to {output_dir}/angles_data.json")
    
    # Create histograms
    print("\nCreating polar histograms...")
    create_polar_histogram(
        angles_bstar_ATv,
        f'{output_dir}/histogram_A_bstar_vs_ATv.pdf',
        'Histogram of Angular Deviations: A·b* vs A^T·v',
        show_dashed_lines=show_dashed_lines
    )
    
    create_polar_histogram(
        angles_Bopt_ATv,
        f'{output_dir}/histogram_A_Bopt_vs_ATv.pdf',
        'Histogram of Angular Deviations: A·B_opt vs A^T·v',
        show_dashed_lines=show_dashed_lines
    )
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Results saved in: {output_dir}/")
    print(f"  - angles_data.json: Raw angle measurements")
    print(f"  - histogram_A_bstar_vs_ATv.pdf: Angle histogram for A·b* vs A^T·v")
    print(f"  - histogram_A_Bopt_vs_ATv.pdf: Angle histogram for A·B_opt vs A^T·v")
    print("="*60)
