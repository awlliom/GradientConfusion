from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import json
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from attacks.decision.boundary_attack import BoundaryAttack
from attacks.decision.rays_attack import RaySAttack
from attacks.decision.sign_flip_attack import SignFlipAttack
from attacks.decision.evo_attack import EvolutionaryAttack
from attacks.decision.opt_attack import OptAttack
from attacks.decision.geoda_attack import GeoDAttack
from attacks.decision.hsja_attack import HSJAttack
from attacks.decision.sign_opt_attack import SignOPTAttack
from attacks.score.parsimonious_attack import ParsimoniousAttack
from attacks.score.square_attack import SquareAttack
from attacks.score.sign_attack import SignAttack
from attacks.score.zo_sign_sgd_attack import ZOSignSGDAttack
from attacks.score.bandit_attack import BanditAttack
from attacks.score.nes_attack import NESAttack
from attacks.score.gsba_attack import GSBAAttack
from attacks.score.brusli_attack import BruSLeAttack
from datasets.dataset import Dataset
from utils.compute import l2_proj_maker, linf_proj_maker
from utils.defense import compute_batch_jv_chunked, parallel_optimization
from utils.misc import create_dir
from utils.model_loader import load_torch_models

warnings.filterwarnings("ignore", message="delta_grad == 0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plot styling (matches loss_surface_cifar10.py)
LINEWIDTH = 1.6
COLOR_UNDEF = '#2E7D32'  # green
COLOR_DEF = '#D32F2F'    # red

# Defense settings (match defense_cifar10.py)
DEFENSE_EPSI = 0.5
DEFENSE_TOP_K = 5
DEFENSE_CHUNK = 10


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


def _to_eval_tensor(xs, x_ori, p_norm, epsilon, proj_2):
    if isinstance(xs, torch.Tensor):
        x_eval = (xs.permute(0, 3, 1, 2) / 255.).to(device)
    else:
        x_eval = (torch.FloatTensor(xs.transpose(0, 3, 1, 2)) / 255.).to(device)

    if p_norm == 'inf':
        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
    else:
        x_eval = proj_2(x_eval)
    return torch.clamp(x_eval, 0, 1)


def _normalize_traces(trace_a, trace_b, mode):
    if mode == 'none':
        return trace_a, trace_b
    combined = np.concatenate([trace_a, trace_b])
    mean = float(combined.mean())
    std = float(combined.std()) if combined.std() > 0 else 1.0
    return (trace_a - mean) / std, (trace_b - mean) / std


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _get_target_labels(model, x_batch, y_batch, target_type):
    logit = model(torch.FloatTensor(x_batch.transpose(0, 3, 1, 2) / 255.).to(device)).to(device)
    if target_type == 'random':
        label = torch.randint(low=0, high=logit.shape[1], size=y_batch.shape).long().to(device)
    elif target_type == 'least_likely':
        label = logit.argmin(dim=1)
    elif target_type == 'most_likely':
        label = torch.argsort(logit, dim=1, descending=True)[:, 1]
    elif target_type == 'median':
        label = torch.argsort(logit, dim=1, descending=True)[:, 4]
    elif 'label' in target_type:
        label = torch.ones_like(y_batch) * int(target_type[5:])
    else:
        raise ValueError("Unknown target_type: {}".format(target_type))
    return label.detach()


def _build_attacker(config, dset):
    attacker = eval(config['attack_name'])(
        **config['attack_config'],
        lb=dset.min_value,
        ub=dset.max_value
    )
    return attacker


def _set_attacker_proj(attacker, xs_t):
    if attacker.p == '2':
        _proj = l2_proj_maker(xs_t, attacker.epsilon)
        attacker._proj = lambda x: torch.clamp(_proj(x), attacker.lb, attacker.ub)
    else:
        _proj = linf_proj_maker(xs_t, attacker.epsilon)
        attacker._proj = lambda x: torch.clamp(_proj(x), attacker.lb, attacker.ub)


def _run_attack_trace(attacker, xs_init, loss_fct, loss_history, outer_loss_history, last_loss, plot_state, max_queries, label):
    attacker.is_new_batch = True
    xs_current = xs_init

    if label:
        print("{} run".format(label))

    start_time = time.time()
    outer_iter = 0
    num_axes = len(xs_current.shape[1:])
    dones_mask = torch.zeros(xs_current.shape[0], dtype=torch.bool)

    max_queries = int(max_queries)
    while max_queries == 0 or len(loss_history) < max_queries:
        sugg_xs_t, _ = attacker._perturb(xs_current, loss_fct)
        xs_current = attacker.proj_replace(
            xs_current,
            sugg_xs_t,
            dones_mask.reshape(-1, *[1] * num_axes).float()
        )
        attacker.is_new_batch = False
        outer_iter += 1

        if last_loss["val"] is not None:
            outer_loss_history.append(last_loss["val"])

        elapsed = time.time() - start_time
        elapsed_str = str(datetime.timedelta(seconds=elapsed))
        success_ratio = 1.0 if plot_state["success"] else 0.0
        failure_ratio = 1.0 - success_ratio
        print(
            "Iteration :  {} ave_loss_queries :  {} ave_extra_queries :  {} ave_queries :  {} successes :  {} failures :  {} time:  {}".format(
                outer_iter,
                float(plot_state["query_count"]),
                float(outer_iter),
                float(plot_state["query_count"] + outer_iter),
                success_ratio,
                failure_ratio,
                elapsed_str,
            )
        )
        sys.stdout.flush()

        if max_queries == 0:
            break

    return loss_history, outer_loss_history


def main():
    parser = argparse.ArgumentParser(description="Compare loss traces with and without GC defense on one CIFAR-10 sample.")
    parser.add_argument("--config", default="config-jsons/cifar10_square_linf_config.json")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--output-dir", default="loss_trace_results")
    parser.add_argument("--normalize", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    decision_attacks = {
        "SignOPTAttack",
        "HSJAttack",
        "GeoDAttack",
        "OptAttack",
        "EvolutionaryAttack",
        "SignFlipAttack",
        "RaySAttack",
        "BoundaryAttack"
    }
    model_required_attacks = {"GSBAAttack", "BruSLeAttack"}
    if config["attack_name"] in decision_attacks:
        raise ValueError("This script expects a score-based attack. Pick a score attack config (e.g. square).")
    if config["attack_name"] in model_required_attacks:
        raise ValueError("This script uses a loss oracle only. Pick a score attack that doesn't require direct model access.")

    # Override settings for single-sample tracing
    config['num_eval_examples'] = 1
    config['attack_config']['batch_size'] = 1
    config['attack_config']['max_loss_queries'] = args.max_iters

    seed = config['seed'] if args.seed is None else args.seed
    _set_seed(seed)

    if 'gpu' in config['device'] and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    dset = Dataset(config['dset_name'], config)
    model = load_torch_models(config['modeln'])
    model.eval()

    eval_xs = dset.data.eval_data.xs
    n_eval = eval_xs.shape[0]
    if args.sample_index is None:
        sample_idx = int(np.random.randint(0, n_eval))
    else:
        sample_idx = int(args.sample_index)
        if sample_idx < 0 or sample_idx >= n_eval:
            raise ValueError("sample-index out of range: {} (0..{})".format(sample_idx, n_eval - 1))

    x_batch, y_batch = dset.get_eval_data(sample_idx, sample_idx + 1)
    y_batch = torch.LongTensor(y_batch).to(device)

    target = config.get("target", False)
    if target:
        y_batch = _get_target_labels(model, x_batch, y_batch, config.get("target_type", "random"))

    p_norm = config['attack_config']['p']
    epsilon = config['attack_config']['epsilon'] / 255.

    x_ori = torch.FloatTensor(x_batch.transpose(0, 3, 1, 2) / 255.).to(device)
    proj_2 = None
    if p_norm != 'inf':
        proj_2 = l2_proj_maker(x_ori, epsilon)

    def _make_loss_fct(use_defense, fixed_class_idx):
        loss_history = []
        margin_history = []
        last_loss = {"val": None}
        plot_state = {"query_count": 0, "success": False}

        def loss_fct(xs, es=False):
            if isinstance(xs, torch.Tensor):
                x_eval = (xs.permute(0, 3, 1, 2) / 255.).to(device)
            else:
                x_eval = (torch.FloatTensor(xs.transpose(0, 3, 1, 2)) / 255.).to(device)

            if p_norm == 'inf':
                x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
            else:
                x_eval = proj_2(x_eval)
            x_eval = torch.clamp(x_eval, 0, 1)

            if use_defense:
                with torch.enable_grad():
                    model.eval()
                    b_star, probs = compute_batch_jv_chunked(model, x_eval, chunk_size=DEFENSE_CHUNK)
                    b_star = b_star.unsqueeze(-1)
                y_logit = parallel_optimization(b_star, probs, epsi=DEFENSE_EPSI, top_k=DEFENSE_TOP_K)
                prob_out = y_logit.detach()
            else:
                with torch.no_grad():
                    y_logit = model(x_eval)
                    prob_out = torch.softmax(y_logit.detach(), dim=1)

            loss = cw_loss(y_logit, y_batch, target)
            loss_val = loss.detach().view(-1)[0].item()
            loss_history.append(loss_val)
            last_loss["val"] = loss_val
            plot_state["query_count"] += 1

            fixed_prob = prob_out[:, fixed_class_idx]
            if prob_out.shape[1] > 1:
                min_val = torch.finfo(prob_out.dtype).min
                others = prob_out.clone()
                others[:, fixed_class_idx] = min_val
                second_best = others.max(dim=1).values
            else:
                second_best = torch.zeros_like(fixed_prob)
            margin_val = (fixed_prob - second_best)[0].item()
            margin_history.append(margin_val)

            pred = torch.argmax(y_logit.detach(), axis=1)
            correct = pred == y_batch
            if target:
                plot_state["success"] = bool(correct.item())
            else:
                plot_state["success"] = bool((~correct).item())

            if es:
                y_logit = y_logit.detach()
                correct = torch.argmax(y_logit, axis=1) == y_batch
                if target:
                    return correct, loss.detach()
                else:
                    return ~correct, loss.detach()
            return loss.detach()

        return loss_fct, loss_history, margin_history, last_loss, plot_state

    xs_t = torch.tensor(x_batch).to(device)

    def zscore(values):
        if not values:
            return []
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std == 0.0:
            return [0.0 for _ in values]
        return [(v - mean) / std for v in values]

    def _fixed_class_index(use_defense):
        with torch.enable_grad():
            x_init_eval = torch.clamp(x_ori.clone(), 0, 1)
            if use_defense:
                model.eval()
                b_star, probs = compute_batch_jv_chunked(model, x_init_eval, chunk_size=DEFENSE_CHUNK)
                b_star = b_star.unsqueeze(-1)
                init_probs = parallel_optimization(b_star, probs, epsi=DEFENSE_EPSI, top_k=DEFENSE_TOP_K).detach()
            else:
                init_logits = model(x_init_eval)
                init_probs = torch.softmax(init_logits.detach(), dim=1)
        return int(torch.argmax(init_probs, dim=1)[0].item())

    # Undefended attack trace (loss queries)
    _set_seed(seed)
    attacker_undef = _build_attacker(config, dset)
    _set_attacker_proj(attacker_undef, xs_t)
    fixed_class_undef = _fixed_class_index(False)
    loss_fct_undef, loss_hist_undef, margin_hist_undef, last_undef, state_undef = _make_loss_fct(
        False, fixed_class_undef
    )
    outer_loss_undef = []
    loss_trace_undef, outer_loss_undef = _run_attack_trace(
        attacker_undef,
        xs_t.clone(),
        loss_fct_undef,
        loss_hist_undef,
        outer_loss_undef,
        last_undef,
        state_undef,
        args.max_iters,
        "Undefended"
    )

    # Defended attack trace
    _set_seed(seed)
    attacker_def = _build_attacker(config, dset)
    _set_attacker_proj(attacker_def, xs_t)
    fixed_class_def = _fixed_class_index(True)
    loss_fct_def, loss_hist_def, margin_hist_def, last_def, state_def = _make_loss_fct(
        True, fixed_class_def
    )
    outer_loss_def = []
    loss_trace_def, outer_loss_def = _run_attack_trace(
        attacker_def,
        xs_t.clone(),
        loss_fct_def,
        loss_hist_def,
        outer_loss_def,
        last_def,
        state_def,
        args.max_iters,
        "Defended"
    )
    loss_trace_undef_norm = zscore([-v for v in loss_trace_undef])
    loss_trace_def_norm = zscore([-v for v in loss_trace_def])
    outer_loss_undef_norm = zscore([-v for v in outer_loss_undef])
    outer_loss_def_norm = zscore([-v for v in outer_loss_def])
    margin_trace_undef_norm = zscore(margin_hist_undef)
    margin_trace_def_norm = zscore(margin_hist_def)

    create_dir(args.output_dir)

    iterations = list(range(1, max(len(loss_trace_undef_norm), len(loss_trace_def_norm)) + 1))
    results = {
        "config": args.config,
        "sample_index": sample_idx,
        "attack_name": config['attack_name'],
        "iterations": iterations,
        "loss_undef": list(loss_trace_undef),
        "loss_def": list(loss_trace_def),
        "loss_undef_norm": list(loss_trace_undef_norm),
        "loss_def_norm": list(loss_trace_def_norm),
        "loss_undef_outer": list(outer_loss_undef),
        "loss_def_outer": list(outer_loss_def),
        "loss_undef_outer_norm": list(outer_loss_undef_norm),
        "loss_def_outer_norm": list(outer_loss_def_norm),
        "fixed_i_margin_undef": list(margin_hist_undef),
        "fixed_i_margin_def": list(margin_hist_def),
        "fixed_i_margin_undef_norm": list(margin_trace_undef_norm),
        "fixed_i_margin_def_norm": list(margin_trace_def_norm),
        "normalize_mode": "zscore",
        "defense": "gc",
        "defense_epsi": DEFENSE_EPSI,
        "defense_top_k": DEFENSE_TOP_K
    }

    json_path = os.path.join(args.output_dir, "loss_trace_compare.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(8, 4))
    if loss_trace_undef_norm:
        plt.plot(np.arange(1, len(loss_trace_undef_norm) + 1), loss_trace_undef_norm,
                 color=COLOR_UNDEF, linewidth=LINEWIDTH, label='Undefended Loss')
    if loss_trace_def_norm:
        plt.plot(np.arange(1, len(loss_trace_def_norm) + 1), loss_trace_def_norm,
                 color=COLOR_DEF, linewidth=LINEWIDTH, linestyle='--', label='Defended Loss')
    plt.xlabel('Query Iterations')
    plt.ylabel('Normalized Loss')
    # plt.title('Original and defensive normalized loss values')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "loss_trace_compare.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 4))
    if outer_loss_undef_norm:
        plt.plot(np.arange(1, len(outer_loss_undef_norm) + 1), outer_loss_undef_norm,
                 color=COLOR_UNDEF, linewidth=LINEWIDTH, label='Undefended Loss (outer)')
    if outer_loss_def_norm:
        plt.plot(np.arange(1, len(outer_loss_def_norm) + 1), outer_loss_def_norm,
                 color=COLOR_DEF, linewidth=LINEWIDTH, linestyle='--', label='Defended Loss (outer)')
    # plt.title('Original and defensive normalized loss values (outer iteration)')
    plt.xlabel('Query Iterations')
    plt.ylabel('Normalized Loss')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outer_plot_path = os.path.join(args.output_dir, "loss_trace_outer_iter_compare.pdf")
    plt.savefig(outer_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 4))
    if margin_trace_undef_norm:
        plt.plot(np.arange(1, len(margin_trace_undef_norm) + 1), margin_trace_undef_norm,
                 color=COLOR_UNDEF, linewidth=LINEWIDTH, label='Undefended Loss')
    if margin_trace_def_norm:
        plt.plot(np.arange(1, len(margin_trace_def_norm) + 1), margin_trace_def_norm,
                 color=COLOR_DEF, linewidth=LINEWIDTH, linestyle='--', label='Defended Loss')
    # plt.title('Fixed-i normalized loss values')
    plt.xlabel('Query Iterations')
    plt.ylabel('Normalized Loss')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fixed_i_plot_path = os.path.join(args.output_dir, "fixed_i_margin_compare.pdf")
    plt.savefig(fixed_i_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved plot to:", plot_path)
    print("Saved outer-iter plot to:", outer_plot_path)
    print("Saved fixed-i plot to:", fixed_i_plot_path)
    print("Saved traces to:", json_path)
    print("Sample index:", sample_idx)


if __name__ == "__main__":
    main()
