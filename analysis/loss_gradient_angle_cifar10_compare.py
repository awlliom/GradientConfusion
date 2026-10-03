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

import numpy as np
import torch

from attacks.decision.boundary_attack import BoundaryAttack
from attacks.decision.evo_attack import EvolutionaryAttack
from attacks.decision.geoda_attack import GeoDAttack
from attacks.decision.hsja_attack import HSJAttack
from attacks.decision.opt_attack import OptAttack
from attacks.decision.rays_attack import RaySAttack
from attacks.decision.sign_flip_attack import SignFlipAttack
from attacks.decision.sign_opt_attack import SignOPTAttack
from attacks.score.bandit_attack import BanditAttack
from attacks.score.brusli_attack import BruSLeAttack
from attacks.score.gsba_attack import GSBAAttack
from attacks.score.nes_attack import NESAttack
from attacks.score.parsimonious_attack import ParsimoniousAttack
from attacks.score.sign_attack import SignAttack
from attacks.score.square_attack import SquareAttack
from attacks.score.zo_sign_sgd_attack import ZOSignSGDAttack
from datasets.dataset import Dataset
from utils.compute import l2_proj_maker, linf_proj_maker
from utils.defense import compute_batch_jv_chunked, parallel_optimization
from utils.misc import create_dir
from utils.model_loader import load_torch_models

warnings.filterwarnings("ignore", message="delta_grad == 0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LINEWIDTH = 1.7
COLOR_AAA = "#1565C0"
COLOR_GC = "#D32F2F"

AAA_DEFAULTS = {
    "temperature": 1.0,
    "dev": 0.5,
    "attractor_interval": 4.0,
    "calibration_loss_weight": 5.0,
    "optimizer_lr": 0.1,
    "num_iter": 100,
}


def cw_loss(logit, label, target=False):
    if target:
        _, argsort = logit.sort(dim=1, descending=True)
        target_is_max = argsort[:, 0].eq(label)
        second_max_index = target_is_max.long() * argsort[:, 1] + (~target_is_max).long() * argsort[:, 0]
        target_logit = logit[torch.arange(logit.shape[0]), label]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return target_logit - second_max_logit

    _, argsort = logit.sort(dim=1, descending=True)
    gt_is_max = argsort[:, 0].eq(label)
    second_max_index = gt_is_max.long() * argsort[:, 1] + (~gt_is_max).long() * argsort[:, 0]
    gt_logit = logit[torch.arange(logit.shape[0]), label]
    second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
    return second_max_logit - gt_logit


def _to_eval_tensor(xs, x_ori, p_norm, epsilon, proj_2):
    if isinstance(xs, torch.Tensor):
        x_eval = (xs.permute(0, 3, 1, 2) / 255.0).to(device)
    else:
        x_eval = (torch.FloatTensor(xs.transpose(0, 3, 1, 2)) / 255.0).to(device)

    if p_norm == "inf":
        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
    else:
        x_eval = proj_2(x_eval)
    return torch.clamp(x_eval, 0, 1)


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _load_aaa_params(defense_config_path):
    aaa_params = dict(AAA_DEFAULTS)
    try:
        with open(defense_config_path) as f:
            defense_cfg = json.load(f)
        aaa_cfg = defense_cfg.get("aaa", {})
        if isinstance(aaa_cfg, dict):
            for key in aaa_params:
                if key in aaa_cfg:
                    aaa_params[key] = aaa_cfg[key]
    except (OSError, ValueError, TypeError):
        pass

    aaa_params["temperature"] = float(max(1e-6, aaa_params["temperature"]))
    aaa_params["attractor_interval"] = float(max(1e-6, aaa_params["attractor_interval"]))
    aaa_params["calibration_loss_weight"] = float(max(0.0, aaa_params["calibration_loss_weight"]))
    aaa_params["optimizer_lr"] = float(max(0.0, aaa_params["optimizer_lr"]))
    aaa_params["num_iter"] = int(max(0, int(aaa_params["num_iter"])))
    aaa_params["dev"] = float(aaa_params["dev"])
    return aaa_params


def _apply_aaa_defense_logits(logits, aaa_params):
    if logits.shape[1] < 2 or aaa_params["num_iter"] <= 0:
        return logits.detach()

    logits_ori = logits.detach()
    prob_ori = torch.softmax(logits_ori / aaa_params["temperature"], dim=1)
    prob_max_ori = prob_ori.max(dim=1).values
    value, index_ori = torch.topk(logits_ori, k=2, dim=1)

    activation = logits_ori.clone().requires_grad_(True)
    batch_idx = torch.arange(activation.shape[0], device=activation.device)
    mask_first = torch.zeros_like(activation)
    mask_first[batch_idx, index_ori[:, 0]] = 1.0

    interval = aaa_params["attractor_interval"]
    margin_ori = value[:, 0] - value[:, 1]
    attractor = ((margin_ori / interval + aaa_params["dev"]).round() - aaa_params["dev"]) * interval
    target_margin = margin_ori - 0.7 * interval * torch.sin(
        (1.0 - 2.0 / interval * (margin_ori - attractor)) * torch.pi
    )

    optimizer = torch.optim.Adam([activation], lr=aaa_params["optimizer_lr"])
    for _ in range(aaa_params["num_iter"]):
        prob = torch.softmax(activation, dim=1)
        loss_calibration = ((prob * mask_first).max(dim=1).values - prob_max_ori).abs().mean()
        value_cur, _ = torch.topk(activation, k=2, dim=1)
        margin_cur = value_cur[:, 0] - value_cur[:, 1]
        loss_defense = (margin_cur - target_margin).abs().mean()
        loss = loss_defense + aaa_params["calibration_loss_weight"] * loss_calibration
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return activation.detach()


def _get_target_labels(model, x_batch, y_batch, target_type):
    logit = model(torch.FloatTensor(x_batch.transpose(0, 3, 1, 2) / 255.0).to(device)).to(device)
    if target_type == "random":
        label = torch.randint(low=0, high=logit.shape[1], size=y_batch.shape).long().to(device)
    elif target_type == "least_likely":
        label = logit.argmin(dim=1)
    elif target_type == "most_likely":
        label = torch.argsort(logit, dim=1, descending=True)[:, 1]
    elif target_type == "median":
        label = torch.argsort(logit, dim=1, descending=True)[:, 4]
    elif "label" in target_type:
        label = torch.ones_like(y_batch) * int(target_type[5:])
    else:
        raise ValueError("Unknown target_type: {}".format(target_type))
    return label.detach()


def _build_attacker(config, dset):
    return eval(config["attack_name"])(
        **config["attack_config"],
        lb=dset.min_value,
        ub=dset.max_value,
    )


def _set_attacker_proj(attacker, xs_t):
    if attacker.p == "2":
        _proj = l2_proj_maker(xs_t, attacker.epsilon)
        attacker._proj = lambda x: torch.clamp(_proj(x), attacker.lb, attacker.ub)
    else:
        _proj = linf_proj_maker(xs_t, attacker.epsilon)
        attacker._proj = lambda x: torch.clamp(_proj(x), attacker.lb, attacker.ub)


def _softmax_v_from_probs(probs):
    if probs.shape[1] < 2:
        return torch.zeros_like(probs)
    _, sorted_indices = torch.sort(probs, dim=1, descending=True)
    top1 = sorted_indices[:, 0]
    top2 = sorted_indices[:, 1]
    batch_idx = torch.arange(probs.shape[0], device=probs.device)

    delta = probs[batch_idx, top2] - probs[batch_idx, top1]
    v = delta.unsqueeze(1) * probs
    v[batch_idx, top1] += probs[batch_idx, top1]
    v[batch_idx, top2] -= probs[batch_idx, top2]
    return v


def _cw_grad_wrt_scores(scores, label, target=False):
    batch_idx = torch.arange(scores.shape[0], device=scores.device)
    v = torch.zeros_like(scores)

    _, argsort = scores.sort(dim=1, descending=True)
    top1 = argsort[:, 0]
    top2 = argsort[:, 1]

    if target:
        target_is_max = top1.eq(label)
        max_other = target_is_max.long() * top2 + (~target_is_max).long() * top1
        v[batch_idx, label] += 1.0
        v[batch_idx, max_other] -= 1.0
    else:
        gt_is_max = top1.eq(label)
        max_other = gt_is_max.long() * top2 + (~gt_is_max).long() * top1
        v[batch_idx, max_other] += 1.0
        v[batch_idx, label] -= 1.0
    return v


def _batched_angle_degrees(a, b, eps=1e-12):
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    dot = (a_flat * b_flat).sum(dim=1)
    na = torch.norm(a_flat, dim=1)
    nb = torch.norm(b_flat, dim=1)
    valid = (na > eps) & (nb > eps)
    out = torch.full_like(dot, float("nan"))
    cos = torch.zeros_like(dot)
    cos[valid] = (dot[valid] / (na[valid] * nb[valid])).clamp(-1.0, 1.0)
    out[valid] = torch.acos(cos[valid]) * (180.0 / np.pi)
    return out


def _compute_angle_def_vs_undef(
    model,
    x_eval,
    y_batch,
    target,
    use_true_loss,
    scores_def=None,
    probs_def=None,
):
    with torch.enable_grad():
        x_req = x_eval.detach().clone().requires_grad_(True)
        model.eval()
        logits_base = model(x_req)

        if use_true_loss:
            # Exact CW-loss gradient for undefended model output.
            loss_undef = cw_loss(logits_base, y_batch, target).sum()
            g_undef = torch.autograd.grad(
                outputs=loss_undef,
                inputs=x_req,
                retain_graph=True,
                create_graph=False,
            )[0]

            # Defended scores are non-differentiable in this pipeline (AAA/GC),
            # so we use exact CW gradient in score-space and map it by A^T.
            if scores_def is None:
                raise ValueError("scores_def must be provided when use_true_loss=True.")
            v_def = _cw_grad_wrt_scores(
                scores_def.to(x_req.device, dtype=logits_base.dtype),
                y_batch,
                target,
            )
        else:
            if probs_def is None:
                raise ValueError("probs_def must be provided when use_true_loss=False.")
            probs_base = torch.softmax(logits_base, dim=1)
            v_base = _softmax_v_from_probs(probs_base)
            v_def = _softmax_v_from_probs(probs_def.to(x_req.device, dtype=logits_base.dtype))
            g_undef = torch.autograd.grad(
                outputs=logits_base,
                inputs=x_req,
                grad_outputs=v_base,
                retain_graph=True,
                create_graph=False,
            )[0]

        g_def = torch.autograd.grad(
            outputs=logits_base,
            inputs=x_req,
            grad_outputs=v_def,
            retain_graph=False,
            create_graph=False,
        )[0]

    return _batched_angle_degrees(g_def, g_undef)


def _run_attack_outer_angles(
    attacker,
    xs_init,
    loss_fct,
    score_probs_fct,
    x_ori,
    p_norm,
    epsilon,
    proj_2,
    model,
    max_iters,
    label,
    y_batch,
    target,
    use_true_loss,
):
    attacker.is_new_batch = True
    xs_current = xs_init
    outer_iter = 0
    num_axes = len(xs_current.shape[1:])
    batch_size = xs_current.shape[0]
    num_loss_queries = torch.zeros(batch_size, device=xs_current.device)
    outer_angles = []
    start_time = time.time()

    max_iters = int(max_iters)
    print("{} run".format(label))
    while max_iters == 0 or outer_iter < max_iters:
        if max_iters > 0 and torch.any(num_loss_queries >= max_iters):
            break

        sugg_xs_t, num_loss_queries_per_step = attacker._perturb(xs_current, loss_fct)
        num_loss_queries_per_step = num_loss_queries_per_step.to(xs_current.device, dtype=torch.float32)
        no_freeze_mask = torch.zeros(batch_size, device=xs_current.device).reshape(-1, *[1] * num_axes)
        xs_current = attacker.proj_replace(xs_current, sugg_xs_t, no_freeze_mask)
        attacker.is_new_batch = False
        outer_iter += 1
        num_loss_queries += num_loss_queries_per_step

        x_eval = _to_eval_tensor(xs_current, x_ori, p_norm, epsilon, proj_2)
        scores_def, probs_def = score_probs_fct(x_eval)
        angle_vals = _compute_angle_def_vs_undef(
            model=model,
            x_eval=x_eval,
            y_batch=y_batch,
            target=target,
            use_true_loss=use_true_loss,
            scores_def=scores_def,
            probs_def=probs_def,
        )
        outer_angles.append(float(angle_vals.detach().cpu().view(-1)[0].item()))

        elapsed = time.time() - start_time
        print(
            "Iteration :  {} ave_loss_queries :  {} ave_extra_queries :  {} ave_queries :  {} time:  {}".format(
                outer_iter,
                float(num_loss_queries.mean().item()),
                float(outer_iter),
                float(num_loss_queries.mean().item() + outer_iter),
                str(datetime.timedelta(seconds=elapsed)),
            )
        )
        sys.stdout.flush()

    return outer_angles


def _plot_angle_curves(angles_aaa, angles_gc, output_path, line_style):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    plt.figure(figsize=(8, 4))

    if angles_aaa:
        plt.plot(
            np.arange(len(angles_aaa)),
            angles_aaa,
            color=COLOR_AAA,
            linewidth=LINEWIDTH,
            linestyle=line_style,
            label="AAA vs Undefended",
        )
    if angles_gc:
        plt.plot(
            np.arange(len(angles_gc)),
            angles_gc,
            color=COLOR_GC,
            linewidth=LINEWIDTH,
            linestyle=line_style,
            label="GCD vs Undefended",
        )

    plt.xlabel("Query Iterations")
    plt.ylabel("Angle (degrees)")
    plt.ylim(0, 180)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _select_sample_indices(args, n_eval, seed):
    if args.sample_indices:
        sample_indices = [int(idx) for idx in args.sample_indices]
    elif args.sample_index is not None:
        sample_indices = [int(args.sample_index)]
    else:
        if args.num_samples <= 0:
            raise ValueError("num-samples must be positive.")
        if args.num_samples > n_eval:
            raise ValueError("num-samples={} exceeds eval set size {}.".format(args.num_samples, n_eval))
        rng = np.random.RandomState(seed)
        sample_indices = rng.choice(n_eval, size=args.num_samples, replace=False).astype(int).tolist()

    for idx in sample_indices:
        if idx < 0 or idx >= n_eval:
            raise ValueError("sample index out of range: {} (0..{})".format(idx, n_eval - 1))

    if len(sample_indices) == 0:
        raise ValueError("No sample indices were selected.")
    return sample_indices


def _aggregate_mean_trace(traces):
    if not traces:
        return []
    max_len = max(len(trace) for trace in traces) if traces else 0
    if max_len == 0:
        return []

    means = []
    for idx in range(max_len):
        vals = []
        for trace in traces:
            if idx >= len(trace):
                continue
            val = float(trace[idx])
            if np.isfinite(val):
                vals.append(val)
        if vals:
            means.append(float(np.mean(vals)))
        else:
            means.append(float("nan"))
    return means


def _finite_mean(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def _assert_trace_mean_correct(mode_name, traces_by_sample_mode, mean_trace, tol=1e-9):
    recomputed = _aggregate_mean_trace(list(traces_by_sample_mode.values()))
    if len(recomputed) != len(mean_trace):
        raise RuntimeError(
            "Mean trace length mismatch for {}: recomputed {} vs stored {}".format(
                mode_name, len(recomputed), len(mean_trace)
            )
        )

    for idx, (a, b) in enumerate(zip(recomputed, mean_trace)):
        a_f = float(a)
        b_f = float(b)
        if np.isnan(a_f) and np.isnan(b_f):
            continue
        if np.isnan(a_f) != np.isnan(b_f):
            raise RuntimeError("NaN mismatch at iter {} for {}".format(idx, mode_name))
        if abs(a_f - b_f) > tol:
            raise RuntimeError(
                "Mean mismatch at iter {} for {}: recomputed {:.12f} vs stored {:.12f}".format(
                    idx, mode_name, a_f, b_f
                )
            )


def _assert_sample_means_correct(mode_name, traces_by_sample_mode, sample_mean_angles_mode, tol=1e-9):
    for sample_key, trace in traces_by_sample_mode.items():
        recomputed = _finite_mean(trace)
        stored = sample_mean_angles_mode.get(sample_key)
        if recomputed is None and stored is None:
            continue
        if recomputed is None or stored is None:
            raise RuntimeError(
                "Sample mean missing mismatch for {} sample {}: recomputed={} stored={}".format(
                    mode_name, sample_key, recomputed, stored
                )
            )
        if abs(float(recomputed) - float(stored)) > tol:
            raise RuntimeError(
                "Sample mean mismatch for {} sample {}: recomputed {:.12f} vs stored {:.12f}".format(
                    mode_name, sample_key, float(recomputed), float(stored)
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare per-iteration angles between undefended and defended loss-gradient proxies (AAA, GCD)."
    )
    parser.add_argument("--config", default="config-jsons/cifar10_square_linf_config.json")
    parser.add_argument("--defense-config", default="config-jsons/defense_config.json")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit eval sample indices. Overrides --sample-index and --num-samples.",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--output-dir", default="angle_trace_results")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gc-epsi", type=float, default=0.5)
    parser.add_argument("--gc-top-k", type=int, default=5)
    parser.add_argument("--gc-chunk-size", type=int, default=10)
    parser.add_argument(
        "--true-loss",
        action="store_true",
        help=(
            "Use CW-loss gradient vectors for angle measurement. "
            "Undefended gradient is exact; defended gradient uses score-space CW gradient "
            "mapped through base-model Jacobian (AAA/GC outputs are non-differentiable)."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)
    aaa_params = _load_aaa_params(args.defense_config)

    decision_attacks = {
        "SignOPTAttack",
        "HSJAttack",
        "GeoDAttack",
        "OptAttack",
        "EvolutionaryAttack",
        "SignFlipAttack",
        "RaySAttack",
        "BoundaryAttack",
    }
    model_required_attacks = {"GSBAAttack", "BruSLeAttack"}
    if config["attack_name"] in decision_attacks:
        raise ValueError("This script expects a score-based attack. Pick a score attack config (for example square).")
    if config["attack_name"] in model_required_attacks:
        raise ValueError("This script uses a loss oracle only. Pick a score attack that doesn't require direct model access.")

    config["num_eval_examples"] = 1
    config["attack_config"]["batch_size"] = 1
    config["attack_config"]["max_loss_queries"] = args.max_iters

    seed = config["seed"] if args.seed is None else args.seed
    _set_seed(seed)

    if "gpu" in config["device"] and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    dset = Dataset(config["dset_name"], config)
    model = load_torch_models(config["modeln"])
    model.eval()

    eval_xs = dset.data.eval_data.xs
    n_eval = eval_xs.shape[0]
    sample_indices = _select_sample_indices(args, n_eval, seed)

    target = bool(config.get("target", False))
    p_norm = config["attack_config"]["p"]
    epsilon = config["attack_config"]["epsilon"] / 255.0

    traces_by_sample = {"aaa": {}, "gc": {}}
    sample_mean_angles = {"aaa": {}, "gc": {}}
    overall_values = {"aaa": [], "gc": []}

    for sample_pos, sample_idx in enumerate(sample_indices):
        sample_seed = seed + sample_pos
        _set_seed(sample_seed)
        print("=" * 72)
        print("Sample {} / {} (eval index: {})".format(sample_pos + 1, len(sample_indices), sample_idx))
        x_batch, y_batch_np = dset.get_eval_data(sample_idx, sample_idx + 1)
        y_batch = torch.LongTensor(y_batch_np).to(device)

        if target:
            y_batch = _get_target_labels(model, x_batch, y_batch, config.get("target_type", "random"))

        x_ori = torch.FloatTensor(x_batch.transpose(0, 3, 1, 2) / 255.0).to(device)
        proj_2 = None if p_norm == "inf" else l2_proj_maker(x_ori, epsilon)
        xs_t = torch.tensor(x_batch, dtype=torch.float32, device=device)

        def _build_mode_functions(mode):
            if mode == "aaa":

                def _score_and_probs(x_eval):
                    with torch.no_grad():
                        base_logits = model(x_eval)
                    y_score = _apply_aaa_defense_logits(base_logits, aaa_params)
                    probs = torch.softmax(y_score.detach(), dim=1)
                    return y_score, probs

            elif mode == "gc":

                def _score_and_probs(x_eval):
                    with torch.enable_grad():
                        model.eval()
                        b_star, probs_base = compute_batch_jv_chunked(model, x_eval, chunk_size=args.gc_chunk_size)
                        y_score = parallel_optimization(
                            b_star.unsqueeze(-1),
                            probs_base,
                            epsi=args.gc_epsi,
                            top_k=args.gc_top_k,
                        ).to(x_eval.device, dtype=torch.float32)
                    probs = y_score / y_score.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    return y_score, probs.detach()

            else:
                raise ValueError("Unsupported mode: {}".format(mode))

            def loss_fct(xs, es=False):
                x_eval = _to_eval_tensor(xs, x_ori, p_norm, epsilon, proj_2)
                y_score, _ = _score_and_probs(x_eval)
                loss = cw_loss(y_score, y_batch, target)
                if es:
                    y_det = y_score.detach()
                    is_correct = torch.argmax(y_det, axis=1) == y_batch
                    if target:
                        return is_correct, loss.detach()
                    return ~is_correct, loss.detach()
                return loss.detach()

            def score_probs_fct(x_eval):
                y_score, probs = _score_and_probs(x_eval)
                return y_score.detach(), probs.detach()

            return loss_fct, score_probs_fct

        for mode, label in (("aaa", "AAA"), ("gc", "GCD")):
            _set_seed(sample_seed)
            attacker = _build_attacker(config, dset)
            _set_attacker_proj(attacker, xs_t)
            loss_fct, score_probs_fct = _build_mode_functions(mode)
            trace = _run_attack_outer_angles(
                attacker=attacker,
                xs_init=xs_t.clone(),
                loss_fct=loss_fct,
                score_probs_fct=score_probs_fct,
                x_ori=x_ori,
                p_norm=p_norm,
                epsilon=epsilon,
                proj_2=proj_2,
                model=model,
                max_iters=args.max_iters,
                label=label,
                y_batch=y_batch,
                target=target,
                use_true_loss=bool(args.true_loss),
            )
            traces_by_sample[mode][str(sample_idx)] = list(trace)
            sample_mean = _finite_mean(trace)
            sample_mean_angles[mode][str(sample_idx)] = sample_mean
            if sample_mean is not None:
                print("{} vs undefended mean angle: {:.4f} deg".format(label, sample_mean))
            overall_values[mode].extend([float(v) for v in trace if np.isfinite(v)])

    create_dir(args.output_dir)
    avg_trace_aaa = _aggregate_mean_trace(list(traces_by_sample["aaa"].values()))
    avg_trace_gc = _aggregate_mean_trace(list(traces_by_sample["gc"].values()))
    max_len = max(len(avg_trace_aaa), len(avg_trace_gc))

    avg_of_sample_means_aaa = _finite_mean(
        [v for v in sample_mean_angles["aaa"].values() if v is not None and np.isfinite(v)]
    )
    avg_of_sample_means_gc = _finite_mean(
        [v for v in sample_mean_angles["gc"].values() if v is not None and np.isfinite(v)]
    )
    global_mean_aaa = _finite_mean(overall_values["aaa"])
    global_mean_gc = _finite_mean(overall_values["gc"])

    # Consistency checks so mean calculations are guaranteed correct.
    _assert_trace_mean_correct("AAA", traces_by_sample["aaa"], avg_trace_aaa)
    _assert_trace_mean_correct("GCD", traces_by_sample["gc"], avg_trace_gc)
    _assert_sample_means_correct("AAA", traces_by_sample["aaa"], sample_mean_angles["aaa"])
    _assert_sample_means_correct("GCD", traces_by_sample["gc"], sample_mean_angles["gc"])

    results = {
        "config": args.config,
        "defense_config": args.defense_config,
        "sample_index": sample_indices[0] if len(sample_indices) == 1 else None,
        "sample_indices": sample_indices,
        "num_samples": len(sample_indices),
        "attack_name": config["attack_name"],
        "iterations": list(range(max_len)),
        "angle_aaa_vs_undef": list(avg_trace_aaa),
        "angle_gc_vs_undef": list(avg_trace_gc),
        "angle_aaa_vs_undef_by_sample": traces_by_sample["aaa"],
        "angle_gc_vs_undef_by_sample": traces_by_sample["gc"],
        "sample_mean_angle_aaa_vs_undef": sample_mean_angles["aaa"],
        "sample_mean_angle_gc_vs_undef": sample_mean_angles["gc"],
        "average_of_sample_means": {
            "aaa_vs_undef": avg_of_sample_means_aaa,
            "gc_vs_undef": avg_of_sample_means_gc,
        },
        "global_mean_angle_over_all_iters_and_samples": {
            "aaa_vs_undef": global_mean_aaa,
            "gc_vs_undef": global_mean_gc,
        },
        "angle_metric_mode": "true_cw_loss" if args.true_loss else "softmax_margin_proxy",
        "gradient_mode": "proxy",
        "gradient_proxy_definition": (
            "Undefended gradient: exact CW-loss gradient w.r.t input. "
            "Defended gradient: A^T v with A=d(base_logits)/d(input) and "
            "v=d(CW loss)/d(defense scores), because AAA/GC scores are non-differentiable."
            if args.true_loss
            else "g(x)=A^T v where A=d(logits)/d(input), v is top-2 softmax-margin gradient from defense output probabilities."
        ),
        "gc_params": {
            "epsi": float(args.gc_epsi),
            "top_k": int(args.gc_top_k),
            "chunk_size": int(args.gc_chunk_size),
        },
        "aaa_params": aaa_params,
    }

    json_path = os.path.join(args.output_dir, "loss_gradient_angle_compare.json")
    line_path = os.path.join(args.output_dir, "loss_gradient_angle_compare_line.pdf")
    dotted_path = os.path.join(args.output_dir, "loss_gradient_angle_compare_dotted.pdf")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _plot_angle_curves(avg_trace_aaa, avg_trace_gc, line_path, line_style="-")
    _plot_angle_curves(avg_trace_aaa, avg_trace_gc, dotted_path, line_style=":")

    print("Saved line plot to:", line_path)
    print("Saved dotted plot to:", dotted_path)
    print("Saved angle traces to:", json_path)
    print("Sample indices:", sample_indices)
    if avg_of_sample_means_aaa is not None:
        print("AAA vs undefended average angle (mean of sample means): {:.4f} deg".format(avg_of_sample_means_aaa))
    if avg_of_sample_means_gc is not None:
        print("GCD vs undefended average angle (mean of sample means): {:.4f} deg".format(avg_of_sample_means_gc))


if __name__ == "__main__":
    main()
