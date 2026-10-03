import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.dataset import Dataset
from utils.model_loader import load_torch_models, load_torch_models_imagesub
from utils.defense import compute_batch_jv_chunked, parallel_optimization

SUPPORTED_DEFENSES = {"irnd", "rfd", "ornd", "rls", "aaa", "gc"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_inputs(x_batch: np.ndarray, dset_name: str, device: torch.device) -> torch.Tensor:
    x = torch.FloatTensor(x_batch.copy().transpose(0, 3, 1, 2))
    if dset_name in ("cifar10", "cifar10aug"):
        x = x / 255.0
    return x.to(device)


def load_model_for_dataset(model_name: str, dset_name: str) -> torch.nn.Module:
    if dset_name in ("imagenet", "imagenet_sub"):
        return load_torch_models_imagesub(model_name)
    return load_torch_models(model_name)


def iter_eval_batches(dset: Dataset, num_eval_examples: int, batch_size: int):
    num_batches = int(math.ceil(num_eval_examples / batch_size))
    for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, num_eval_examples)
        yield bstart, bend


def run_probs(
    model: torch.nn.Module,
    dset: Dataset,
    dset_name: str,
    num_eval_examples: int,
    batch_size: int,
    device: torch.device,
    use_gc: bool = False,
    gc_eps: float = 0.5,
    gc_top_k: int = 5,
    gc_chunk_size: int = 10,
) -> np.ndarray:
    probs_batches = []
    model.eval()
    for bstart, bend in iter_eval_batches(dset, num_eval_examples, batch_size):
        x_batch, _ = dset.get_eval_data(bstart, bend)
        x_tensor = prepare_inputs(x_batch, dset_name, device)
        if use_gc:
            with torch.enable_grad():
                b_star, probs = compute_batch_jv_chunked(
                    model, x_tensor, chunk_size=gc_chunk_size
                )
                b_star = b_star.unsqueeze(-1)
                out = parallel_optimization(
                    b_star, probs, epsi=gc_eps, top_k=gc_top_k
                )
                batch_probs = out.float()
        else:
            with torch.no_grad():
                logits = model(x_tensor)
                batch_probs = F.softmax(logits, dim=1)
        probs_batches.append(batch_probs.detach().cpu())
    return torch.cat(probs_batches, dim=0).numpy()


def format_table(rows):
    header = "| defense | min_l1 | avg_l1 | max_l1 | min_l2 | avg_l2 | max_l2 |"
    sep = "|---|---|---|---|---|---|---|"
    body = [
        (
            "| {defense} | {min_l1:.6f} | {avg_l1:.6f} | {max_l1:.6f} | "
            "{min_l2:.6f} | {avg_l2:.6f} | {max_l2:.6f} |"
        ).format(**row)
        for row in rows
    ]
    return "\n".join([header, sep] + body)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute L1 differences between undefended and defended model probabilities."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to attack config JSON (used for dataset/model selection).",
    )
    parser.add_argument(
        "--defense-config",
        default="config-jsons/defense_config.json",
        help="Path to defense_config.json (default: config-jsons/defense_config.json).",
    )
    parser.add_argument(
        "--defenses",
        default="auto",
        help="Comma-separated defenses to run or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        default="defense_output_l1_results",
        help="Directory for outputs (default: defense_output_l1_results).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: use config attack_config.batch_size).",
    )
    parser.add_argument(
        "--num-eval-examples",
        type=int,
        default=None,
        help="Override num_eval_examples (default: use config).",
    )
    parser.add_argument(
        "--gc-eps",
        type=float,
        default=0.5,
        help="GC defense epsilon (default: 0.5).",
    )
    parser.add_argument(
        "--gc-chunk-size",
        type=int,
        default=10,
        help="GC defense chunk size (default: 10).",
    )
    parser.add_argument(
        "--gc-top-k",
        type=int,
        default=None,
        help="GC defense top_k (default: 5 for CIFAR-10, 10 for ImageNet).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config not found: {config_path}")
        return 1

    defense_config_path = Path(args.defense_config)
    if not defense_config_path.is_file():
        print(f"Defense config not found: {defense_config_path}")
        return 1

    config = json.loads(config_path.read_text())
    defense_config_text = defense_config_path.read_text()
    defense_config = json.loads(defense_config_text)

    dset_name = config["dset_name"]
    model_name = config["modeln"]
    num_eval_examples = args.num_eval_examples or config["num_eval_examples"]
    batch_size = args.batch_size or config["attack_config"]["batch_size"]

    if args.defenses == "auto":
        defenses = [
            key
            for key in defense_config.keys()
            if not key.startswith("_") and key not in {"defense", "none"}
        ]
    else:
        defenses = [d.strip() for d in args.defenses.split(",") if d.strip()]

    # Filter unsupported defenses but keep order for clarity.
    filtered_defenses = []
    for defense in defenses:
        if defense in SUPPORTED_DEFENSES:
            filtered_defenses.append(defense)
        else:
            print(f"Skipping unsupported defense: {defense}")
    defenses = filtered_defenses

    if not defenses:
        print("No defenses selected.")
        return 1

    if not torch.cuda.is_available():
        if "aaa" in defenses:
            print("AAA defense requires CUDA; remove 'aaa' or run with CUDA.")
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset = Dataset(dset_name, config)

    output_dir = Path(args.output_dir) / config_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    gc_top_k = args.gc_top_k
    if gc_top_k is None:
        gc_top_k = 10 if dset_name in ("imagenet", "imagenet_sub") else 5

    base_probs = None
    rows = []

    try:
        # Run undefended model once.
        defense_config["defense"] = "none"
        defense_config_path.write_text(json.dumps(defense_config, indent=4))

        set_seed(config.get("seed", 0))
        base_model = load_model_for_dataset(model_name, dset_name)
        base_probs = run_probs(
            base_model,
            dset,
            dset_name,
            num_eval_examples,
            batch_size,
            device,
        )
        base_path = output_dir / "probs_none.npy"
        np.save(base_path, base_probs)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run each defense once.
        for defense in defenses:
            if defense == "gc":
                # GC uses the undefended model but post-processes probabilities.
                defense_config["defense"] = "none"
            else:
                defense_config["defense"] = defense
            defense_config_path.write_text(json.dumps(defense_config, indent=4))

            set_seed(config.get("seed", 0))
            model = load_model_for_dataset(model_name, dset_name)
            use_gc = defense == "gc"
            probs = run_probs(
                model,
                dset,
                dset_name,
                num_eval_examples,
                batch_size,
                device,
                use_gc=use_gc,
                gc_eps=args.gc_eps,
                gc_top_k=gc_top_k,
                gc_chunk_size=args.gc_chunk_size,
            )
            probs_path = output_dir / f"probs_{defense}.npy"
            np.save(probs_path, probs)

            diff = probs - base_probs
            l1 = np.abs(diff).sum(axis=1)
            l2 = np.sqrt((diff ** 2).sum(axis=1))
            rows.append(
                {
                    "defense": defense,
                    "min_l1": float(l1.min()),
                    "avg_l1": float(l1.mean()),
                    "max_l1": float(l1.max()),
                    "min_l2": float(l2.min()),
                    "avg_l2": float(l2.mean()),
                    "max_l2": float(l2.max()),
                }
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        defense_config_path.write_text(defense_config_text)

    # Write tables.
    csv_path = output_dir / "l1_table.csv"
    with csv_path.open("w") as f:
        f.write("defense,min_l1,avg_l1,max_l1,min_l2,avg_l2,max_l2\n")
        for row in rows:
            f.write(
                (
                    "{defense},{min_l1:.6f},{avg_l1:.6f},{max_l1:.6f},"
                    "{min_l2:.6f},{avg_l2:.6f},{max_l2:.6f}\n"
                ).format(**row)
            )

    md_path = output_dir / "l1_table.md"
    md_path.write_text(format_table(rows))

    meta = {
        "config": str(config_path),
        "defense_config": str(defense_config_path),
        "dataset": dset_name,
        "model": model_name,
        "num_eval_examples": num_eval_examples,
        "batch_size": batch_size,
        "defenses": defenses,
        "output_dir": str(output_dir),
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(md_path.read_text())
    print(f"\nSaved: {base_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
