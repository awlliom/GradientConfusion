import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def find_configs(config_dir: Path, prefix: str) -> list[Path]:
    configs = []
    for path in sorted(config_dir.glob(f"{prefix}_*.json")):
        name = path.name
        if "_res" in name:
            continue
        if "_config" not in name:
            continue
        configs.append(path)
    return configs


def available_models(prefix: str) -> list[str]:
    if prefix == "cifar10":
        return ["resnet", "wrn", "vgg"]
    if prefix == "imagenet":
        return ["Resnet50", "ViT"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run attack scripts for all dataset configs and defense modes."
    )
    parser.add_argument(
        "--config-dir",
        default="config-jsons",
        help="Directory containing attack config JSONs (default: config-jsons).",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        choices=["all", "cifar10", "imagenet"],
        help="Which dataset configs to run (default: all).",
    )
    parser.add_argument(
        "--defense-config",
        default="config-jsons/defense_config.json",
        help="Path to defense_config.json (default: config-jsons/defense_config.json).",
    )
    parser.add_argument(
        "--defenses",
        "--defense",
        dest="defenses",
        default="auto",
        help="Comma-separated defense list (including 'none') or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--models",
        default="auto",
        help="Comma-separated model list or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--attacks",
        default="all",
        help="Comma-separated attack_name list or 'all' (default: all).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next config if a run fails.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files for a defense.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config_dir = (repo_root / args.config_dir).resolve()
    if not config_dir.is_dir():
        print(f"Config directory not found: {config_dir}", file=sys.stderr)
        return 1

    defense_config_path = (repo_root / args.defense_config).resolve()
    if not defense_config_path.is_file():
        print(f"Defense config not found: {defense_config_path}", file=sys.stderr)
        return 1

    original_defense_text = defense_config_path.read_text()
    try:
        defense_config = json.loads(original_defense_text)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {defense_config_path}: {exc}", file=sys.stderr)
        return 1

    if args.defenses == "auto":
        defenses = [
            key
            for key in defense_config.keys()
            if not key.startswith("_") and key not in {"defense", "none"}
        ]
        if not defenses:
            print(
                f"No defense entries found in {defense_config_path}",
                file=sys.stderr,
            )
            return 1
    else:
        defenses = [d.strip() for d in args.defenses.split(",") if d.strip()]
        if not defenses:
            print("No defenses specified.", file=sys.stderr)
            return 1

    runs = []
    if args.datasets in ("all", "cifar10"):
        runs.append(("cifar10", repo_root / "attack_cifar10.py"))
    if args.datasets in ("all", "imagenet"):
        runs.append(("imagenet", repo_root / "attack_imagenet.py"))

    if args.attacks == "all":
        attack_patterns = None
    else:
        raw_patterns = [a.strip() for a in args.attacks.split(",") if a.strip()]
        if not raw_patterns:
            print("No attacks specified.", file=sys.stderr)
            return 1
        attack_patterns = []
        for pattern in raw_patterns:
            try:
                attack_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                print(f"Invalid attack regex '{pattern}': {exc}", file=sys.stderr)
                return 1

    try:
        any_configs = False
        ran_any = False
        for defense in defenses:
            if defense not in defense_config:
                print(
                    f"Defense '{defense}' not found in {defense_config_path}",
                    file=sys.stderr,
                )
                if not args.continue_on_error:
                    return 1
                continue

            defense_config["defense"] = defense
            defense_config_path.write_text(
                json.dumps(defense_config, indent=4)
            )
            print(f"Defense set to: {defense}")

            for prefix, attack_script in runs:
                configs = find_configs(config_dir, prefix)
                if not configs:
                    print(
                        f"No {prefix} configs found in {config_dir}",
                        file=sys.stderr,
                    )
                    continue
                if args.models == "auto":
                    models = available_models(prefix)
                else:
                    requested = [m.strip() for m in args.models.split(",") if m.strip()]
                    allowed = set(available_models(prefix))
                    models = [m for m in requested if m in allowed]
                    invalid = [m for m in requested if m not in allowed]
                    if invalid:
                        print(
                            f"Skipping unsupported {prefix} models: {', '.join(invalid)}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error and not models:
                            return 1
                if not models:
                    print(f"No models selected for {prefix}", file=sys.stderr)
                    if not args.continue_on_error:
                        return 1
                    continue
                any_configs = True
                if not attack_script.is_file():
                    print(
                        f"{attack_script.name} not found at {attack_script}",
                        file=sys.stderr,
                    )
                    return 1

                for config_path in configs:
                    config_text = config_path.read_text()
                    try:
                        config = json.loads(config_text)
                    except json.JSONDecodeError as exc:
                        print(
                            f"Invalid JSON in {config_path}: {exc}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            return 1
                        continue

                    if "attack_name" not in config:
                        print(
                            f"Missing attack_name in {config_path}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            return 1
                        continue

                    attack_name = config["attack_name"]
                    if attack_patterns is not None:
                        if not any(p.search(attack_name) for p in attack_patterns):
                            continue

                    if "modeln" not in config:
                        print(
                            f"Missing modeln in {config_path}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            return 1
                        continue

                    try:
                        for model in models:
                            config["modeln"] = model
                            config_path.write_text(json.dumps(config, indent=2))
                            print(
                                f"Running: {attack_script.name} {config_path} "
                                f"(model={model})"
                            )
                            ran_any = True
                            try:
                                subprocess.run(
                                    [sys.executable, str(attack_script), str(config_path)],
                                    cwd=repo_root,
                                    check=True,
                                )
                            except subprocess.CalledProcessError as exc:
                                print(
                                    f"Failed: {config_path} (exit code {exc.returncode})",
                                    file=sys.stderr,
                                )
                                if not args.continue_on_error:
                                    return exc.returncode
                                continue

                            res_path = Path(f"{config_path}_res.json")
                            if not res_path.exists():
                                print(
                                    f"Result file not found: {res_path}",
                                    file=sys.stderr,
                                )
                                if not args.continue_on_error:
                                    return 1
                                continue

                            new_res_path = res_path.with_name(
                                f"{res_path.stem}_{defense}_{model}.json"
                            )
                            if new_res_path.exists() and not args.overwrite:
                                print(
                                    f"Result exists: {new_res_path}",
                                    file=sys.stderr,
                                )
                                if not args.continue_on_error:
                                    return 1
                                continue

                            res_path.replace(new_res_path)
                            print(f"Saved: {new_res_path}")
                    finally:
                        config_path.write_text(config_text)

        if not any_configs:
            return 1
        if not ran_any:
            print("No runs executed. Check your --attacks filter.", file=sys.stderr)
            return 1
    finally:
        defense_config_path.write_text(original_defense_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
