from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import os

import numpy as np

LINEWIDTH = 1.7
COLOR_AAA = "#1565C0"
COLOR_GC = "#D32F2F"


def _sorted_keys(keys):
    def _key_fn(k):
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, str(k))

    return sorted(keys, key=_key_fn)


def _safe_float_list(values):
    if values is None:
        return []
    return [float(v) for v in values]


def _extract_traces(data):
    mean_aaa = _safe_float_list(data.get("angle_aaa_vs_undef", []))
    mean_gc = _safe_float_list(data.get("angle_gc_vs_undef", []))

    by_sample_aaa = data.get("angle_aaa_vs_undef_by_sample", {})
    by_sample_gc = data.get("angle_gc_vs_undef_by_sample", {})

    sample_traces_aaa = []
    sample_traces_gc = []

    if isinstance(by_sample_aaa, dict) or isinstance(by_sample_gc, dict):
        keys_aaa = set(by_sample_aaa.keys()) if isinstance(by_sample_aaa, dict) else set()
        keys_gc = set(by_sample_gc.keys()) if isinstance(by_sample_gc, dict) else set()
        sample_ids = _sorted_keys(keys_aaa | keys_gc)
        for key in sample_ids:
            trace_aaa = by_sample_aaa.get(key, []) if isinstance(by_sample_aaa, dict) else []
            trace_gc = by_sample_gc.get(key, []) if isinstance(by_sample_gc, dict) else []
            sample_traces_aaa.append(_safe_float_list(trace_aaa))
            sample_traces_gc.append(_safe_float_list(trace_gc))

    return mean_aaa, mean_gc, sample_traces_aaa, sample_traces_gc


def _plot_angle_curves(
    angles_aaa,
    angles_gc,
    sample_traces_aaa,
    sample_traces_gc,
    output_path,
    line_style,
    zoom_y=True,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    plt.figure(figsize=(8, 4))
    all_angles = []
    legend_handles = []

    num_samples = max(len(sample_traces_aaa), len(sample_traces_gc))
    if num_samples <= 10:
        cmap = plt.get_cmap("tab10")
    elif num_samples <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("gist_ncar")

    def _sample_color(sample_idx):
        if num_samples <= 0:
            return "#9E9E9E"
        if num_samples <= 20:
            return cmap(sample_idx % cmap.N)
        denom = max(1, num_samples - 1)
        return cmap(float(sample_idx) / float(denom))

    for sample_idx in range(num_samples):
        color = _sample_color(sample_idx)
        trace_aaa = sample_traces_aaa[sample_idx] if sample_idx < len(sample_traces_aaa) else []
        trace_gc = sample_traces_gc[sample_idx] if sample_idx < len(sample_traces_gc) else []

        if trace_aaa is not None and len(trace_aaa) > 0:
            vals = np.asarray(trace_aaa, dtype=np.float64)
            vals_finite = vals[np.isfinite(vals)]
            all_angles.extend(vals_finite.tolist())
            plt.plot(
                np.arange(len(vals)),
                vals,
                color=color,
                linewidth=max(0.8, LINEWIDTH * 0.8),
                linestyle="--",
                alpha=0.28,
                zorder=1,
            )

        if trace_gc is not None and len(trace_gc) > 0:
            vals = np.asarray(trace_gc, dtype=np.float64)
            vals_finite = vals[np.isfinite(vals)]
            all_angles.extend(vals_finite.tolist())
            plt.plot(
                np.arange(len(vals)),
                vals,
                color=color,
                linewidth=max(0.8, LINEWIDTH * 0.8),
                linestyle="-",
                alpha=0.28,
                zorder=1,
            )

    if num_samples > 0:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#666666",
                linewidth=max(0.8, LINEWIDTH * 0.8),
                linestyle="--",
                alpha=0.7,
                label="AAA sample traces (dashed)",
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#666666",
                linewidth=max(0.8, LINEWIDTH * 0.8),
                linestyle="-",
                alpha=0.7,
                label="GCD sample traces (continuous)",
            )
        )

    if angles_aaa is not None and len(angles_aaa) > 0:
        vals = np.asarray(angles_aaa, dtype=np.float64)
        vals_finite = vals[np.isfinite(vals)]
        all_angles.extend(vals_finite.tolist())
        mean_aaa_handle = plt.plot(
            np.arange(len(vals)),
            vals,
            color=COLOR_AAA,
            linewidth=LINEWIDTH,
            linestyle=line_style,
            label="AAA vs Undefended (mean)",
            zorder=3,
        )[0]
        legend_handles.append(mean_aaa_handle)

    if angles_gc is not None and len(angles_gc) > 0:
        vals = np.asarray(angles_gc, dtype=np.float64)
        vals_finite = vals[np.isfinite(vals)]
        all_angles.extend(vals_finite.tolist())
        mean_gc_handle = plt.plot(
            np.arange(len(vals)),
            vals,
            color=COLOR_GC,
            linewidth=LINEWIDTH,
            linestyle=line_style,
            label="GCD vs Undefended (mean)",
            zorder=3,
        )[0]
        legend_handles.append(mean_gc_handle)

    plt.xlabel("Query Iterations")
    plt.ylabel("Angular Deviation")

    if zoom_y and all_angles:
        y_min = float(np.min(all_angles))
        y_max = float(np.max(all_angles))
        if y_max > y_min:
            plt.ylim(y_min, y_max)
        else:
            half_span = max(1e-3, abs(y_min) * 0.01)
            plt.ylim(y_min - half_span, y_max + half_span)
    else:
        plt.ylim(0, 180)

    plt.grid(True, alpha=0.3)
    if legend_handles:
        plt.legend(handles=legend_handles, loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _finite_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, None
    return float(arr.mean()), float(arr.min()), float(arr.max())


def _build_interval_rows(iterations, angles_aaa, angles_gc, interval_size):
    max_len = max(len(angles_aaa), len(angles_gc))
    if max_len == 0:
        return []

    if iterations is None or len(iterations) < max_len:
        iterations = list(range(max_len))
    else:
        iterations = list(iterations[:max_len])

    rows = []
    for start in range(0, max_len, interval_size):
        end = min(start + interval_size - 1, max_len - 1)
        seg_aaa = angles_aaa[start : end + 1] if start < len(angles_aaa) else []
        seg_gc = angles_gc[start : end + 1] if start < len(angles_gc) else []

        aaa_mean, aaa_min, aaa_max = _finite_stats(seg_aaa)
        gc_mean, gc_min, gc_max = _finite_stats(seg_gc)
        diff = None
        if aaa_mean is not None and gc_mean is not None:
            diff = aaa_mean - gc_mean

        rows.append(
            {
                "iter_start": int(iterations[start]),
                "iter_end": int(iterations[end]),
                "n_aaa": int(np.isfinite(np.asarray(seg_aaa, dtype=np.float64)).sum()),
                "n_gc": int(np.isfinite(np.asarray(seg_gc, dtype=np.float64)).sum()),
                "aaa_mean": aaa_mean,
                "aaa_min": aaa_min,
                "aaa_max": aaa_max,
                "gc_mean": gc_mean,
                "gc_min": gc_min,
                "gc_max": gc_max,
                "aaa_minus_gc_mean": diff,
            }
        )

    return rows


def _fmt_float(value):
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def _save_interval_table_csv(rows, csv_path):
    fieldnames = [
        "iter_start",
        "iter_end",
        "n_aaa",
        "n_gc",
        "aaa_mean",
        "aaa_min",
        "aaa_max",
        "gc_mean",
        "gc_min",
        "gc_max",
        "aaa_minus_gc_mean",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in [
                "aaa_mean",
                "aaa_min",
                "aaa_max",
                "gc_mean",
                "gc_min",
                "gc_max",
                "aaa_minus_gc_mean",
            ]:
                out[key] = _fmt_float(out[key])
            writer.writerow(out)


def _save_interval_table_md(rows, md_path, interval_size):
    lines = []
    lines.append("# Angle Summary Table")
    lines.append("")
    lines.append("Interval size: {} iterations".format(interval_size))
    lines.append("")
    lines.append(
        "| iter_start | iter_end | n_aaa | n_gc | aaa_mean | aaa_min | aaa_max | gc_mean | gc_min | gc_max | aaa_minus_gc_mean |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in rows:
        lines.append(
            "| {iter_start} | {iter_end} | {n_aaa} | {n_gc} | {aaa_mean} | {aaa_min} | {aaa_max} | {gc_mean} | {gc_min} | {gc_max} | {aaa_minus_gc_mean} |".format(
                iter_start=row["iter_start"],
                iter_end=row["iter_end"],
                n_aaa=row["n_aaa"],
                n_gc=row["n_gc"],
                aaa_mean=_fmt_float(row["aaa_mean"]),
                aaa_min=_fmt_float(row["aaa_min"]),
                aaa_max=_fmt_float(row["aaa_max"]),
                gc_mean=_fmt_float(row["gc_mean"]),
                gc_min=_fmt_float(row["gc_min"]),
                gc_max=_fmt_float(row["gc_max"]),
                aaa_minus_gc_mean=_fmt_float(row["aaa_minus_gc_mean"]),
            )
        )
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_table(rows, interval_size):
    print("Angle summary by {}-iteration intervals:".format(interval_size))
    header = (
        "iter_start iter_end n_aaa n_gc aaa_mean aaa_min aaa_max "
        "gc_mean gc_min gc_max aaa_minus_gc_mean"
    )
    print(header)
    for row in rows:
        print(
            "{:>10d} {:>8d} {:>5d} {:>4d} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>17}".format(
                row["iter_start"],
                row["iter_end"],
                row["n_aaa"],
                row["n_gc"],
                _fmt_float(row["aaa_mean"]),
                _fmt_float(row["aaa_min"]),
                _fmt_float(row["aaa_max"]),
                _fmt_float(row["gc_mean"]),
                _fmt_float(row["gc_min"]),
                _fmt_float(row["gc_max"]),
                _fmt_float(row["aaa_minus_gc_mean"]),
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load loss_gradient_angle_compare.json, regenerate plots, and export "
            "a tabular angle summary without re-running defenses."
        )
    )
    parser.add_argument(
        "--input-json",
        default="final_angles/loss_gradient_angle_compare.json",
        help="Path to loss_gradient_angle_compare.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs. Default: same directory as input JSON.",
    )
    parser.add_argument(
        "--interval-size",
        type=int,
        default=50,
        help="Iteration interval size for table aggregation (default: 50).",
    )
    parser.add_argument(
        "--line-name",
        default="loss_gradient_angle_compare_line_from_json.pdf",
        help="Output filename for line plot.",
    )
    parser.add_argument(
        "--dotted-name",
        default="loss_gradient_angle_compare_dotted_from_json.pdf",
        help="Output filename for dotted plot.",
    )
    parser.add_argument(
        "--table-csv-name",
        default="loss_gradient_angle_interval_table_50.csv",
        help="Output filename for interval table CSV.",
    )
    parser.add_argument(
        "--table-md-name",
        default="loss_gradient_angle_interval_table_50.md",
        help="Output filename for interval table Markdown.",
    )
    parser.add_argument(
        "--no-zoom-y",
        action="store_true",
        help="Disable dynamic y-axis zoom (use fixed 0..180).",
    )
    parser.add_argument(
        "--hide-sample-traces",
        action="store_true",
        help="Hide pale per-sample dashed traces and show only mean lines.",
    )
    args = parser.parse_args()

    if args.interval_size <= 0:
        raise ValueError("interval-size must be positive.")

    input_json = os.path.abspath(args.input_json)
    if not os.path.isfile(input_json):
        raise FileNotFoundError("Input JSON not found: {}".format(input_json))

    with open(input_json) as f:
        data = json.load(f)

    mean_aaa, mean_gc, sample_traces_aaa, sample_traces_gc = _extract_traces(data)
    if args.hide_sample_traces:
        sample_traces_aaa = []
        sample_traces_gc = []

    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir is not None
        else os.path.dirname(input_json)
    )
    os.makedirs(output_dir, exist_ok=True)

    line_path = os.path.join(output_dir, args.line_name)
    dotted_path = os.path.join(output_dir, args.dotted_name)
    table_csv_path = os.path.join(output_dir, args.table_csv_name)
    table_md_path = os.path.join(output_dir, args.table_md_name)

    _plot_angle_curves(
        mean_aaa,
        mean_gc,
        sample_traces_aaa,
        sample_traces_gc,
        line_path,
        line_style="-",
        zoom_y=not args.no_zoom_y,
    )
    _plot_angle_curves(
        mean_aaa,
        mean_gc,
        sample_traces_aaa,
        sample_traces_gc,
        dotted_path,
        line_style=":",
        zoom_y=not args.no_zoom_y,
    )

    iterations = data.get("iterations")
    rows = _build_interval_rows(iterations, mean_aaa, mean_gc, args.interval_size)
    _save_interval_table_csv(rows, table_csv_path)
    _save_interval_table_md(rows, table_md_path, args.interval_size)
    _print_table(rows, args.interval_size)

    print("Saved line plot to:", line_path)
    print("Saved dotted plot to:", dotted_path)
    print("Saved interval CSV table to:", table_csv_path)
    print("Saved interval Markdown table to:", table_md_path)


if __name__ == "__main__":
    main()
