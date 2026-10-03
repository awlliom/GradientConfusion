from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


def create_polar_histogram(
    angles_dict,
    ax,
    title,
    show_dashed_lines=True,
    dashed_stat="mean",
):
    """
    Create a polar histogram similar to the analysis scripts.
    angles_dict: {sample_idx: [angles]}
    """
    all_angles = []
    for _, angles in angles_dict.items():
        all_angles.extend(angles)

    all_angles = np.array(all_angles)

    n_bins = 36
    bins = np.linspace(0, 180, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_centers_rad = np.deg2rad(bin_centers)
    bin_width = np.deg2rad(bins[1] - bins[0])

    # Draw samples with more queries first (back), fewer queries last (front)
    sample_items = sorted(
        angles_dict.items(),
        key=lambda x: (len(x[1]), x[0]),
        reverse=True,
    )
    num_series = len(sample_items)
    cmap_name = "tab10" if num_series <= 10 else "tab20"
    cmap = cm.get_cmap(cmap_name, num_series)
    max_count = 0
    stat_lines = []
    legend_entries = []

    fill_alpha = 0.5

    for idx, (sample_idx, angles) in enumerate(sample_items):
        angles = np.array(angles)
        hist, _ = np.histogram(angles, bins=bins)
        max_count = max(max_count, hist.max() if hist.size else 0)
        color = cmap(idx)
        ax.bar(
            bin_centers_rad,
            hist,
            width=bin_width,
            bottom=0.0,
            color=color,
            edgecolor=color,
            linewidth=0.6,
            alpha=fill_alpha,
            label=str(sample_idx),
            zorder=2,
        )
        legend_entries.append((str(sample_idx), color))
        if angles.size:
            if dashed_stat == "median":
                stat_angle = float(np.median(angles))
            else:
                stat_angle = float(np.mean(angles))
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
    ax.tick_params(labelsize=10)

    ax.grid(True, linewidth=0.5, color="gray", alpha=0.3, zorder=1)

    if show_dashed_lines and max_count > 0 and stat_lines:
        line_end = r_data_max * 1.0
        label_base = r_data_max * 1.2
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
                fontsize=17,
                ha="center",
                va="bottom",
                zorder=4,
                clip_on=False,
            )

    # if title:
    #     ax.text(
    #         0.5,
    #         0.1,
    #         title,
    #         transform=ax.transAxes,
    #         ha="center",
    #         va="top",
    #         fontsize=14,
    #     )

    return legend_entries, all_angles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot angle-analysis polar histograms from a saved JSON."
    )
    parser.add_argument(
        "--input-json",
        action="append",
        required=True,
        help="Path to angles_data.json from a previous run (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save PDFs (default: alongside JSON).",
    )
    parser.add_argument(
        "--show-dashed-lines",
        action="store_true",
        help="Show dashed statistic lines on the polar histogram.",
    )
    parser.add_argument(
        "--hide-dashed-lines",
        action="store_true",
        help="Hide dashed statistic lines on the polar histogram.",
    )
    parser.add_argument(
        "--dashed-stat",
        choices=["mean", "median"],
        default="mean",
        help="Statistic for dashed lines (default: mean).",
    )
    parser.add_argument(
        "--title-prefix",
        default="Histogram of Angular Deviations",
        help="Prefix for plot titles.",
    )
    parser.add_argument(
        "--output-name",
        default="histogram_Bopt_ATv_combined.pdf",
        help="Output PDF filename (default: histogram_Bopt_ATv_combined.pdf).",
    )
    parser.add_argument(
        "--separate-legend",
        action="store_true",
        help="Also save a no-legend plot and a standalone legend image.",
    )
    parser.add_argument(
        "--no-legend-name",
        default="histogram_Bopt_ATv_nolegend.pdf",
        help="Filename for the no-legend plot (default: histogram_Bopt_ATv_nolegend.pdf).",
    )
    parser.add_argument(
        "--legend-name",
        default="legend_samples.pdf",
        help="Filename for the standalone legend image (default: legend_samples.pdf).",
    )
    parser.add_argument(
        "--panel-titles",
        nargs="*",
        default=None,
        help="Optional titles for each panel, in order.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input_json]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input JSON not found: {path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    show_dashed_lines = args.show_dashed_lines or not args.hide_dashed_lines

    num_panels = len(input_paths)
    fig, axes = plt.subplots(
        1,
        num_panels,
        figsize=(6 * num_panels, 6),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )
    axes = axes[0]

    legend_entries = None
    stats = []

    if args.panel_titles:
        panel_titles = args.panel_titles
    else:
        panel_titles = ["a) GCD", "b) AAA", "c) RLS"]

    for idx, input_path in enumerate(input_paths):
        with input_path.open("r") as f:
            data = json.load(f)

        if "angles_Bopt_ATv" not in data or not data["angles_Bopt_ATv"]:
            print(f"No angles_Bopt_ATv found in {input_path}")
            continue

        angles_dict = {int(k): v for k, v in data["angles_Bopt_ATv"].items()}
        title = panel_titles[idx] if idx < len(panel_titles) else ""

        entries, all_angles = create_polar_histogram(
            angles_dict,
            axes[idx],
            title,
            show_dashed_lines=show_dashed_lines,
            dashed_stat=args.dashed_stat,
        )
        stats.append((input_path, all_angles))
        if legend_entries is None:
            legend_entries = entries

    if legend_entries:
        legend_entries = sorted(legend_entries, key=lambda x: int(x[0]))
        handles = [
            Patch(facecolor=color, edgecolor=color, alpha=1.0)
            for label, color in legend_entries
        ]
        labels = [label for label, _ in legend_entries]
        if args.separate_legend:
            no_legend_path = output_dir / args.no_legend_name
            no_leg_fig, no_leg_axes = plt.subplots(
                1,
                num_panels,
                figsize=(6 * num_panels, 4.2),
                subplot_kw={"projection": "polar"},
                squeeze=False,
            )
            no_leg_axes = no_leg_axes[0]
            for i, input_path in enumerate(input_paths):
                with input_path.open("r") as f:
                    data = json.load(f)
                if "angles_Bopt_ATv" not in data or not data["angles_Bopt_ATv"]:
                    continue
                angles_dict = {int(k): v for k, v in data["angles_Bopt_ATv"].items()}
                title = panel_titles[i] if i < len(panel_titles) else ""
                create_polar_histogram(
                    angles_dict,
                    no_leg_axes[i],
                    title,
                    show_dashed_lines=show_dashed_lines,
                    dashed_stat=args.dashed_stat,
                )
            no_leg_fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.02, wspace=0.25)
            no_leg_fig.savefig(
                no_legend_path,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.0,
                dpi=600,
            )
            plt.close(no_leg_fig)
            print(f"Saved no-legend histogram to {no_legend_path}")

            legend_fig = plt.figure(figsize=(6, 1.2))
            legend_fig.legend(
                handles,
                labels,
                title="Sample",
                loc="center",
                ncol=len(labels),
                fontsize=13,
                title_fontsize=13,
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                edgecolor="#B0B7C3",
                facecolor="white",
                borderpad=0.6,
                handletextpad=0.5,
                labelspacing=0.4,
                handlelength=0.9,
                handleheight=0.6,
            )
            legend_path = output_dir / args.legend_name
            legend_fig.savefig(legend_path, format="pdf", bbox_inches="tight", dpi=600)
            plt.close(legend_fig)
            print(f"Saved legend-only image to {legend_path}")

        fig.legend(
            handles,
            labels,
            title="Sample",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.15),
            ncol=len(labels),
            fontsize=13,
            title_fontsize=13,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="#B0B7C3",
            facecolor="white",
            borderpad=0.6,
            handletextpad=0.5,
            labelspacing=0.4,
            handlelength=0.9,
            handleheight=0.6,
        )

    fig.tight_layout(rect=[0, 0.1, 1, 1], pad=0.8)
    out_path = output_dir / args.output_name
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=600)
    plt.close(fig)

    print(f"Saved combined histogram to {out_path}")
    for path, all_angles in stats:
        print(
            f"{path}: Mean {np.mean(all_angles):.2f}°, "
            f"Median {np.median(all_angles):.2f}°, Std {np.std(all_angles):.2f}°"
        )


if __name__ == "__main__":
    main()
