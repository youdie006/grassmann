import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NAME_MAP = {
    "attention": "TransformerLM",
    "grassmann": "GrassmannLM",
}


def load_results(result_paths):
    rows = []
    for path in result_paths:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "strategy": data["strategy"],
                "name": NAME_MAP.get(data["strategy"], data["strategy"]),
                "num_params": data["num_params"],
                "best_val_loss": data["best_val_loss"],
                "best_val_ppl": data["best_val_ppl"],
                "train_losses": data["train_losses"],
                "val_losses": data["val_losses"],
            }
        )
    return rows


def plot_loss_curves(rows, out_path: Path):
    plt.figure(figsize=(8, 5))
    for row in rows:
        epochs = np.arange(1, len(row["train_losses"]) + 1)
        plt.plot(
            epochs,
            row["val_losses"],
            marker="o",
            linewidth=2.0,
            label=f'{row["name"]} (val)',
        )
    plt.title("Validation Loss by Epoch (Hydra Run 2026-02-23 02-26-47)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.xticks(epochs)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_best_metrics(rows, out_path: Path):
    names = [row["name"] for row in rows]
    best_losses = [row["best_val_loss"] for row in rows]
    best_ppls = [row["best_val_ppl"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    bars1 = axes[0].bar(names, best_losses)
    axes[0].set_title("Best Validation Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars1, best_losses):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars2 = axes[1].bar(names, best_ppls)
    axes[1].set_title("Best Validation Perplexity (log scale)")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars2, best_ppls):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Best Metric Comparison (Hydra Run 2026-02-23 02-26-47)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from Hydra results.json files")
    parser.add_argument("--results", nargs="+", required=True, help="Paths to results.json files")
    parser.add_argument(
        "--out-dir",
        default="training_results_hydra/figures",
        help="Directory to save generated figures",
    )
    args = parser.parse_args()

    rows = load_results(args.results)
    rows = sorted(rows, key=lambda x: x["strategy"])
    out_dir = Path(args.out_dir)

    loss_path = out_dir / "run_2026-02-23_02-26-47_val_loss_curve.png"
    metric_path = out_dir / "run_2026-02-23_02-26-47_best_metrics.png"

    plot_loss_curves(rows, loss_path)
    plot_best_metrics(rows, metric_path)

    print(f"saved: {loss_path}")
    print(f"saved: {metric_path}")


if __name__ == "__main__":
    main()
