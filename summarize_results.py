import argparse
import json
from pathlib import Path


def load_results(root: Path):
    rows = []
    for result_file in sorted(root.rglob("results.json")):
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "strategy": data["strategy"],
                "num_params_m": data["num_params"] / 1e6,
                "best_val_ppl": data["best_val_ppl"],
                "best_val_loss": data["best_val_loss"],
                "path": str(result_file),
            }
        )
    return rows


def make_markdown_table(rows):
    lines = [
        "| Model | Params (M) | Best Val Loss | Best Val PPL | Result Path |",
        "|---|---:|---:|---:|---|",
    ]
    name_map = {
        "attention": "TransformerLM",
        "grassmann": "GrassmannLM",
    }
    for row in sorted(rows, key=lambda x: x["strategy"]):
        model_name = name_map.get(row["strategy"], row["strategy"])
        lines.append(
            f"| {model_name} | {row['num_params_m']:.2f} | "
            f"{row['best_val_loss']:.4f} | {row['best_val_ppl']:.2f} | {row['path']} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize Hydra experiment results")
    parser.add_argument("--root", type=str, required=True, help="Root dir containing results.json files")
    parser.add_argument("--out", type=str, default=None, help="Optional output markdown file")
    args = parser.parse_args()

    rows = load_results(Path(args.root))
    if not rows:
        raise SystemExit(f"No results.json found under: {args.root}")

    table = make_markdown_table(rows)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        print(f"\nSaved table: {out}")


if __name__ == "__main__":
    main()
