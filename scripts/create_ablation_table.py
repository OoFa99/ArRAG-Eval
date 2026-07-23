"""
scripts/create_ablation_table.py

Pulls every run from the "ArRAG-Eval" MLFlow experiment and renders a
markdown comparison table (context precision / faithfulness / answer
relevance) — the ablation table format from the roadmap's "what to share
on LinkedIn" example.

If a config was run more than once (e.g. you reran an ablation after a
fix), only the most recent run for that experiment_name is kept, so the
table doesn't show stale duplicate rows from earlier, possibly-broken
runs.

Run:
    python scripts/create_ablation_table.py
Writes:
    ablation_table.md
"""

import mlflow

MLFLOW_EXPERIMENT_NAME = "ArRAG-Eval-Full"
OUTPUT_PATH = "ablation_table.md"

METRIC_COLUMNS = [
    ("context_precision", "Context Precision"),
    ("faithfulness", "Faithfulness"),
    ("answer_relevance", "Answer Relevance"),
]

# Ablation configs read naturally in this order (dense → hybrid → agentic
# tells the story of what each addition buys you); anything else found in
# the experiment gets appended alphabetically after these.
PREFERRED_ORDER = ["ablation-dense", "ablation-hybrid", "ablation-agentic"]


def get_latest_run_per_config(runs):
    """Keep only the most recent MLFlow run per experiment_name param."""
    latest = {}
    for run in runs:
        name = run.data.params.get("experiment_name", run.info.run_name or run.info.run_id)
        existing = latest.get(name)
        if existing is None or run.info.start_time > existing.info.start_time:
            latest[name] = run
    return latest


def format_row(name, run):
    cells = [name]
    for metric_key, _ in METRIC_COLUMNS:
        value = run.data.metrics.get(metric_key)
        cells.append(f"{value:.3f}" if value is not None else "—")
    return "| " + " | ".join(cells) + " |"


def main():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        print(
            f"No MLFlow experiment named '{MLFLOW_EXPERIMENT_NAME}' found. "
            f"Run experiments/run_baseline_with_mlflow.py or run_ablation.py first."
        )
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if not runs:
        print(f"No runs found in experiment '{MLFLOW_EXPERIMENT_NAME}'.")
        return

    latest_by_config = get_latest_run_per_config(runs)

    ordered_names = [n for n in PREFERRED_ORDER if n in latest_by_config]
    ordered_names += sorted(n for n in latest_by_config if n not in PREFERRED_ORDER)

    header = "| Configuration | " + " | ".join(label for _, label in METRIC_COLUMNS) + " |"
    separator = "|---" * (len(METRIC_COLUMNS) + 1) + "|"

    lines = [header, separator]
    for name in ordered_names:
        lines.append(format_row(name, latest_by_config[name]))

    table_md = "\n".join(lines)
    print(table_md)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# ArRAG-Eval Ablation Results\n\n{table_md}\n")

    print(f"\n✓ Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()