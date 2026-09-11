from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autoresearch.dataset import load_records
    from autoresearch.features import compute_reference_stats, evolutionary_metric_matrix
else:
    from .dataset import load_records
    from .features import compute_reference_stats, evolutionary_metric_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize head-only evolutionary metrics for generated CDS.")
    parser.add_argument("--species", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--max-aa-len", type=int, default=512)
    parser.add_argument("--head-codons", type=int, default=10)
    return parser.parse_args()


def summarize_metrics(df: pd.DataFrame, metric_matrix: np.ndarray, metric_names: list[str]) -> dict[str, Any]:
    summary = {
        f"{name}_mean": float(metric_matrix[:, idx].mean())
        for idx, name in enumerate(metric_names)
    }
    grouped_best: list[np.ndarray] = []
    for _, sub_df in df.assign(_metric_row=list(metric_matrix)).groupby("record_id", sort=False):
        best_row = np.stack(sub_df["_metric_row"].to_list(), axis=0).max(axis=0)
        grouped_best.append(best_row)
    best_matrix = np.stack(grouped_best, axis=0)
    for idx, name in enumerate(metric_names):
        summary[f"{name}_bestof5_mean"] = float(best_matrix[:, idx].mean())
    return summary


def main() -> None:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv).expanduser().resolve()
    df = pd.read_csv(predictions_csv)
    required = {"record_id", "predicted_cds"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Predictions CSV missing required columns: {missing}")

    _, records = load_records(
        species=args.species,
        max_aa_len=args.max_aa_len,
        require_abundance=False,
    )
    reference = compute_reference_stats(
        [record.cds for record in records],
        top_pair_k=64,
        prefix_window=args.head_codons,
    )
    cds_list = [str(cds) for cds in df["predicted_cds"].tolist()]
    metric_matrix, metric_names = evolutionary_metric_matrix(cds_list, reference)
    payload = {
        "species": args.species,
        "predictions_csv": str(predictions_csv),
        "num_sequences": int(len(df)),
        "num_prompts": int(df["record_id"].astype(str).nunique()),
        "prefix_window": int(reference.prefix_window),
        "metrics": summarize_metrics(df, metric_matrix, metric_names),
    }

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
