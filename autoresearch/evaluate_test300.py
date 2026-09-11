from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autoresearch.common import RUNS_DIR, default_run_name, save_json
    from autoresearch.dataset import load_records
    from autoresearch.evolution import optimize_prefix_for_aas
    from autoresearch.features import compute_reference_stats, evolutionary_metric_matrix
    from autoresearch.model import load_model_from_checkpoint
    from autoresearch.prompt_library import PromptRecord, build_library_splits
    from autoresearch.vocab import SRC_PAD_ID, decode_cds, split_codons
else:
    from .common import RUNS_DIR, default_run_name, save_json
    from .dataset import load_records
    from .evolution import optimize_prefix_for_aas
    from .features import compute_reference_stats, evolutionary_metric_matrix
    from .model import load_model_from_checkpoint
    from .prompt_library import PromptRecord, build_library_splits
    from .vocab import SRC_PAD_ID, decode_cds, split_codons


DEFAULT_PROMPT_FASTA = Path(__file__).resolve().parent / "assets" / "prompt_library" / "uniref10_sampled.fasta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate autoresearch test300 predictions and metric summaries.")
    parser.add_argument("--species", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--method", choices=["seq2seq", "evolution_hybrid"], required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default=str(RUNS_DIR))
    parser.add_argument("--prompt-fasta", default=str(DEFAULT_PROMPT_FASTA))
    parser.add_argument("--library-train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-aa-len", type=int, default=512)
    parser.add_argument("--samples-per-aas", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--sample", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=4042)
    parser.add_argument("--evo-population-size", type=int, default=32)
    parser.add_argument("--evo-generations", type=int, default=20)
    parser.add_argument("--evo-elite-fraction", type=float, default=0.25)
    parser.add_argument("--evo-mutation-rate", type=float, default=0.12)
    parser.add_argument("--evo-mutations-per-child", type=int, default=2)
    parser.add_argument("--head-codons", type=int, default=10)
    return parser.parse_args()


def get_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_target_dtype(dtype_flag: str) -> torch.dtype | None:
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "bf16":
        return torch.bfloat16
    return None


def pad_source_batch(records: list[PromptRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(record.source_ids) for record in records)
    rows = [list(record.source_ids) + [SRC_PAD_ID] * (max_len - len(record.source_ids)) for record in records]
    src_ids = torch.tensor(rows, dtype=torch.long)
    return src_ids, src_ids.eq(SRC_PAD_ID)


def finalize_cds(seq: str, preferred_stop: str) -> str:
    cds = str(seq).upper().replace("U", "T")
    if not cds.endswith(("TAA", "TAG", "TGA")):
        cds += preferred_stop
    return cds


def build_test_prompts(args: argparse.Namespace) -> list[PromptRecord]:
    _, _, test_records = build_library_splits(
        Path(args.prompt_fasta).expanduser().resolve(),
        max_aa_len=args.max_aa_len,
        train_size=args.library_train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    return test_records


def load_reference(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    dataset_meta, records = load_records(
        species=args.species,
        max_aa_len=args.max_aa_len,
        require_abundance=False,
    )
    reference = compute_reference_stats(
        [record.cds for record in records],
        top_pair_k=64,
        prefix_window=args.head_codons,
    )
    stop_codons = [split_codons(record.cds)[-1] for record in records if split_codons(record.cds)]
    preferred_stop = pd.Series(stop_codons).value_counts().idxmax() if stop_codons else "TAA"
    return dataset_meta, reference, preferred_stop


def generate_seq2seq_rows(args: argparse.Namespace, prompts: list[PromptRecord], preferred_stop: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = get_device(args.device)
    target_dtype = get_target_dtype(args.dtype)
    model, _, payload = load_model_from_checkpoint(Path(args.checkpoint_path).expanduser().resolve(), map_location="cpu")
    if target_dtype is not None:
        model = model.to(dtype=target_dtype)
    model = model.to(device)
    model.eval()
    autocast_enabled = device.type == "cuda" and target_dtype is not None
    autocast_dtype = target_dtype if target_dtype is not None else torch.float16

    rows: list[dict[str, Any]] = []
    try:
        for chunk_start in range(0, len(prompts), args.batch_size):
            chunk = prompts[chunk_start : chunk_start + args.batch_size]
            src_ids, src_padding_mask = pad_source_batch(chunk)
            src_ids = src_ids.to(device)
            src_padding_mask = src_padding_mask.to(device)

            for sample_idx in range(args.samples_per_aas):
                local_seed = int(args.sample_seed + chunk_start * args.samples_per_aas + sample_idx)
                torch.manual_seed(local_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(local_seed)
                with torch.no_grad(), torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    pred_ids = model.generate(
                        src_ids,
                        src_padding_mask=src_padding_mask,
                        sample=args.sample,
                        temperature=args.temperature,
                        top_k=args.top_k,
                    )
                decoded = [decode_cds(item.tolist()) for item in pred_ids.cpu()]
                for prompt, cds in zip(chunk, decoded):
                    rows.append(
                        {
                            "record_id": prompt.record_id,
                            "gene_name": prompt.gene_name,
                            "aas": prompt.aas,
                            "sample_idx": int(sample_idx),
                            "predicted_cds": finalize_cds(cds, preferred_stop),
                        }
                    )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows, {
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "samples_per_aas": int(args.samples_per_aas),
        "sample": bool(args.sample),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "sample_seed": int(args.sample_seed),
        "trained_max_aa_len": int(
            payload.get("dataset_meta", {}).get("max_aa_len")
            or payload.get("model_config", {}).get("max_seq_len")
            or 0
        ),
    }


def generate_evolution_hybrid_rows(
    args: argparse.Namespace,
    prompts: list[PromptRecord],
    reference,
    preferred_stop: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seq_rows, seq_info = generate_seq2seq_rows(args, prompts, preferred_stop)
    rng = np.random.default_rng(args.seed)
    prefix_map = {
        prompt.record_id: optimize_prefix_for_aas(
            prompt.aas,
            reference=reference,
            population_size=args.evo_population_size,
            generations=args.evo_generations,
            elite_fraction=args.evo_elite_fraction,
            mutation_rate=args.evo_mutation_rate,
            mutations_per_child=args.evo_mutations_per_child,
            rng=rng,
        )
        for prompt in prompts
    }

    rows: list[dict[str, Any]] = []
    for row in seq_rows:
        prefix_payload = prefix_map[str(row["record_id"])]
        prefix_codons = split_codons(str(prefix_payload["best_prefix_cds"]))
        full_codons = split_codons(str(row["predicted_cds"]))
        tail_codons = full_codons[len(prefix_codons) :]
        hybrid_cds = "".join(prefix_codons + tail_codons)
        rows.append(
            {
                **row,
                "predicted_cds": finalize_cds(hybrid_cds, preferred_stop),
                "evo_prefix_cds": str(prefix_payload["best_prefix_cds"]),
                "prefix_cai_top10": float(prefix_payload["metrics"].get("prefix_cai_top10", 0.0)),
                "prefix_cpai_top10": float(prefix_payload["metrics"].get("prefix_cpai_top10", 0.0)),
                "prefix_mfe_per_nt": float(prefix_payload["metrics"].get("prefix_mfe_per_nt", 0.0)),
            }
        )

    return rows, {
        **seq_info,
        "head_codons": int(args.head_codons),
        "evo_population_size": int(args.evo_population_size),
        "evo_generations": int(args.evo_generations),
        "evo_elite_fraction": float(args.evo_elite_fraction),
        "evo_mutation_rate": float(args.evo_mutation_rate),
        "evo_mutations_per_child": int(args.evo_mutations_per_child),
    }


def summarize_head_metrics(rows: list[dict[str, Any]], reference) -> dict[str, float]:
    cds_list = [str(row["predicted_cds"]) for row in rows]
    metric_matrix, metric_names = evolutionary_metric_matrix(cds_list, reference)
    summary = {
        f"{name}_mean": float(metric_matrix[:, idx].mean())
        for idx, name in enumerate(metric_names)
    }
    prompt_metric_values: dict[str, list[np.ndarray]] = {}
    for row, metric_row in zip(rows, metric_matrix):
        prompt_metric_values.setdefault(str(row["record_id"]), []).append(metric_row)
    best_metric_rows = []
    for record_id, metric_rows in prompt_metric_values.items():
        metric_stack = np.stack(metric_rows, axis=0)
        best_metric_rows.append(metric_stack.max(axis=0))
    best_metric_matrix = np.stack(best_metric_rows, axis=0)
    for idx, name in enumerate(metric_names):
        summary[f"{name}_bestof5_mean"] = float(best_metric_matrix[:, idx].mean())
    return summary


def main() -> None:
    args = parse_args()
    run_name = args.run_name or default_run_name(f"{args.species}_{args.method}")
    output_dir = Path(args.output_root).expanduser().resolve() / args.species / run_name / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = build_test_prompts(args)
    dataset_meta, reference, preferred_stop = load_reference(args)
    if args.method == "seq2seq":
        rows, method_info = generate_seq2seq_rows(args, prompts, preferred_stop)
    else:
        rows, method_info = generate_evolution_hybrid_rows(args, prompts, reference, preferred_stop)

    metrics_summary = summarize_head_metrics(rows, reference)
    predictions_csv = output_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_csv, index=False)
    summary = {
        "species": args.species,
        "method": args.method,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "predictions_csv": str(predictions_csv),
        "dataset_meta": dataset_meta,
        "method_info": method_info,
        "head_metric_summary": metrics_summary,
        "head_reference": {
            "prefix_window": int(reference.prefix_window),
            "prefix_nt_window": int(reference.prefix_window) * 3,
        },
        "num_predictions": int(len(rows)),
        "num_prompts": int(len(prompts)),
    }
    save_json(output_dir / "summary.json", summary)
    print(f"output_dir: {summary['output_dir']}")
    print(f"predictions_csv: {summary['predictions_csv']}")


if __name__ == "__main__":
    main()
