from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autoresearch.common import save_json
    from autoresearch.dataset import load_records
    from autoresearch.evolution import optimize_prefix_for_aas
    from autoresearch.features import compute_reference_stats
    from autoresearch.model import load_model_from_checkpoint
    from autoresearch.vocab import SRC_PAD_ID, decode_cds, encode_source, normalize_aas, split_codons
else:
    from .common import save_json
    from .dataset import load_records
    from .evolution import optimize_prefix_for_aas
    from .features import compute_reference_stats
    from .model import load_model_from_checkpoint
    from .vocab import SRC_PAD_ID, decode_cds, encode_source, normalize_aas, split_codons


@dataclass
class InferenceSpec:
    method: str
    dataset: str
    checkpoint_path: str
    species: str | None = None
    run_name: str = ""
    head_codons: int = 10
    batch_size: int = 32
    sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    sample_seed: int = 4042
    device: str = "cpu"
    dtype: str = "fp32"
    max_aa_len: int = 512
    evo_population_size: int = 32
    evo_generations: int = 20
    evo_elite_fraction: float = 0.25
    evo_mutation_rate: float = 0.12
    evo_mutations_per_child: int = 2

    @classmethod
    def from_path(cls, path: str | Path) -> "InferenceSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified aas2cds inference interface for RL and EA-hybrid autoresearch outputs.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--aas", default="")
    parser.add_argument("--aas-file", default="")
    parser.add_argument("--output-csv", default="predictions.csv")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--sample-seed", type=int, default=-1)
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


def normalize_input_aas(seq: str) -> str:
    aas = normalize_aas(seq)
    return aas if aas.endswith("*") else f"{aas}*"


def load_aas_inputs(*, aas: str, aas_file: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if aas.strip():
        rows.append({"record_id": "input_00001", "gene_name": "input_00001", "aas": normalize_input_aas(aas)})
        return rows
    if not aas_file:
        raise ValueError("Either --aas or --aas-file is required")
    path = Path(aas_file).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith(">"):
        current_id = None
        current_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if current_id is not None:
                    rows.append(
                        {
                            "record_id": current_id,
                            "gene_name": current_id,
                            "aas": normalize_input_aas("".join(current_lines)),
                        }
                    )
                current_id = stripped[1:].split()[0] or f"input_{len(rows) + 1:05d}"
                current_lines = []
            else:
                current_lines.append(stripped)
        if current_id is not None:
            rows.append(
                {
                    "record_id": current_id,
                    "gene_name": current_id,
                    "aas": normalize_input_aas("".join(current_lines)),
                }
            )
        if rows:
            return rows
    for line_idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(
            {
                "record_id": f"input_{line_idx:05d}",
                "gene_name": f"input_{line_idx:05d}",
                "aas": normalize_input_aas(stripped),
            }
        )
    if not rows:
        raise ValueError(f"No amino-acid sequences found in {path}")
    return rows


def pad_source_batch(rows: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_source(row["aas"]) for row in rows]
    max_len = max(len(item) for item in encoded)
    src_rows = [item + [SRC_PAD_ID] * (max_len - len(item)) for item in encoded]
    src_ids = torch.tensor(src_rows, dtype=torch.long)
    return src_ids, src_ids.eq(SRC_PAD_ID)


def finalize_cds(seq: str, preferred_stop: str) -> str:
    cds = str(seq).upper().replace("U", "T")
    if not cds.endswith(("TAA", "TAG", "TGA")):
        cds += preferred_stop
    return cds


def load_reference(spec: InferenceSpec):
    _, records = load_records(
        species=spec.species,
        dataset=spec.dataset,
        max_aa_len=spec.max_aa_len,
        require_abundance=False,
    )
    reference = compute_reference_stats(
        [record.cds for record in records],
        top_pair_k=64,
        prefix_window=spec.head_codons,
    )
    stop_codons = [split_codons(record.cds)[-1] for record in records if split_codons(record.cds)]
    preferred_stop = "TAA"
    if stop_codons:
        counts: dict[str, int] = {}
        for codon in stop_codons:
            counts[codon] = counts.get(codon, 0) + 1
        preferred_stop = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return reference, preferred_stop


def generate_seq2seq_rows(
    rows: list[dict[str, str]],
    *,
    spec: InferenceSpec,
    preferred_stop: str,
    batch_size: int,
    device_name: str,
    dtype_name: str,
    sample: bool,
    temperature: float,
    top_k: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    device = get_device(device_name)
    target_dtype = get_target_dtype(dtype_name)
    model, _, _ = load_model_from_checkpoint(Path(spec.checkpoint_path).expanduser().resolve(), map_location="cpu")
    if target_dtype is not None:
        model = model.to(dtype=target_dtype)
    model = model.to(device)
    model.eval()
    autocast_enabled = device.type == "cuda" and target_dtype is not None
    autocast_dtype = target_dtype if target_dtype is not None else torch.float16

    output_rows: list[dict[str, Any]] = []
    try:
        for chunk_start in range(0, len(rows), batch_size):
            chunk = rows[chunk_start : chunk_start + batch_size]
            src_ids, src_padding_mask = pad_source_batch(chunk)
            src_ids = src_ids.to(device)
            src_padding_mask = src_padding_mask.to(device)
            local_seed = int(sample_seed + chunk_start)
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
                    sample=sample,
                    temperature=temperature,
                    top_k=top_k,
                )
            decoded = [decode_cds(item.tolist()) for item in pred_ids.cpu()]
            for row, cds in zip(chunk, decoded):
                output_rows.append(
                    {
                        "record_id": row["record_id"],
                        "gene_name": row["gene_name"],
                        "aas": row["aas"],
                        "predicted_cds": finalize_cds(cds, preferred_stop),
                    }
                )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output_rows


def generate_evolution_hybrid_rows(
    rows: list[dict[str, str]],
    *,
    spec: InferenceSpec,
    reference,
    preferred_stop: str,
    batch_size: int,
    device_name: str,
    dtype_name: str,
    sample: bool,
    temperature: float,
    top_k: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    seq_rows = generate_seq2seq_rows(
        rows,
        spec=spec,
        preferred_stop=preferred_stop,
        batch_size=batch_size,
        device_name=device_name,
        dtype_name=dtype_name,
        sample=sample,
        temperature=temperature,
        top_k=top_k,
        sample_seed=sample_seed,
    )
    rng = torch.Generator(device="cpu")
    _ = rng
    np_rng = __import__("numpy").random.default_rng(sample_seed)
    prefix_map = {
        row["record_id"]: optimize_prefix_for_aas(
            row["aas"],
            reference=reference,
            population_size=spec.evo_population_size,
            generations=spec.evo_generations,
            elite_fraction=spec.evo_elite_fraction,
            mutation_rate=spec.evo_mutation_rate,
            mutations_per_child=spec.evo_mutations_per_child,
            rng=np_rng,
        )
        for row in rows
    }

    output_rows: list[dict[str, Any]] = []
    for row in seq_rows:
        prefix_payload = prefix_map[str(row["record_id"])]
        prefix_codons = split_codons(str(prefix_payload["best_prefix_cds"]))
        full_codons = split_codons(str(row["predicted_cds"]))
        tail_codons = full_codons[len(prefix_codons) :]
        hybrid_cds = "".join(prefix_codons + tail_codons)
        output_rows.append(
            {
                **row,
                "predicted_cds": finalize_cds(hybrid_cds, preferred_stop),
                "evo_prefix_cds": str(prefix_payload["best_prefix_cds"]),
                "prefix_cai_top10": float(prefix_payload["metrics"].get("prefix_cai_top10", 0.0)),
                "prefix_cpai_top10": float(prefix_payload["metrics"].get("prefix_cpai_top10", 0.0)),
                "prefix_mfe_per_nt": float(prefix_payload["metrics"].get("prefix_mfe_per_nt", 0.0)),
            }
        )
    return output_rows


def main() -> None:
    args = parse_args()
    spec = InferenceSpec.from_path(args.spec)
    rows = load_aas_inputs(aas=args.aas, aas_file=args.aas_file)
    batch_size = int(args.batch_size or spec.batch_size)
    device_name = args.device or spec.device
    dtype_name = args.dtype or spec.dtype
    sample = bool(args.sample or spec.sample)
    temperature = float(spec.temperature if args.temperature < 0 else args.temperature)
    top_k = int(spec.top_k if args.top_k < 0 else args.top_k)
    sample_seed = int(spec.sample_seed if args.sample_seed < 0 else args.sample_seed)

    reference, preferred_stop = load_reference(spec)
    if spec.method == "seq2seq":
        output_rows = generate_seq2seq_rows(
            rows,
            spec=spec,
            preferred_stop=preferred_stop,
            batch_size=batch_size,
            device_name=device_name,
            dtype_name=dtype_name,
            sample=sample,
            temperature=temperature,
            top_k=top_k,
            sample_seed=sample_seed,
        )
    elif spec.method == "evolution_hybrid":
        output_rows = generate_evolution_hybrid_rows(
            rows,
            spec=spec,
            reference=reference,
            preferred_stop=preferred_stop,
            batch_size=batch_size,
            device_name=device_name,
            dtype_name=dtype_name,
            sample=sample,
            temperature=temperature,
            top_k=top_k,
            sample_seed=sample_seed,
        )
    else:
        raise ValueError(f"Unsupported inference method: {spec.method}")

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(output_rows).to_csv(output_csv, index=False)
    summary = {
        "spec": asdict(spec),
        "output_csv": str(output_csv),
        "num_predictions": int(len(output_rows)),
        "method": spec.method,
        "device": str(get_device(device_name)),
        "dtype": dtype_name,
        "sample": sample,
        "temperature": temperature,
        "top_k": top_k,
    }
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else output_csv.with_suffix(".json")
    save_json(output_json, summary)
    print(f"output_csv: {output_csv}")
    print(f"summary_json: {output_json}")


if __name__ == "__main__":
    main()
