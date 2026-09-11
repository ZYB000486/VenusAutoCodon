"""Generate synonymous CDSs with a released GRPO-trained model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoresearch.infer import InferenceSpec, generate_seq2seq_rows
from autoresearch.vocab import normalize_aas, translate_cds
from .io import model_directory, read_fasta, write_csv


def generate(records: list[tuple[str, str]], model_dir: str | Path, *, samples: int = 5,
             device: str = "cpu", batch_size: int = 32, temperature: float = 1.0,
             top_k: int = 0, seed: int = 42, greedy: bool = False) -> list[dict]:
    folder = Path(model_dir)
    metadata = json.loads((folder / "metadata.json").read_text())
    max_length = int(metadata["generator"]["model_config"]["max_seq_len"])
    if samples < 1 or batch_size < 1 or temperature <= 0 or top_k < 0:
        raise ValueError("samples/batch-size/temperature must be positive; top-k must be non-negative")
    if greedy and samples != 1:
        raise ValueError("Use --samples 1 with --greedy")
    rows = []
    for record_id, protein in records:
        aas = normalize_aas(protein)
        if "*" in aas.rstrip("*") or aas.endswith("**"):
            raise ValueError(f"Internal or repeated stop in protein {record_id}")
        aas = aas.rstrip("*") + "*"
        if len(aas) <= 1 or len(aas) > max_length:
            raise ValueError(f"Protein {record_id} must have 1–{max_length - 1} amino acids (excluding the terminal stop)")
        for sample_id in range(samples):
            rows.append({"record_id": f"{record_id}|sample_{sample_id + 1}", "gene_name": record_id, "aas": aas})
    spec = InferenceSpec(method="seq2seq", dataset="", checkpoint_path=str(folder / "generator.pt"),
                         species=metadata["species"], max_aa_len=max_length)
    result = generate_seq2seq_rows(rows, spec=spec, preferred_stop="TAA", batch_size=batch_size,
                                  device_name=device, dtype_name="fp32", sample=not greedy,
                                  temperature=temperature, top_k=top_k, sample_seed=seed)
    for row in result:
        if translate_cds(row["predicted_cds"]) != row["aas"]:
            raise RuntimeError(f"Generated CDS does not encode input protein: {row['record_id']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True)
    parser.add_argument("--models-dir", default="models")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--protein", help="One amino-acid sequence")
    inputs.add_argument("--input", help="Protein FASTA file")
    parser.add_argument("--output", default="predictions_cds.csv")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    records = read_fasta(args.input) if args.input else [("protein_1", args.protein)]
    rows = generate(records, model_directory(args.models_dir, args.species), samples=args.samples,
                    device=args.device, batch_size=args.batch_size, temperature=args.temperature,
                    top_k=args.top_k, seed=args.seed, greedy=args.greedy)
    write_csv(args.output, rows)
    print(f"Generated {len(rows)} CDSs: {args.output}")


if __name__ == "__main__":
    main()

