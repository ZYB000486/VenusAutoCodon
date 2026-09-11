"""Score CDSs with the selected species-specific reward model."""
from __future__ import annotations

import argparse
from pathlib import Path

from autoresearch.rm_autoresearch.prepare import HandcraftedRewardModel, normalize_cds
from .io import model_directory, read_fasta, write_csv


def predict(cds_list: list[str], model_dir: str | Path) -> list[float]:
    sequences = [normalize_cds(cds) for cds in cds_list]
    model = HandcraftedRewardModel.from_path(Path(model_dir) / "reward_model.json")
    return model.predict_abundance(sequences).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True)
    parser.add_argument("--models-dir", default="models")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cds", help="One coding DNA sequence")
    inputs.add_argument("--input", help="CDS FASTA file")
    parser.add_argument("--output", default="predictions_reward.csv")
    args = parser.parse_args()
    records = read_fasta(args.input) if args.input else [("sequence_1", args.cds)]
    scores = predict([seq for _, seq in records], model_directory(args.models_dir, args.species))
    write_csv(args.output, [{"record_id": name, "cds": normalize_cds(seq), "reward_score": score}
                            for (name, seq), score in zip(records, scores)])
    print(f"Scored {len(records)} CDSs: {args.output}")


if __name__ == "__main__":
    main()

