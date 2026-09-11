from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autoresearch.common import save_json
    from autoresearch.dataset import discover_species_datasets, load_records, resolve_dataset
else:
    from .common import save_json
    from .dataset import discover_species_datasets, load_records, resolve_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one autoresearch dataset and recommend the execution path.")
    parser.add_argument("--species", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max-aa-len", type=int, default=512)
    parser.add_argument("--list-species", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_species:
        print(json.dumps({name: str(path) for name, path in discover_species_datasets().items()}, indent=2, ensure_ascii=False))
        return

    dataset_path = resolve_dataset(species=args.species, dataset=args.dataset)
    dataset_meta, _ = load_records(
        species=args.species,
        dataset=args.dataset,
        max_aa_len=args.max_aa_len,
        require_abundance=False,
    )
    summary = {
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_meta["dataset_name"],
        "has_abundance": dataset_meta["has_abundance"],
        "recommended_mode": "policy_init_then_karpathy_rm_rl" if dataset_meta["has_abundance"] else "policy_init_then_evolution_hybrid",
        "dataset_meta": dataset_meta,
    }
    if args.output:
        save_json(Path(args.output).expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
