"""Prepare fresh UniRef50 protein prompts for a new AutoResearch run."""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def fasta_records(text: str):
    name, chunks = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(chunks)
            name, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip().upper())
    if name is not None:
        yield name, "".join(chunks)


def download_prompts(train_path: Path, validation_path: Path, *, count: int = 10000,
                     validation_count: int = 300, max_aa_len: int = 512, seed: int = 42) -> None:
    if not 0 < validation_count < count:
        raise ValueError("Require 0 < validation_count < count")
    if train_path.resolve() == validation_path.resolve():
        raise ValueError("Training and validation paths must differ")
    if train_path.exists() or validation_path.exists():
        raise FileExistsError("Refusing to overwrite an existing prompt split")
    url = "https://rest.uniprot.org/uniref/search?query=identity%3A0.5&format=fasta&size=500"
    records, seen, pages = [], set(), 0
    release = None
    while url and len(records) < count:
        with urllib.request.urlopen(url, timeout=120) as response:
            release = response.headers.get("X-UniProt-Release", release)
            text = response.read().decode()
            next_link = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", ""))
            url = next_link.group(1) if next_link else None
        pages += 1
        for name, sequence in fasta_records(text):
            if 0 < len(sequence) < max_aa_len and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY") and sequence not in seen:
                seen.add(sequence)
                records.append((name, sequence))
                if len(records) == count:
                    break
        print(f"UniRef prompts: {len(records)}/{count} ({pages} pages)", flush=True)
        if pages >= 2000:
            raise RuntimeError("Could not collect enough valid UniRef prompts")
    if len(records) != count:
        raise RuntimeError(f"Only {len(records)} valid prompts available; expected {count}")
    random.Random(seed).shuffle(records)
    for path, subset in [(validation_path, records[:validation_count]), (train_path, records[validation_count:])]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in subset))
    metadata = {"source": "UniRef50", "release": release, "accessed": datetime.now(timezone.utc).isoformat(),
                "count": count, "validation_count": validation_count, "max_aa_len": max_aa_len,
                "seed": seed, "selection": "first valid unique sequences from the API, shuffled before splitting"}
    train_path.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "assets/prompt_library"))
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--validation-count", type=int, default=300)
    args = parser.parse_args()
    folder = Path(args.output_dir)
    download_prompts(folder / "uniref10_fixed_train.fasta", folder / "uniref10_fixed_validation.fasta",
                     count=args.count, validation_count=args.validation_count)


if __name__ == "__main__":
    main()
