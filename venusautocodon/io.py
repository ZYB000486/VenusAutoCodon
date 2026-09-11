from __future__ import annotations

import csv
from pathlib import Path


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name = None
    chunks: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks).upper()))
            name = line[1:].strip()
            if not name:
                raise ValueError("FASTA headers must not be empty")
            chunks = []
        elif name is None:
            raise ValueError("Expected a FASTA header before sequence data")
        else:
            chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks).upper()))
    if not records or any(not sequence for _, sequence in records):
        raise ValueError("FASTA must contain non-empty sequences")
    if len({name for name, _ in records}) != len(records):
        raise ValueError("FASTA record identifiers must be unique")
    return records


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No output records")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_directory(root: str | Path, species: str) -> Path:
    if not species or any(c not in "abcdefghijklmnopqrstuvwxyz_" for c in species):
        raise ValueError("Use a species identifier from models/manifest.json")
    folder = Path(root) / species
    if not (folder / "metadata.json").exists():
        raise FileNotFoundError(f"Model not downloaded. Run: python -m venusautocodon.download_models --species {species} --output {root}")
    return folder

