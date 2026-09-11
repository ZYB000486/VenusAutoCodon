from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vocab import encode_source, normalize_aas


@dataclass(frozen=True)
class PromptRecord:
    record_id: str
    gene_name: str
    aas: str
    source_ids: tuple[int, ...]


def iter_fasta_records(path: Path):
    header: str | None = None
    seq_chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        yield header, "".join(seq_chunks)


def load_library_records(fasta_path: Path, max_aa_len: int) -> tuple[dict[str, object], list[PromptRecord]]:
    records: list[PromptRecord] = []
    invalid = 0
    too_long = 0
    total_entries = 0
    for idx, (header, seq) in enumerate(iter_fasta_records(fasta_path)):
        total_entries += 1
        try:
            aas = normalize_aas(seq)
            source_ids = tuple(encode_source(aas))
        except Exception:
            invalid += 1
            continue
        if len(source_ids) == 0 or len(source_ids) > max_aa_len:
            too_long += 1
            continue
        records.append(
            PromptRecord(
                record_id=f"lib_{idx}",
                gene_name=f"lib_{idx}",
                aas=aas,
                source_ids=source_ids,
            )
        )
    meta = {
        "prompt_source": "protein_library",
        "prompt_fasta": str(fasta_path),
        "total_entries": int(total_entries),
        "valid_entries": int(len(records)),
        "invalid_entries": int(invalid),
        "too_long_entries": int(too_long),
    }
    return meta, records


def build_library_splits(
    fasta_path: Path,
    *,
    max_aa_len: int,
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[dict[str, object], list[PromptRecord], list[PromptRecord]]:
    meta, records = load_library_records(fasta_path, max_aa_len=max_aa_len)
    required = int(train_size) + int(test_size)
    if len(records) < required:
        raise ValueError(
            f"Protein library does not contain enough valid sequences: requested={required} available={len(records)}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    train_end = int(train_size)
    test_end = train_end + int(test_size)
    selected = [records[int(idx)] for idx in order[:test_end]]
    train_records = selected[:train_end]
    test_records = selected[train_end:test_end]
    meta = {
        **meta,
        "split_seed": int(seed),
        "train_size": int(len(train_records)),
        "test_size": int(len(test_records)),
    }
    return meta, train_records, test_records


def load_prompt_records_csv(path: Path, max_aa_len: int) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"record_id", "gene_name", "aas"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prompt CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            aas = normalize_aas(str(row["aas"]))
            source_ids = tuple(encode_source(aas))
            if len(source_ids) == 0 or len(source_ids) > max_aa_len:
                continue
            records.append(
                PromptRecord(
                    record_id=str(row["record_id"]),
                    gene_name=str(row["gene_name"]),
                    aas=aas,
                    source_ids=source_ids,
                )
            )
    if not records:
        raise ValueError(f"No valid prompt records found in CSV: {path}")
    return records


def save_prompt_records_csv(path: Path, records: list[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "gene_name", "aas", "aa_length"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "record_id": record.record_id,
                    "gene_name": record.gene_name,
                    "aas": record.aas,
                    "aa_length": len(record.source_ids),
                }
            )
