from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import DATA_DIR
from .vocab import (
    SRC_PAD_ID,
    TGT_BOS_ID,
    TGT_PAD_ID,
    encode_source,
    encode_target_from_codons,
    split_codons,
    translate_cds,
)

REQUIRED_BASE_COLUMNS = ("gene_name", "cds")


def discover_species_datasets() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for csv_path in sorted(DATA_DIR.glob("*/*.csv")):
        mapping[csv_path.parent.name] = csv_path.resolve()
    return mapping


def resolve_dataset(species: str | None = None, dataset: str | Path | None = None) -> Path:
    if dataset is not None:
        return Path(dataset).expanduser().resolve()
    if species is None:
        raise ValueError("Either species or dataset must be provided")
    mapping = discover_species_datasets()
    if species not in mapping:
        raise KeyError(f"Unknown species={species!r}. Available={sorted(mapping)}")
    return mapping[species]


def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_BASE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    columns = list(REQUIRED_BASE_COLUMNS)
    has_abundance = "abundance" in df.columns
    if has_abundance:
        columns.append("abundance")

    cleaned = df[columns].copy()
    cleaned["gene_name"] = cleaned["gene_name"].astype(str).str.strip()
    cleaned["cds"] = cleaned["cds"].astype(str).str.strip().str.upper().str.replace("U", "T", regex=False)
    cleaned = cleaned[cleaned["gene_name"] != ""]
    cleaned = cleaned[cleaned["cds"] != ""]
    cleaned = cleaned[cleaned["cds"].map(lambda text: len(text) % 3 == 0)]
    cleaned = cleaned[cleaned["cds"].map(lambda text: set(text).issubset({"A", "C", "G", "T"}))]

    if has_abundance:
        cleaned["abundance"] = pd.to_numeric(cleaned["abundance"], errors="coerce")
        cleaned = cleaned.dropna(subset=["abundance"])

    return cleaned.reset_index(drop=True)


def load_dataset_frame(species: str | None = None, dataset: str | Path | None = None) -> tuple[Path, pd.DataFrame]:
    path = resolve_dataset(species=species, dataset=dataset)
    df = pd.read_csv(path)
    return path, _normalize_table(df)


def dataset_has_abundance(species: str | None = None, dataset: str | Path | None = None) -> bool:
    _, df = load_dataset_frame(species=species, dataset=dataset)
    return "abundance" in df.columns


@dataclass(frozen=True)
class SequenceRecord:
    gene_name: str
    cds: str
    aas: str
    codons: tuple[str, ...]
    source_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    abundance: float | None

    @property
    def aa_length(self) -> int:
        return len(self.source_ids)


class Seq2SeqDataset(Dataset):
    def __init__(self, records: list[SequenceRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SequenceRecord:
        return self.records[index]


def load_records(
    species: str | None = None,
    dataset: str | Path | None = None,
    *,
    max_aa_len: int = 512,
    require_abundance: bool = False,
) -> tuple[dict[str, Any], list[SequenceRecord]]:
    dataset_path, df = load_dataset_frame(species=species, dataset=dataset)
    has_abundance = "abundance" in df.columns
    if require_abundance and not has_abundance:
        raise ValueError(f"{dataset_path} does not provide abundance labels")

    records: list[SequenceRecord] = []
    skipped_invalid = 0
    skipped_too_long = 0

    for row in df.itertuples(index=False):
        gene_name = str(row.gene_name)
        cds = str(row.cds).strip().upper()
        abundance = float(row.abundance) if has_abundance else None
        try:
            aas = translate_cds(cds)
            codons = tuple(split_codons(cds))
            source_ids = tuple(encode_source(aas))
            target_ids = tuple(encode_target_from_codons(codons))
        except Exception:
            skipped_invalid += 1
            continue

        if len(source_ids) == 0 or len(source_ids) > max_aa_len:
            skipped_too_long += 1
            continue
        if len(source_ids) != len(target_ids):
            skipped_invalid += 1
            continue

        records.append(
            SequenceRecord(
                gene_name=gene_name,
                cds=cds,
                aas=aas,
                codons=codons,
                source_ids=source_ids,
                target_ids=target_ids,
                abundance=abundance,
            )
        )

    aa_lengths = np.array([record.aa_length for record in records], dtype=np.int64) if records else np.array([], dtype=np.int64)
    abundance_values = (
        np.array([record.abundance for record in records if record.abundance is not None], dtype=np.float64)
        if has_abundance
        else np.array([], dtype=np.float64)
    )
    meta = {
        "dataset_name": dataset_path.parent.name,
        "dataset_path": str(dataset_path),
        "has_abundance": bool(has_abundance),
        "raw_rows": int(len(df)),
        "kept_rows": int(len(records)),
        "skipped_invalid_rows": int(skipped_invalid),
        "skipped_too_long_rows": int(skipped_too_long),
        "max_aa_len": int(max_aa_len),
        "aa_length": {
            "mean": float(aa_lengths.mean()) if aa_lengths.size else 0.0,
            "median": float(np.median(aa_lengths)) if aa_lengths.size else 0.0,
            "p95": int(np.quantile(aa_lengths, 0.95)) if aa_lengths.size else 0,
            "max": int(aa_lengths.max()) if aa_lengths.size else 0,
        },
    }
    if abundance_values.size:
        meta["abundance"] = {
            "mean": float(abundance_values.mean()),
            "std": float(abundance_values.std()),
            "min": float(abundance_values.min()),
            "max": float(abundance_values.max()),
        }
    return meta, records


def split_records(
    records: list[SequenceRecord],
    *,
    seed: int = 42,
    train_frac: float = 0.85,
) -> dict[str, list[SequenceRecord]]:
    if not records:
        raise ValueError("No records available for splitting")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("Invalid split fraction")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    train_end = int(len(order) * train_frac)
    return {
        "train": [records[int(idx)] for idx in order[:train_end]],
        "test": [records[int(idx)] for idx in order[train_end:]],
    }


def slice_splits(
    split_map: dict[str, list[SequenceRecord]],
    *,
    max_train_records: int | None = None,
    max_test_records: int | None = None,
) -> dict[str, list[SequenceRecord]]:
    limits = {
        "train": max_train_records,
        "test": max_test_records,
    }
    return {
        split_name: items if limits[split_name] is None else items[: limits[split_name]]
        for split_name, items in split_map.items()
    }


def collate_examples(batch: list[SequenceRecord]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    src_max_len = max(item.aa_length for item in batch)
    tgt_max_len = max(len(item.target_ids) for item in batch)

    src_ids = []
    tgt_input_ids = []
    labels = []
    gene_names = []
    aas = []
    cds = []
    abundance = []

    for item in batch:
        src = list(item.source_ids)
        tgt = list(item.target_ids)
        src_pad = src + [SRC_PAD_ID] * (src_max_len - len(src))
        tgt_in = [TGT_BOS_ID] + tgt[:-1]
        tgt_in_pad = tgt_in + [TGT_PAD_ID] * (tgt_max_len - len(tgt_in))
        labels_pad = tgt + [TGT_PAD_ID] * (tgt_max_len - len(tgt))

        src_ids.append(src_pad)
        tgt_input_ids.append(tgt_in_pad)
        labels.append(labels_pad)
        gene_names.append(item.gene_name)
        aas.append(item.aas)
        cds.append(item.cds)
        abundance.append(item.abundance)

    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    tgt_input_tensor = torch.tensor(tgt_input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return {
        "src_ids": src_tensor,
        "src_padding_mask": src_tensor.eq(SRC_PAD_ID),
        "tgt_input_ids": tgt_input_tensor,
        "tgt_padding_mask": tgt_input_tensor.eq(TGT_PAD_ID),
        "labels": labels_tensor,
        "gene_name": gene_names,
        "aas": aas,
        "cds": cds,
        "abundance": abundance,
    }
