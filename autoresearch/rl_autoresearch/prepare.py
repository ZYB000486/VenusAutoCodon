from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

TIME_BUDGET = 900

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY*")
AA_TO_CODONS = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "C": ["TGT", "TGC"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["TTT", "TTC"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "K": ["AAA", "AAG"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "M": ["ATG"],
    "N": ["AAT", "AAC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "*": ["TAA", "TAG", "TGA"],
}
CODON_TOKENS = sorted({codon for codons in AA_TO_CODONS.values() for codon in codons})
CODON_TO_AA = {codon: aa for aa, codons in AA_TO_CODONS.items() for codon in codons}
SRC_VOCAB = SPECIAL_TOKENS + AA_TOKENS
TGT_VOCAB = SPECIAL_TOKENS + CODON_TOKENS
SRC_TOKEN_TO_ID = {token: idx for idx, token in enumerate(SRC_VOCAB)}
TGT_TOKEN_TO_ID = {token: idx for idx, token in enumerate(TGT_VOCAB)}
TGT_ID_TO_TOKEN = {idx: token for token, idx in TGT_TOKEN_TO_ID.items()}
SRC_PAD_ID = SRC_TOKEN_TO_ID["<pad>"]
TGT_BOS_ID = TGT_TOKEN_TO_ID["<bos>"]
TGT_PAD_ID = TGT_TOKEN_TO_ID["<pad>"]
REQUIRED_BASE_COLUMNS = ("gene_name", "cds")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_TRAIN_PROMPT_FASTA = (
    Path(__file__).resolve().parents[1] / "assets" / "prompt_library" / "uniref10_fixed_train.fasta"
)
DEFAULT_VALIDATION_PROMPT_FASTA = (
    Path(__file__).resolve().parents[1] / "assets" / "prompt_library" / "uniref10_fixed_validation.fasta"
)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_dataset(species: str | None = None, dataset: str | Path | None = None) -> Path:
    if dataset is not None:
        return Path(dataset).expanduser().resolve()
    if species is None:
        raise ValueError("Either species or dataset must be provided")
    candidate = DEFAULT_DATA_DIR / species / f"{species}.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset not found for species={species!r}: {candidate}")
    return candidate.resolve()


def normalize_aas(seq: str) -> str:
    text = (seq or "").strip().upper()
    if not text:
        raise ValueError("Empty amino-acid sequence")
    invalid = sorted({token for token in text if token not in AA_TO_CODONS})
    if invalid:
        raise ValueError(f"Invalid amino-acid tokens: {''.join(invalid)}")
    return text


def normalize_cds(seq: str) -> str:
    text = (seq or "").strip().upper().replace("U", "T")
    if not text:
        raise ValueError("Empty CDS")
    if len(text) % 3 != 0:
        raise ValueError("CDS length must be divisible by 3")
    invalid = sorted({token for token in text if token not in {"A", "C", "G", "T"}})
    if invalid:
        raise ValueError(f"Invalid nucleotide tokens: {''.join(invalid)}")
    return text


def split_codons(cds: str) -> list[str]:
    seq = normalize_cds(cds)
    return [seq[index : index + 3] for index in range(0, len(seq), 3)]


def translate_cds(cds: str) -> str:
    aas: list[str] = []
    for codon in split_codons(cds):
        aa = CODON_TO_AA.get(codon)
        if aa is None:
            raise ValueError(f"Unsupported codon: {codon}")
        aas.append(aa)
    return "".join(aas)


def encode_source(aas: str) -> list[int]:
    return [SRC_TOKEN_TO_ID[token] for token in normalize_aas(aas)]


def encode_target_from_codons(codons: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for codon in codons:
        token = codon.upper()
        if token not in TGT_TOKEN_TO_ID:
            raise ValueError(f"Unknown codon token: {token}")
        ids.append(TGT_TOKEN_TO_ID[token])
    return ids


def decode_target_ids(token_ids: Iterable[int], skip_special: bool = True) -> list[str]:
    codons: list[str] = []
    for token_id in token_ids:
        token = TGT_ID_TO_TOKEN[int(token_id)]
        if skip_special and token in SPECIAL_TOKENS:
            continue
        codons.append(token)
    return codons


def decode_cds(token_ids: Iterable[int], skip_special: bool = True) -> str:
    return "".join(decode_target_ids(token_ids, skip_special=skip_special))


def build_constraint_bias() -> torch.Tensor:
    bias = torch.zeros((len(SRC_VOCAB), len(TGT_VOCAB)), dtype=torch.float32)
    valid_target_ids = {TGT_TOKEN_TO_ID[codon] for codon in CODON_TOKENS}
    all_invalid = torch.full((len(TGT_VOCAB),), -1e9, dtype=torch.float32)
    for aa in AA_TOKENS:
        row = all_invalid.clone()
        for codon in AA_TO_CODONS[aa]:
            row[TGT_TOKEN_TO_ID[codon]] = 0.0
        bias[SRC_TOKEN_TO_ID[aa]] = row
    for token in SPECIAL_TOKENS:
        token_id = SRC_TOKEN_TO_ID[token]
        bias[token_id, list(valid_target_ids)] = 0.0
    return bias


def load_dataset_frame(species: str | None = None, dataset: str | Path | None = None) -> tuple[Path, pd.DataFrame]:
    path = resolve_dataset(species=species, dataset=dataset)
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_BASE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    columns = list(REQUIRED_BASE_COLUMNS)
    has_abundance = "abundance" in frame.columns
    if has_abundance:
        columns.append("abundance")
    cleaned = frame[columns].copy()
    cleaned["gene_name"] = cleaned["gene_name"].astype(str).str.strip()
    cleaned["cds"] = cleaned["cds"].astype(str).str.strip().str.upper().str.replace("U", "T", regex=False)
    cleaned = cleaned[cleaned["gene_name"] != ""]
    cleaned = cleaned[cleaned["cds"] != ""]
    cleaned = cleaned[cleaned["cds"].map(lambda text: len(text) % 3 == 0)]
    cleaned = cleaned[cleaned["cds"].map(lambda text: set(text).issubset({"A", "C", "G", "T"}))]
    if has_abundance:
        cleaned["abundance"] = pd.to_numeric(cleaned["abundance"], errors="coerce")
        cleaned = cleaned.dropna(subset=["abundance"])
    return path, cleaned.reset_index(drop=True)


@dataclass(frozen=True)
class SequenceRecord:
    gene_name: str
    cds: str
    aas: str
    source_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    abundance: float | None

    @property
    def aa_length(self) -> int:
        return len(self.source_ids)


def load_records(
    species: str | None = None,
    dataset: str | Path | None = None,
    *,
    max_aa_len: int = 1024,
) -> tuple[dict[str, Any], list[SequenceRecord]]:
    dataset_path, frame = load_dataset_frame(species=species, dataset=dataset)
    has_abundance = "abundance" in frame.columns

    records: list[SequenceRecord] = []
    skipped_invalid = 0
    skipped_too_long = 0
    for row in frame.itertuples(index=False):
        gene_name = str(row.gene_name)
        cds = str(row.cds).strip().upper()
        abundance = float(row.abundance) if has_abundance else None
        try:
            aas = translate_cds(cds)
            source_ids = tuple(encode_source(aas))
            target_ids = tuple(encode_target_from_codons(split_codons(cds)))
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
                source_ids=source_ids,
                target_ids=target_ids,
                abundance=abundance,
            )
        )

    aa_lengths = np.asarray([record.aa_length for record in records], dtype=np.int64) if records else np.asarray([], dtype=np.int64)
    abundance_values = (
        np.asarray([record.abundance for record in records if record.abundance is not None], dtype=np.float64)
        if has_abundance
        else np.asarray([], dtype=np.float64)
    )
    meta = {
        "dataset_name": dataset_path.parent.name,
        "dataset_path": str(dataset_path),
        "has_abundance": bool(has_abundance),
        "raw_rows": int(len(frame)),
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


def iter_fasta_records(path: Path) -> Iterable[tuple[str, str]]:
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


def load_prompt_records_from_fasta(
    fasta_path: Path,
    *,
    max_aa_len: int,
    split_name: str,
    expected_size: int | None = None,
) -> tuple[dict[str, Any], list[SequenceRecord]]:
    records: list[SequenceRecord] = []
    invalid = 0
    too_long = 0
    total_entries = 0
    for idx, (_header, seq) in enumerate(iter_fasta_records(fasta_path)):
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
            SequenceRecord(
                gene_name=f"{split_name}_prompt_{idx}",
                cds="",
                aas=aas,
                source_ids=source_ids,
                target_ids=tuple(),
                abundance=None,
            )
        )
    if expected_size is not None and len(records) != expected_size:
        raise ValueError(
            f"Prompt split size mismatch for {fasta_path}: expected={expected_size} loaded={len(records)}"
        )
    meta = {
        "prompt_source": "fixed_fasta_split",
        "prompt_fasta": str(fasta_path),
        "split_name": split_name,
        "total_entries": int(total_entries),
        "valid_entries": int(len(records)),
        "invalid_entries": int(invalid),
        "too_long_entries": int(too_long),
        "loaded_records": int(len(records)),
    }
    return meta, records


def gc_fraction(cds: str) -> float:
    seq = cds.upper()
    return float(sum(base in {"G", "C"} for base in seq)) / float(max(1, len(seq)))


def gc3_fraction(cds: str) -> float:
    codons = split_codons(cds)
    return float(sum(codon[2] in {"G", "C"} for codon in codons)) / float(max(1, len(codons)))


def prefix_gc3_fraction(cds: str, prefix_codons: int) -> float:
    codons = split_codons(cds)[:prefix_codons]
    return float(sum(codon[2] in {"G", "C"} for codon in codons)) / float(max(1, len(codons)))


@dataclass
class ReferenceStats:
    aa_codon_probs: dict[str, dict[str, float]]
    codon_pair_probs: dict[str, float]
    top_codon_pairs: list[str]
    gc_mean: float
    gc_std: float
    gc3_mean: float
    gc3_std: float
    prefix_gc3_mean: float
    prefix_gc3_std: float
    length_mean: float
    length_std: float
    prefix_window: int = 10
    pair_floor: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceStats":
        return cls(**payload)


def normalize_counts(counts: dict[str, float], allowed: list[str], pseudocount: float = 1.0) -> dict[str, float]:
    total = float(sum(counts.get(token, 0.0) for token in allowed) + pseudocount * len(allowed))
    return {
        token: float(counts.get(token, 0.0) + pseudocount) / total
        for token in allowed
    }


def compute_reference_stats(
    cds_list: list[str],
    *,
    top_pair_k: int = 64,
    prefix_window: int = 10,
) -> ReferenceStats:
    aa_codon_counts = {aa: {codon: 0.0 for codon in codons} for aa, codons in AA_TO_CODONS.items()}
    pair_counts: dict[str, float] = {}
    gc_values: list[float] = []
    gc3_values: list[float] = []
    prefix_gc3_values: list[float] = []
    length_values: list[int] = []

    for cds in cds_list:
        codons = split_codons(cds)
        length_values.append(len(codons))
        gc_values.append(gc_fraction(cds))
        gc3_values.append(gc3_fraction(cds))
        prefix_gc3_values.append(prefix_gc3_fraction(cds, prefix_window))
        for codon in codons:
            aa = CODON_TO_AA.get(codon)
            if aa is not None:
                aa_codon_counts[aa][codon] += 1.0
        for left, right in zip(codons[:-1], codons[1:]):
            key = f"{left}|{right}"
            pair_counts[key] = pair_counts.get(key, 0.0) + 1.0

    aa_codon_probs = {
        aa: normalize_counts(counts, AA_TO_CODONS[aa])
        for aa, counts in aa_codon_counts.items()
    }
    sorted_pairs = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
    top_codon_pairs = [pair for pair, _ in sorted_pairs[:top_pair_k]]
    total_pair_counts = float(sum(pair_counts.values()) + max(1, len(pair_counts)))
    pair_floor = 1.0 / total_pair_counts
    codon_pair_probs = {
        pair: float(count + 1.0) / total_pair_counts
        for pair, count in pair_counts.items()
    }

    gc_arr = np.asarray(gc_values, dtype=np.float64)
    gc3_arr = np.asarray(gc3_values, dtype=np.float64)
    prefix_gc3_arr = np.asarray(prefix_gc3_values, dtype=np.float64)
    length_arr = np.asarray(length_values, dtype=np.float64)
    return ReferenceStats(
        aa_codon_probs=aa_codon_probs,
        codon_pair_probs=codon_pair_probs,
        top_codon_pairs=top_codon_pairs,
        gc_mean=float(gc_arr.mean()) if gc_arr.size else 0.0,
        gc_std=float(gc_arr.std()) if gc_arr.size else 1.0,
        gc3_mean=float(gc3_arr.mean()) if gc3_arr.size else 0.0,
        gc3_std=float(gc3_arr.std()) if gc3_arr.size else 1.0,
        prefix_gc3_mean=float(prefix_gc3_arr.mean()) if prefix_gc3_arr.size else 0.0,
        prefix_gc3_std=float(prefix_gc3_arr.std()) if prefix_gc3_arr.size else 1.0,
        length_mean=float(length_arr.mean()) if length_arr.size else 0.0,
        length_std=float(length_arr.std()) if length_arr.size else 1.0,
        prefix_window=int(prefix_window),
        pair_floor=float(pair_floor),
    )


def cai_like_score(cds: str, reference: ReferenceStats) -> float:
    codons = split_codons(cds)
    if not codons:
        return 0.0
    log_sum = 0.0
    for codon in codons:
        aa = CODON_TO_AA.get(codon)
        if aa is None:
            continue
        prob = reference.aa_codon_probs.get(aa, {}).get(codon, 1e-6)
        log_sum += math.log(max(prob, 1e-12))
    return math.exp(log_sum / max(1, len(codons)))


def cpai_like_score(cds: str, reference: ReferenceStats) -> float:
    codons = split_codons(cds)
    if len(codons) < 2:
        return 0.0
    log_sum = 0.0
    total = 0
    for left, right in zip(codons[:-1], codons[1:]):
        key = f"{left}|{right}"
        prob = reference.codon_pair_probs.get(key, reference.pair_floor)
        log_sum += math.log(max(prob, 1e-12))
        total += 1
    return math.exp(log_sum / max(1, total))


def prefix_cai_score(cds: str, reference: ReferenceStats) -> float:
    codons = split_codons(cds)[: reference.prefix_window]
    if not codons:
        return 0.0
    return cai_like_score("".join(codons), reference)


def zscore(value: float, mean: float, std: float) -> float:
    if std < 1e-8:
        return 0.0
    return (value - mean) / std


def build_feature_matrix(cds_list: list[str], reference: ReferenceStats) -> tuple[np.ndarray, list[str]]:
    feature_rows: list[list[float]] = []
    feature_names = [
        "length_codons",
        "gc_frac",
        "gc3_frac",
        "cai_like",
        "cpai_like",
        "prefix_cai_first10codons",
        "gc_z",
        "gc3_z",
        "length_z",
    ]
    feature_names.extend([f"codon_frac::{codon}" for codon in CODON_TOKENS])
    feature_names.extend([f"prefix_codon_frac::{codon}" for codon in CODON_TOKENS])
    feature_names.extend([f"pair_frac::{pair}" for pair in reference.top_codon_pairs])

    for cds in cds_list:
        codons = split_codons(cds)
        prefix_codons = codons[: reference.prefix_window]
        codon_total = max(1, len(codons))
        prefix_total = max(1, len(prefix_codons))
        codon_counts = {codon: 0 for codon in CODON_TOKENS}
        prefix_counts = {codon: 0 for codon in CODON_TOKENS}
        pair_counts = {pair: 0 for pair in reference.top_codon_pairs}

        for codon in codons:
            codon_counts[codon] = codon_counts.get(codon, 0) + 1
        for codon in prefix_codons:
            prefix_counts[codon] = prefix_counts.get(codon, 0) + 1
        for left, right in zip(codons[:-1], codons[1:]):
            key = f"{left}|{right}"
            if key in pair_counts:
                pair_counts[key] += 1

        gc = gc_fraction(cds)
        gc3 = gc3_fraction(cds)
        row = [
            float(len(codons)),
            gc,
            gc3,
            cai_like_score(cds, reference),
            cpai_like_score(cds, reference),
            prefix_cai_score(cds, reference),
            zscore(gc, reference.gc_mean, reference.gc_std),
            zscore(gc3, reference.gc3_mean, reference.gc3_std),
            zscore(float(len(codons)), reference.length_mean, reference.length_std),
        ]
        row.extend(float(codon_counts[codon]) / float(codon_total) for codon in CODON_TOKENS)
        row.extend(float(prefix_counts[codon]) / float(prefix_total) for codon in CODON_TOKENS)
        row.extend(float(pair_counts[pair]) / float(max(1, len(codons) - 1)) for pair in reference.top_codon_pairs)
        feature_rows.append(row)

    return np.asarray(feature_rows, dtype=np.float64), feature_names


@dataclass
class RewardBundle:
    reference: dict[str, Any]
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    weights: list[float]
    target_mean: float
    target_std: float
    alpha: float
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewardBundle":
        return cls(**payload)


def standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = std.copy()
    safe_std[safe_std < 1e-8] = 1.0
    return (matrix - mean) / safe_std


def predict_linear(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return x_aug @ weights


class HandcraftedRewardModel:
    def __init__(self, bundle: RewardBundle) -> None:
        self.bundle = bundle
        self.reference = ReferenceStats.from_dict(bundle.reference)
        self.feature_mean = np.asarray(bundle.feature_mean, dtype=np.float64)
        self.feature_std = np.asarray(bundle.feature_std, dtype=np.float64)
        self.weights = np.asarray(bundle.weights, dtype=np.float64)

    @classmethod
    def from_path(cls, path: str | Path) -> "HandcraftedRewardModel":
        payload = RewardBundle.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        return cls(payload)

    def predict_abundance(self, cds_list: list[str]) -> np.ndarray:
        x, _ = build_feature_matrix(cds_list, self.reference)
        x_std = standardize(x, self.feature_mean, self.feature_std)
        pred_z = predict_linear(x_std, self.weights)
        return pred_z * self.bundle.target_std + self.bundle.target_mean

    def predict_reward(self, cds_list: list[str]) -> np.ndarray:
        abundance = self.predict_abundance(cds_list)
        return (abundance - self.bundle.target_mean) / max(self.bundle.target_std, 1e-8)


@dataclass
class Seq2SeqConfig:
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 1024
    use_codon_constraint: bool = True

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def config_from_dict(payload: dict[str, Any]) -> Seq2SeqConfig:
    defaults = Seq2SeqConfig().to_dict()
    valid_names = {field.name for field in fields(Seq2SeqConfig)}
    defaults.update({key: value for key, value in payload.items() if key in valid_names})
    return Seq2SeqConfig(**defaults)


class StandardSeq2SeqTransformer(nn.Module):
    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.src_embedding = nn.Embedding(len(SRC_VOCAB), config.d_model, padding_idx=SRC_PAD_ID)
        self.tgt_embedding = nn.Embedding(len(TGT_VOCAB), config.d_model, padding_idx=TGT_PAD_ID)
        self.src_position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.tgt_position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(config.d_model, len(TGT_VOCAB))
        self.register_buffer("constraint_bias", build_constraint_bias(), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.src_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.tgt_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def embed_source(self, src_ids: torch.Tensor) -> torch.Tensor:
        seq_len = src_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Source length {seq_len} exceeds max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=src_ids.device).unsqueeze(0)
        hidden = self.src_embedding(src_ids) * (self.d_model**0.5)
        return self.dropout(hidden + self.src_position_embedding(positions))

    def embed_target(self, tgt_ids: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        seq_len = tgt_ids.size(1)
        max_pos = start_pos + seq_len
        if max_pos > self.config.max_seq_len:
            raise ValueError(f"Target length {max_pos} exceeds max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(start_pos, max_pos, device=tgt_ids.device).unsqueeze(0)
        hidden = self.tgt_embedding(tgt_ids) * (self.d_model**0.5)
        return self.dropout(hidden + self.tgt_position_embedding(positions))

    def causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def encode_memory(self, src_ids: torch.Tensor, src_padding_mask: torch.Tensor | None) -> torch.Tensor:
        src_hidden = self.embed_source(src_ids)
        return self.transformer.encoder(src_hidden, src_key_padding_mask=src_padding_mask)

    def decode_full(
        self,
        tgt_hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        src_padding_mask: torch.Tensor | None,
        tgt_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        tgt_mask = self.causal_mask(tgt_hidden.size(1), tgt_hidden.device)
        return self.transformer.decoder(
            tgt_hidden,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

    def ff_block(self, layer: nn.TransformerDecoderLayer, hidden: torch.Tensor) -> torch.Tensor:
        ff_hidden = layer.linear1(hidden)
        ff_hidden = layer.activation(ff_hidden)
        ff_hidden = layer.dropout(ff_hidden)
        ff_hidden = layer.linear2(ff_hidden)
        return layer.dropout3(ff_hidden)

    def decode_next_token(
        self,
        current_hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        src_padding_mask: torch.Tensor | None,
        layer_caches: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        hidden = current_hidden
        for layer_idx, layer in enumerate(self.transformer.decoder.layers):
            if not layer.norm_first:
                raise RuntimeError("Cached decoding expects norm_first=True")
            current_norm = layer.norm1(hidden)
            cached_norm = layer_caches[layer_idx]
            self_attn_kv = current_norm if cached_norm is None else torch.cat([cached_norm, current_norm], dim=1)
            self_attn_out = layer.self_attn(current_norm, self_attn_kv, self_attn_kv, need_weights=False)[0]
            hidden = hidden + layer.dropout1(self_attn_out)
            cross_query = layer.norm2(hidden)
            cross_attn_out = layer.multihead_attn(
                cross_query,
                memory,
                memory,
                key_padding_mask=src_padding_mask,
                need_weights=False,
            )[0]
            hidden = hidden + layer.dropout2(cross_attn_out)
            ff_input = layer.norm3(hidden)
            hidden = hidden + self.ff_block(layer, ff_input)
            layer_caches[layer_idx] = self_attn_kv
        if self.transformer.decoder.norm is not None:
            hidden = self.transformer.decoder.norm(hidden)
        return hidden, layer_caches

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        *,
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_len = tgt_input_ids.size(1)
        if tgt_len > src_ids.size(1):
            raise ValueError("Target sequence cannot be longer than aligned source sequence")
        memory = self.encode_memory(src_ids, src_padding_mask)
        tgt_hidden = self.embed_target(tgt_input_ids)
        hidden = self.decode_full(
            tgt_hidden,
            memory,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )
        logits = self.output_projection(hidden)
        if self.config.use_codon_constraint:
            logits = logits + self.constraint_bias[src_ids[:, :tgt_len]]
        return logits

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        *,
        src_padding_mask: torch.Tensor | None = None,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if src_padding_mask is None:
            src_padding_mask = src_ids.eq(SRC_PAD_ID)
        if not use_cache:
            raise ValueError("This unit expects cached decoding")
        batch_size, max_src_len = src_ids.shape
        lengths = (~src_padding_mask).sum(dim=1)
        memory = self.encode_memory(src_ids, src_padding_mask)
        layer_caches: list[torch.Tensor | None] = [None] * len(self.transformer.decoder.layers)
        prev_tokens = torch.full((batch_size, 1), fill_value=TGT_BOS_ID, dtype=torch.long, device=src_ids.device)
        generated: list[torch.Tensor] = []
        for step in range(max_src_len):
            current_hidden = self.embed_target(prev_tokens, start_pos=step)
            hidden, layer_caches = self.decode_next_token(
                current_hidden,
                memory,
                src_padding_mask=src_padding_mask,
                layer_caches=layer_caches,
            )
            next_logits = self.output_projection(hidden).squeeze(1)
            if self.config.use_codon_constraint:
                next_logits = next_logits + self.constraint_bias[src_ids[:, step]]
            if sample:
                if temperature <= 0:
                    raise ValueError("temperature must be positive when sampling")
                next_logits = next_logits / temperature
                if top_k > 0:
                    values, indices = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
                    filtered = torch.full_like(next_logits, float("-inf"))
                    filtered.scatter_(1, indices, values)
                    next_logits = filtered
                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = next_logits.argmax(dim=-1)
            active = step < lengths
            next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, TGT_PAD_ID))
            generated.append(next_tokens)
            prev_tokens = next_tokens.unsqueeze(1)
        return torch.stack(generated, dim=1)


def build_model(config: Seq2SeqConfig) -> StandardSeq2SeqTransformer:
    return StandardSeq2SeqTransformer(config)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[StandardSeq2SeqTransformer, Seq2SeqConfig, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    config = config_from_dict(payload["model_config"])
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    return model, config, payload


@dataclass
class RLTrialSpec:
    species: str | None = None
    dataset: str | None = None
    init_checkpoint: str = ""
    reward_bundle_json: str = ""
    train_prompt_fasta: str = str(DEFAULT_TRAIN_PROMPT_FASTA)
    validation_prompt_fasta: str = str(DEFAULT_VALIDATION_PROMPT_FASTA)
    output_dir: str = ""
    seed: int = 42
    max_aa_len: int = 512
    steps: int = 120
    prompt_batch_size: int = 72
    group_size: int = 4
    validation_prompt_batch_size: int = 300
    policy_mini_batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    entropy_coef: float = 0.0
    grad_clip_norm: float = 1.0
    temperature: float = 1.0
    top_k: int = 0
    eval_every: int = 10
    patience_evals: int = 4
    validation_group_size: int = 4
    dtype: str = "bf16"
    device: str = "cpu"
    best_metric_key: str = "sample_reward_mean"
    max_train_records: int | None = None
    max_validation_records: int | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "RLTrialSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass
class RLTrialContext:
    spec: RLTrialSpec
    dataset_meta: dict[str, Any]
    split_sizes: dict[str, int]
    trial_dir: Path
    train_records: list[SequenceRecord]
    validation_records: list[SequenceRecord]
    reward_model: HandcraftedRewardModel


def load_trial_context(spec_path: str | Path, trial_name: str) -> RLTrialContext:
    spec = RLTrialSpec.from_path(spec_path)
    if not spec.output_dir:
        raise ValueError("RLTrialSpec.output_dir is required")
    if not spec.init_checkpoint:
        raise ValueError("RLTrialSpec.init_checkpoint is required")
    if not spec.reward_bundle_json:
        raise ValueError("RLTrialSpec.reward_bundle_json is required")
    if not spec.train_prompt_fasta:
        raise ValueError("RLTrialSpec.train_prompt_fasta is required")
    if not spec.validation_prompt_fasta:
        raise ValueError("RLTrialSpec.validation_prompt_fasta is required")

    dataset_meta, _ = load_records(species=spec.species, dataset=spec.dataset, max_aa_len=spec.max_aa_len)
    train_prompt_meta, train_prompt_records = load_prompt_records_from_fasta(
        Path(spec.train_prompt_fasta).expanduser().resolve(),
        max_aa_len=spec.max_aa_len,
        split_name="train",
    )
    validation_prompt_meta, validation_prompt_records = load_prompt_records_from_fasta(
        Path(spec.validation_prompt_fasta).expanduser().resolve(),
        max_aa_len=spec.max_aa_len,
        split_name="validation",
    )
    if spec.max_train_records is not None:
        train_prompt_records = train_prompt_records[: spec.max_train_records]
    if spec.max_validation_records is not None:
        validation_prompt_records = validation_prompt_records[: spec.max_validation_records]
    if not train_prompt_records or not validation_prompt_records:
        raise ValueError("RL autoresearch requires non-empty fixed UniRef train and validation prompt splits")
    dataset_meta["rl_fixed_train_prompt_fasta"] = str(Path(spec.train_prompt_fasta).expanduser().resolve())
    dataset_meta["rl_fixed_validation_prompt_fasta"] = str(Path(spec.validation_prompt_fasta).expanduser().resolve())
    dataset_meta["rl_train_prompt_source"] = train_prompt_meta
    dataset_meta["rl_validation_prompt_source"] = validation_prompt_meta
    stage_dir = Path(spec.output_dir).expanduser().resolve()
    trial_dir = stage_dir / "trial_runs" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    reward_model = HandcraftedRewardModel.from_path(spec.reward_bundle_json)
    return RLTrialContext(
        spec=spec,
        dataset_meta=dataset_meta,
        split_sizes={"train": len(train_prompt_records), "validation": len(validation_prompt_records)},
        trial_dir=trial_dir,
        train_records=train_prompt_records,
        validation_records=validation_prompt_records,
        reward_model=reward_model,
    )


def autocast_dtype(dtype_flag: str) -> torch.dtype | None:
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "bf16":
        return torch.bfloat16
    return None


def pad_source_batch(records: list[SequenceRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(record.source_ids) for record in records)
    rows = [list(record.source_ids) + [SRC_PAD_ID] * (max_len - len(record.source_ids)) for record in records]
    src_ids = torch.tensor(rows, dtype=torch.long)
    return src_ids, src_ids.eq(SRC_PAD_ID)


def repeat_prompts(src_ids: torch.Tensor, src_padding_mask: torch.Tensor, repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
    if repeats <= 1:
        return src_ids, src_padding_mask
    return (
        src_ids.repeat_interleave(repeats, dim=0),
        src_padding_mask.repeat_interleave(repeats, dim=0),
    )


def decode_cds_batch(token_ids: torch.Tensor) -> list[str]:
    return [decode_cds(row.tolist()) for row in token_ids.cpu()]


def build_tgt_inputs(sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = sequences.contiguous()
    bos = torch.full((labels.size(0), 1), fill_value=TGT_BOS_ID, dtype=labels.dtype, device=labels.device)
    tgt_input_ids = torch.cat([bos, labels[:, :-1]], dim=1)
    tgt_padding_mask = tgt_input_ids.eq(TGT_PAD_ID)
    token_mask = labels.ne(TGT_PAD_ID)
    return tgt_input_ids, tgt_padding_mask, token_mask


def sequence_logprob_stats(
    model: StandardSeq2SeqTransformer,
    *,
    src_ids: torch.Tensor,
    src_padding_mask: torch.Tensor,
    sequences: torch.Tensor,
    amp_dtype: torch.dtype | None,
    mini_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_type = src_ids.device.type
    autocast_enabled = device_type == "cuda" and amp_dtype is not None and amp_dtype != torch.float32
    active_dtype = amp_dtype if amp_dtype is not None else torch.float16
    seq_logps: list[torch.Tensor] = []
    seq_entropies: list[torch.Tensor] = []

    for start in range(0, sequences.size(0), mini_batch_size):
        end = min(start + mini_batch_size, sequences.size(0))
        seq_chunk = sequences[start:end]
        src_chunk = src_ids[start:end]
        src_pad_chunk = src_padding_mask[start:end]
        tgt_input_ids, tgt_padding_mask, token_mask = build_tgt_inputs(seq_chunk)
        with torch.autocast(device_type=device_type, dtype=active_dtype, enabled=autocast_enabled):
            logits = model(
                src_chunk,
                tgt_input_ids,
                src_padding_mask=src_pad_chunk,
                tgt_padding_mask=tgt_padding_mask,
            ).float()
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=seq_chunk.unsqueeze(-1)).squeeze(-1)
            token_mask_f = token_mask.float()
            seq_len = token_mask_f.sum(dim=-1).clamp(min=1.0)
            seq_logp = (token_log_probs * token_mask_f).sum(dim=-1) / seq_len
            probs = log_probs.exp()
            token_entropy = -(probs * log_probs).sum(dim=-1)
            seq_entropy = (token_entropy * token_mask_f).sum(dim=-1) / seq_len
        seq_logps.append(seq_logp)
        seq_entropies.append(seq_entropy)

    return torch.cat(seq_logps, dim=0), torch.cat(seq_entropies, dim=0)


def sample_sequences(
    model: StandardSeq2SeqTransformer,
    *,
    src_ids: torch.Tensor,
    src_padding_mask: torch.Tensor,
    group_size: int,
    temperature: float,
    top_k: int,
    sample_seed: int,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    repeated_src_ids, repeated_src_padding_mask = repeat_prompts(src_ids, src_padding_mask, group_size)
    device_type = repeated_src_ids.device.type
    autocast_enabled = device_type == "cuda" and amp_dtype is not None and amp_dtype != torch.float32
    active_dtype = amp_dtype if amp_dtype is not None else torch.float16
    with torch.random.fork_rng(devices=[repeated_src_ids.device.index] if repeated_src_ids.is_cuda else []):
        torch.manual_seed(int(sample_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sample_seed))
        with torch.autocast(device_type=device_type, dtype=active_dtype, enabled=autocast_enabled):
            return model.generate(
                repeated_src_ids,
                src_padding_mask=repeated_src_padding_mask,
                sample=True,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
            )


def group_advantages(reward_values: np.ndarray) -> np.ndarray:
    reward_values = np.asarray(reward_values, dtype=np.float32)
    mean = reward_values.mean(axis=1, keepdims=True)
    std = reward_values.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (reward_values - mean) / std


@torch.no_grad()
def evaluate_policy(
    model: StandardSeq2SeqTransformer,
    reward_model: HandcraftedRewardModel,
    records: list[SequenceRecord],
    *,
    prompt_batch_size: int,
    group_size: int,
    temperature: float,
    top_k: int,
    amp_dtype: torch.dtype | None,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    if not records:
        return {
            "sample_reward_mean": 0.0,
            "sample_reward_best_mean": 0.0,
        }
    reward_rows: list[np.ndarray] = []
    for chunk_start in range(0, len(records), prompt_batch_size):
        chunk = records[chunk_start : chunk_start + prompt_batch_size]
        src_ids, src_padding_mask = pad_source_batch(chunk)
        src_ids = src_ids.to(device)
        src_padding_mask = src_padding_mask.to(device)
        generated_ids = sample_sequences(
            model,
            src_ids=src_ids,
            src_padding_mask=src_padding_mask,
            group_size=group_size,
            temperature=temperature,
            top_k=top_k,
            sample_seed=seed + chunk_start,
            amp_dtype=amp_dtype,
        )
        cds_batch = decode_cds_batch(generated_ids)
        reward_rows.append(reward_model.predict_reward(cds_batch).reshape(len(chunk), group_size))
    reward_matrix = np.concatenate(reward_rows, axis=0)
    return {
        "sample_reward_mean": float(np.mean(reward_matrix)),
        "sample_reward_best_mean": float(np.mean(reward_matrix.max(axis=1))),
    }


def save_trial_summary(trial_dir: Path, payload: dict[str, Any]) -> Path:
    summary_path = trial_dir / "summary.json"
    save_json(summary_path, payload)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one fixed RL autoresearch trial spec.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--trial-name", default="preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = load_trial_context(args.spec, args.trial_name)
    preview = {
        "objective_name": context.spec.best_metric_key,
        "dataset_meta": context.dataset_meta,
        "split_sizes": context.split_sizes,
        "trial_dir": str(context.trial_dir),
        "time_budget_seconds": TIME_BUDGET,
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
