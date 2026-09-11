from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIME_BUDGET = 300
OBJECTIVE_NAME = "validation_spearman_rho"

REQUIRED_BASE_COLUMNS = ("gene_name", "cds")
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
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()
    denom = math.sqrt(float((y_true_centered**2).sum()) * float((y_pred_centered**2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((y_true_centered * y_pred_centered).sum() / denom)


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_rank = pd.Series(np.asarray(y_true)).rank(method="average").to_numpy(dtype=np.float64)
    pred_rank = pd.Series(np.asarray(y_pred)).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_r(true_rank, pred_rank)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_rho": spearman_rho(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def resolve_dataset(species: str | None = None, dataset: str | Path | None = None) -> Path:
    if dataset is not None:
        return Path(dataset).expanduser().resolve()
    if species is None:
        raise ValueError("Either species or dataset must be provided")
    candidate = DEFAULT_DATA_DIR / species / f"{species}.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset not found for species={species!r}: {candidate}")
    return candidate.resolve()


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


def load_dataset_frame(species: str | None = None, dataset: str | Path | None = None) -> tuple[Path, pd.DataFrame]:
    path = resolve_dataset(species=species, dataset=dataset)
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_BASE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if "abundance" not in frame.columns:
        raise ValueError(f"{path} does not provide abundance labels")

    cleaned = frame[list(REQUIRED_BASE_COLUMNS) + ["abundance"]].copy()
    cleaned["gene_name"] = cleaned["gene_name"].astype(str).str.strip()
    cleaned["cds"] = cleaned["cds"].astype(str).str.strip().str.upper().str.replace("U", "T", regex=False)
    cleaned["abundance"] = pd.to_numeric(cleaned["abundance"], errors="coerce")
    cleaned = cleaned.dropna(subset=["abundance"])
    cleaned = cleaned[cleaned["gene_name"] != ""]
    cleaned = cleaned[cleaned["cds"] != ""]
    cleaned = cleaned[cleaned["cds"].map(lambda text: len(text) % 3 == 0)]
    cleaned = cleaned[cleaned["cds"].map(lambda text: set(text).issubset({"A", "C", "G", "T"}))]
    return path, cleaned.reset_index(drop=True)


@dataclass(frozen=True)
class SequenceRecord:
    gene_name: str
    cds: str
    abundance: float


def load_records(
    species: str | None = None,
    dataset: str | Path | None = None,
    *,
    max_aa_len: int | None = None,
) -> tuple[dict[str, Any], list[SequenceRecord]]:
    dataset_path, frame = load_dataset_frame(species=species, dataset=dataset)

    records: list[SequenceRecord] = []
    skipped_invalid = 0
    skipped_too_long = 0

    for row in frame.itertuples(index=False):
        gene_name = str(row.gene_name)
        cds = str(row.cds)
        try:
            aa_len = len(translate_cds(cds))
        except Exception:
            skipped_invalid += 1
            continue
        if aa_len == 0:
            skipped_too_long += 1
            continue
        if max_aa_len is not None and aa_len > max_aa_len:
            skipped_too_long += 1
            continue
        records.append(SequenceRecord(gene_name=gene_name, cds=cds, abundance=float(row.abundance)))

    aa_lengths = np.asarray([len(record.cds) // 3 for record in records], dtype=np.int64) if records else np.asarray([], dtype=np.int64)
    abundance_values = np.asarray([record.abundance for record in records], dtype=np.float64) if records else np.asarray([], dtype=np.float64)
    meta = {
        "dataset_name": dataset_path.parent.name,
        "dataset_path": str(dataset_path),
        "has_abundance": True,
        "raw_rows": int(len(frame)),
        "kept_rows": int(len(records)),
        "skipped_invalid_rows": int(skipped_invalid),
        "skipped_too_long_rows": int(skipped_too_long),
        "max_aa_len": None if max_aa_len is None else int(max_aa_len),
        "length_filter_applied": bool(max_aa_len is not None),
        "aa_length": {
            "mean": float(aa_lengths.mean()) if aa_lengths.size else 0.0,
            "median": float(np.median(aa_lengths)) if aa_lengths.size else 0.0,
            "p95": int(np.quantile(aa_lengths, 0.95)) if aa_lengths.size else 0,
            "max": int(aa_lengths.max()) if aa_lengths.size else 0,
        },
        "abundance": {
            "mean": float(abundance_values.mean()) if abundance_values.size else 0.0,
            "std": float(abundance_values.std()) if abundance_values.size else 0.0,
            "min": float(abundance_values.min()) if abundance_values.size else 0.0,
            "max": float(abundance_values.max()) if abundance_values.size else 0.0,
        },
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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


def evaluate_bundle(bundle_path: Path, records: list[SequenceRecord]) -> dict[str, float]:
    reward_model = HandcraftedRewardModel.from_path(bundle_path)
    y_true = np.asarray([record.abundance for record in records], dtype=np.float64)
    y_pred = reward_model.predict_abundance([record.cds for record in records]).astype(np.float64)
    return regression_metrics(y_true, y_pred)


@dataclass
class RewardTrialSpec:
    species: str | None = None
    dataset: str | None = None
    validation_dataset: str | None = None
    output_dir: str = ""
    seed: int = 42
    max_aa_len: int | None = None
    reward_alpha: float = 1.0
    top_pair_k: int = 64
    max_train_records: int | None = None
    max_validation_records: int | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "RewardTrialSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass
class RewardTrialContext:
    spec: RewardTrialSpec
    dataset_meta: dict[str, Any]
    split_sizes: dict[str, int]
    trial_dir: Path
    train_records: list[SequenceRecord]
    validation_records: list[SequenceRecord]
    reference: ReferenceStats
    feature_names: list[str]
    x_train: np.ndarray
    x_validation: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray


def load_trial_context(spec_path: str | Path, trial_name: str) -> RewardTrialContext:
    spec = RewardTrialSpec.from_path(spec_path)
    if not spec.output_dir:
        raise ValueError("RewardTrialSpec.output_dir is required")
    if not spec.dataset:
        raise ValueError("RewardTrialSpec.dataset is required and must point to the fixed training split")
    if not spec.validation_dataset:
        raise ValueError("RewardTrialSpec.validation_dataset is required and must point to the fixed validation split")

    dataset_meta, train_records = load_records(
        species=spec.species,
        dataset=spec.dataset,
        max_aa_len=spec.max_aa_len,
    )
    validation_meta, validation_records = load_records(
        species=spec.species,
        dataset=spec.validation_dataset,
        max_aa_len=spec.max_aa_len,
    )
    if spec.max_train_records is not None:
        train_records = train_records[: spec.max_train_records]
    if spec.max_validation_records is not None:
        validation_records = validation_records[: spec.max_validation_records]
    if not train_records or not validation_records:
        raise ValueError("Reward autoresearch requires non-empty fixed train and validation splits")
    dataset_meta["validation_dataset_path"] = validation_meta["dataset_path"]
    dataset_meta["validation_kept_rows"] = validation_meta["kept_rows"]
    dataset_meta["validation_raw_rows"] = validation_meta["raw_rows"]

    stage_dir = Path(spec.output_dir).expanduser().resolve()
    trial_dir = stage_dir / "trial_runs" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    reference = compute_reference_stats([record.cds for record in train_records], top_pair_k=spec.top_pair_k)
    x_train, feature_names = build_feature_matrix([record.cds for record in train_records], reference)
    x_validation, _ = build_feature_matrix([record.cds for record in validation_records], reference)
    y_train = np.asarray([record.abundance for record in train_records], dtype=np.float64)
    y_validation = np.asarray([record.abundance for record in validation_records], dtype=np.float64)

    return RewardTrialContext(
        spec=spec,
        dataset_meta=dataset_meta,
        split_sizes={"train": len(train_records), "validation": len(validation_records)},
        trial_dir=trial_dir,
        train_records=train_records,
        validation_records=validation_records,
        reference=reference,
        feature_names=feature_names,
        x_train=x_train,
        x_validation=x_validation,
        y_train=y_train,
        y_validation=y_validation,
    )


def save_reward_bundle(bundle: RewardBundle, bundle_path: Path) -> None:
    save_json(bundle_path, bundle.to_dict())


def save_trial_summary(trial_dir: Path, payload: dict[str, Any]) -> Path:
    summary_path = trial_dir / "summary.json"
    save_json(summary_path, payload)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one fixed RM autoresearch trial spec.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--trial-name", default="preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = load_trial_context(args.spec, args.trial_name)
    preview = {
        "objective_name": OBJECTIVE_NAME,
        "dataset_meta": context.dataset_meta,
        "split_sizes": context.split_sizes,
        "trial_dir": str(context.trial_dir),
        "feature_count": int(len(context.feature_names)),
        "time_budget_seconds": TIME_BUDGET,
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
