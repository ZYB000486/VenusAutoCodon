from __future__ import annotations

import math
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .vocab import AA_TO_CODONS, CODON_TO_AA, CODON_TOKENS, split_codons


RNAFOLD_CHUNK_SIZE = 512
EVOLUTIONARY_METRIC_NAMES = (
    "prefix_cai_top10",
    "prefix_cpai_top10",
    "prefix_mfe_per_nt",
)


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


def _normalize_counts(counts: dict[str, float], allowed: list[str], pseudocount: float = 1.0) -> dict[str, float]:
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
        aa: _normalize_counts(counts, AA_TO_CODONS[aa])
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


def prefix_cpai_score(cds: str, reference: ReferenceStats) -> float:
    codons = split_codons(cds)[: reference.prefix_window]
    if len(codons) < 2:
        return 0.0
    return cpai_like_score("".join(codons), reference)


def prefix_gc3_match_score(cds: str, reference: ReferenceStats) -> float:
    prefix_gc3 = prefix_gc3_fraction(cds, reference.prefix_window)
    return -abs(prefix_gc3 - reference.prefix_gc3_mean)


def _parse_rnafold_output(text: str) -> list[float]:
    values: list[float] = []
    pattern = re.compile(r"\(\s*([+-]?\d+(?:\.\d+)?)\)")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
    return values


def _run_rnafold(prefixes_rna: list[str]) -> np.ndarray:
    if not prefixes_rna:
        return np.zeros(0, dtype=np.float64)
    try:
        probe = subprocess.run(
            ["RNAfold", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        _ = probe
    except FileNotFoundError as exc:
        raise RuntimeError("RNAfold is required for autoresearch evolutionary MFE scoring, but it was not found in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"RNAfold is installed but `RNAfold --version` failed: {stderr or exc}") from exc

    mfes: list[float] = []
    for start in range(0, len(prefixes_rna), RNAFOLD_CHUNK_SIZE):
        chunk = prefixes_rna[start : start + RNAFOLD_CHUNK_SIZE]
        try:
            proc = subprocess.run(
                ["RNAfold", "--noPS"],
                input=("\n".join(chunk) + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("RNAfold disappeared from PATH during execution.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"RNAfold execution failed: {stderr or exc}") from exc
        chunk_mfes = _parse_rnafold_output(proc.stdout.decode("utf-8"))
        if len(chunk_mfes) != len(chunk):
            raise RuntimeError("RNAfold output parse mismatch.")
        mfes.extend(chunk_mfes)

    return np.asarray(mfes, dtype=np.float64)


def prefix_mfe_per_nt_scores(cds_list: list[str], reference: ReferenceStats) -> np.ndarray:
    prefix_nt = max(1, int(reference.prefix_window)) * 3
    prefixes_rna = [
        str(cds[: min(prefix_nt, len(cds))]).upper().replace("T", "U")
        for cds in cds_list
    ]
    raw = _run_rnafold(prefixes_rna)
    lengths = np.asarray([max(1, len(seq)) for seq in prefixes_rna], dtype=np.float64)
    return raw / lengths


def prefix_mfe_per_nt_score(cds: str, reference: ReferenceStats) -> float:
    return float(prefix_mfe_per_nt_scores([cds], reference)[0])


def _zscore(value: float, mean: float, std: float) -> float:
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
        "prefix_cai_top10",
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
            _zscore(gc, reference.gc_mean, reference.gc_std),
            _zscore(gc3, reference.gc3_mean, reference.gc3_std),
            _zscore(float(len(codons)), reference.length_mean, reference.length_std),
        ]
        row.extend(float(codon_counts[codon]) / float(codon_total) for codon in CODON_TOKENS)
        row.extend(float(prefix_counts[codon]) / float(prefix_total) for codon in CODON_TOKENS)
        row.extend(float(pair_counts[pair]) / float(max(1, len(codons) - 1)) for pair in reference.top_codon_pairs)
        feature_rows.append(row)

    return np.asarray(feature_rows, dtype=np.float64), feature_names


def minmax_normalize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    mins = matrix.min(axis=0, keepdims=True)
    maxs = matrix.max(axis=0, keepdims=True)
    denom = np.where((maxs - mins) < 1e-8, 1.0, maxs - mins)
    return (matrix - mins) / denom


def evolutionary_metric_matrix(cds_list: list[str], reference: ReferenceStats) -> tuple[np.ndarray, list[str]]:
    prefix_mfe = prefix_mfe_per_nt_scores(cds_list, reference)
    rows = np.column_stack(
        [
            np.asarray([prefix_cai_score(cds, reference) for cds in cds_list], dtype=np.float64),
            np.asarray([prefix_cpai_score(cds, reference) for cds in cds_list], dtype=np.float64),
            prefix_mfe,
        ]
    )
    return rows, list(EVOLUTIONARY_METRIC_NAMES)
