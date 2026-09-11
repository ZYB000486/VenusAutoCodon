from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autoresearch.common import RUNS_DIR, default_run_name, save_json
    from autoresearch.dataset import load_records
    from autoresearch.features import (
        EVOLUTIONARY_METRIC_NAMES,
        ReferenceStats,
        compute_reference_stats,
        evolutionary_metric_matrix,
        minmax_normalize_columns,
    )
    from autoresearch.vocab import AA_TO_CODONS
else:
    from .common import RUNS_DIR, default_run_name, save_json
    from .dataset import load_records
    from .features import (
        EVOLUTIONARY_METRIC_NAMES,
        ReferenceStats,
        compute_reference_stats,
        evolutionary_metric_matrix,
        minmax_normalize_columns,
    )
    from .vocab import AA_TO_CODONS


@dataclass
class EvolutionConfig:
    species: str | None = None
    dataset: str | None = None
    run_name: str = ""
    output_root: str = str(RUNS_DIR)
    seed: int = 42
    max_aa_len: int = 512
    population_size: int = 32
    generations: int = 20
    elite_fraction: float = 0.25
    mutation_rate: float = 0.12
    mutations_per_child: int = 2
    subset_size: int | None = 128
    head_codons: int = 10


def parse_args() -> EvolutionConfig:
    parser = argparse.ArgumentParser(description="Optimize synonymous CDS with a head-focused NSGA-II search.")
    parser.add_argument("--species", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default=str(RUNS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-aa-len", type=int, default=512)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--mutation-rate", type=float, default=0.12)
    parser.add_argument("--mutations-per-child", type=int, default=2)
    parser.add_argument("--subset-size", type=int, default=128)
    parser.add_argument("--head-codons", type=int, default=10)
    return EvolutionConfig(**vars(parser.parse_args()))


def _random_synonymous_cds(aas: str, rng: np.random.Generator) -> str:
    codons = [rng.choice(AA_TO_CODONS[aa]) for aa in aas]
    return "".join(codons)


def _mutate_synonymous(
    cds: str,
    aas: str,
    rng: np.random.Generator,
    mutation_rate: float,
    mutations_per_child: int,
    mutable_prefix_codons: int | None = None,
) -> str:
    codons = [cds[idx : idx + 3] for idx in range(0, len(cds), 3)]
    mutable_positions = [idx for idx, aa in enumerate(aas) if len(AA_TO_CODONS[aa]) > 1]
    if mutable_prefix_codons is not None:
        mutable_positions = [idx for idx in mutable_positions if idx < int(mutable_prefix_codons)]
    if not mutable_positions:
        return cds
    num_mutations = min(len(mutable_positions), max(1, mutations_per_child))
    for idx in rng.choice(mutable_positions, size=num_mutations, replace=False):
        if rng.random() > mutation_rate:
            continue
        aa = aas[int(idx)]
        alternatives = [codon for codon in AA_TO_CODONS[aa] if codon != codons[int(idx)]]
        if alternatives:
            codons[int(idx)] = str(rng.choice(alternatives))
    return "".join(codons)


def _dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left >= right - 1e-12) and np.any(left > right + 1e-12))


def _fast_non_dominated_sort(objectives: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    n_items = int(objectives.shape[0])
    dominates: list[list[int]] = [[] for _ in range(n_items)]
    dominated_counts = np.zeros(n_items, dtype=np.int64)
    ranks = np.full(n_items, -1, dtype=np.int64)
    fronts: list[list[int]] = [[]]

    for i in range(n_items):
        for j in range(i + 1, n_items):
            if _dominates(objectives[i], objectives[j]):
                dominates[i].append(j)
                dominated_counts[j] += 1
            elif _dominates(objectives[j], objectives[i]):
                dominates[j].append(i)
                dominated_counts[i] += 1
        if dominated_counts[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[int] = []
        for idx in fronts[front_index]:
            for dominated_idx in dominates[idx]:
                dominated_counts[dominated_idx] -= 1
                if dominated_counts[dominated_idx] == 0:
                    ranks[dominated_idx] = front_index + 1
                    next_front.append(dominated_idx)
        if next_front:
            fronts.append(next_front)
        front_index += 1

    return fronts, ranks


def _crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    distances = np.zeros(len(front), dtype=np.float64)
    if len(front) <= 2:
        distances[:] = np.inf
        return distances

    front_objectives = objectives[np.asarray(front, dtype=np.int64)]
    n_objectives = int(front_objectives.shape[1])
    for obj_idx in range(n_objectives):
        order = np.argsort(front_objectives[:, obj_idx])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        min_value = float(front_objectives[order[0], obj_idx])
        max_value = float(front_objectives[order[-1], obj_idx])
        width = max_value - min_value
        if width <= 1e-12:
            continue
        for pos in range(1, len(front) - 1):
            if np.isinf(distances[order[pos]]):
                continue
            next_value = float(front_objectives[order[pos + 1], obj_idx])
            prev_value = float(front_objectives[order[pos - 1], obj_idx])
            distances[order[pos]] += (next_value - prev_value) / width
    return distances


def _evaluate_population(
    population: list[str],
    reference: ReferenceStats,
    metric_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[list[int]], np.ndarray, np.ndarray]:
    missing = [seq for seq in dict.fromkeys(population) if seq not in metric_cache]
    if missing:
        metric_matrix, _ = evolutionary_metric_matrix(missing, reference)
        for seq, row in zip(missing, metric_matrix):
            metric_cache[seq] = row.astype(np.float64, copy=True)

    raw_metrics = np.vstack([metric_cache[seq] for seq in population]).astype(np.float64, copy=False)
    fronts, ranks = _fast_non_dominated_sort(raw_metrics)
    crowding = np.zeros(len(population), dtype=np.float64)
    for front in fronts:
        if not front:
            continue
        crowding_values = _crowding_distance(raw_metrics, front)
        for idx, distance in zip(front, crowding_values):
            crowding[idx] = float(distance)
    scores = minmax_normalize_columns(raw_metrics).sum(axis=1)
    return scores, raw_metrics, fronts, ranks, crowding


def _representative_index(
    scores: np.ndarray,
    fronts: list[list[int]],
    crowding: np.ndarray,
) -> int:
    first_front = fronts[0] if fronts else list(range(len(scores)))
    return max(
        first_front,
        key=lambda idx: (float(scores[idx]), float(crowding[idx])),
    )


def _binary_tournament(
    population_size: int,
    ranks: np.ndarray,
    crowding: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
) -> int:
    left = int(rng.integers(0, population_size))
    right = int(rng.integers(0, population_size))
    if int(ranks[left]) < int(ranks[right]):
        return left
    if int(ranks[left]) > int(ranks[right]):
        return right
    if float(crowding[left]) > float(crowding[right]):
        return left
    if float(crowding[left]) < float(crowding[right]):
        return right
    if float(scores[left]) >= float(scores[right]):
        return left
    return right


def _select_next_population(
    candidates: list[str],
    *,
    population_size: int,
    reference: ReferenceStats,
    metric_cache: dict[str, np.ndarray],
) -> list[str]:
    scores, raw_metrics, fronts, _, crowding = _evaluate_population(candidates, reference, metric_cache)
    selected: list[str] = []
    for front in fronts:
        if not front:
            continue
        if len(selected) + len(front) <= population_size:
            selected.extend(candidates[idx] for idx in front)
            continue
        ordered_front = sorted(
            front,
            key=lambda idx: (float(crowding[idx]), float(scores[idx])),
            reverse=True,
        )
        remaining = population_size - len(selected)
        selected.extend(candidates[idx] for idx in ordered_front[:remaining])
        break
    return selected


def optimize_prefix_for_aas(
    aas: str,
    *,
    reference: ReferenceStats,
    population_size: int,
    generations: int,
    elite_fraction: float,
    mutation_rate: float,
    mutations_per_child: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    head_aas = aas[: reference.prefix_window]
    population = [_random_synonymous_cds(head_aas, rng) for _ in range(population_size)]
    metric_cache: dict[str, np.ndarray] = {}
    best_prefix_cds = population[0]
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    metric_names = list(EVOLUTIONARY_METRIC_NAMES)

    for _ in range(generations):
        scores, raw_metrics, fronts, ranks, crowding = _evaluate_population(population, reference, metric_cache)
        rep_idx = _representative_index(scores, fronts, crowding)
        if float(scores[rep_idx]) > best_score:
            best_score = float(scores[rep_idx])
            best_prefix_cds = population[rep_idx]
            best_metrics = {
                metric_names[idx]: float(raw_metrics[rep_idx, idx])
                for idx in range(len(metric_names))
            }

        offspring: list[str] = []
        while len(offspring) < population_size:
            parent_idx = _binary_tournament(len(population), ranks, crowding, scores, rng)
            parent = population[parent_idx]
            offspring.append(
                _mutate_synonymous(
                    parent,
                    head_aas,
                    rng,
                    mutation_rate=mutation_rate,
                    mutations_per_child=mutations_per_child,
                )
            )
        population = _select_next_population(
            population + offspring,
            population_size=population_size,
            reference=reference,
            metric_cache=metric_cache,
        )

    scores, raw_metrics, fronts, _, crowding = _evaluate_population(population, reference, metric_cache)
    rep_idx = _representative_index(scores, fronts, crowding)
    if float(scores[rep_idx]) > best_score:
        best_score = float(scores[rep_idx])
        best_prefix_cds = population[rep_idx]
        best_metrics = {
            metric_names[idx]: float(raw_metrics[rep_idx, idx])
            for idx in range(len(metric_names))
        }

    return {
        "prefix_codons": int(reference.prefix_window),
        "best_fitness": float(best_score),
        "best_prefix_cds": best_prefix_cds,
        "metrics": best_metrics,
    }


def _optimize_one_gene(
    gene_name: str,
    aas: str,
    *,
    reference: ReferenceStats,
    population_size: int,
    generations: int,
    elite_fraction: float,
    mutation_rate: float,
    mutations_per_child: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    population = [_random_synonymous_cds(aas, rng) for _ in range(population_size)]
    metric_cache: dict[str, np.ndarray] = {}
    best_cds = population[0]
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    metric_names = list(EVOLUTIONARY_METRIC_NAMES)

    for _ in range(generations):
        scores, raw_metrics, fronts, ranks, crowding = _evaluate_population(population, reference, metric_cache)
        rep_idx = _representative_index(scores, fronts, crowding)
        if float(scores[rep_idx]) > best_score:
            best_score = float(scores[rep_idx])
            best_cds = population[rep_idx]
            best_metrics = {
                metric_names[idx]: float(raw_metrics[rep_idx, idx])
                for idx in range(len(metric_names))
            }

        offspring: list[str] = []
        while len(offspring) < population_size:
            parent_idx = _binary_tournament(len(population), ranks, crowding, scores, rng)
            parent = population[parent_idx]
            offspring.append(
                _mutate_synonymous(
                    parent,
                    aas,
                    rng,
                    mutation_rate=mutation_rate,
                    mutations_per_child=mutations_per_child,
                    mutable_prefix_codons=reference.prefix_window,
                )
            )
        population = _select_next_population(
            population + offspring,
            population_size=population_size,
            reference=reference,
            metric_cache=metric_cache,
        )

    scores, raw_metrics, fronts, _, crowding = _evaluate_population(population, reference, metric_cache)
    rep_idx = _representative_index(scores, fronts, crowding)
    if float(scores[rep_idx]) > best_score:
        best_score = float(scores[rep_idx])
        best_cds = population[rep_idx]
        best_metrics = {
            metric_names[idx]: float(raw_metrics[rep_idx, idx])
            for idx in range(len(metric_names))
        }

    prefix_summary = optimize_prefix_for_aas(
        aas,
        reference=reference,
        population_size=population_size,
        generations=generations,
        elite_fraction=elite_fraction,
        mutation_rate=mutation_rate,
        mutations_per_child=mutations_per_child,
        rng=rng,
    )
    return {
        "gene_name": gene_name,
        "aa_length": int(len(aas)),
        "best_fitness": float(best_score),
        "best_cds": best_cds,
        "metrics": best_metrics,
        "best_prefix_cds": str(prefix_summary["best_prefix_cds"]),
        "prefix_metrics": dict(prefix_summary["metrics"]),
        "prefix_codons": int(prefix_summary["prefix_codons"]),
    }


def run_evolutionary_search(config: EvolutionConfig) -> dict[str, Any]:
    dataset_meta, records = load_records(
        species=config.species,
        dataset=config.dataset,
        max_aa_len=config.max_aa_len,
        require_abundance=False,
    )
    rng = np.random.default_rng(config.seed)
    if config.subset_size is None or config.subset_size >= len(records):
        selected_records = records
    else:
        order = rng.permutation(len(records))
        selected_records = [records[int(idx)] for idx in order[: config.subset_size]]
    reference = compute_reference_stats([record.cds for record in records], top_pair_k=64, prefix_window=config.head_codons)

    output_root = Path(config.output_root).expanduser().resolve()
    run_name = config.run_name or default_run_name(f"{dataset_meta['dataset_name']}_evo")
    evo_dir = output_root / dataset_meta["dataset_name"] / run_name / "evolution"
    evo_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _optimize_one_gene(
            record.gene_name,
            record.aas,
            reference=reference,
            population_size=config.population_size,
            generations=config.generations,
            elite_fraction=config.elite_fraction,
            mutation_rate=config.mutation_rate,
            mutations_per_child=config.mutations_per_child,
            rng=rng,
        )
        for record in selected_records
    ]

    fitness = np.asarray([row["best_fitness"] for row in results], dtype=np.float64)
    summary = {
        "dataset_name": dataset_meta["dataset_name"],
        "dataset_path": dataset_meta["dataset_path"],
        "run_name": run_name,
        "evolution_dir": str(evo_dir),
        "dataset_meta": dataset_meta,
        "config": asdict(config),
        "optimized_genes": int(len(results)),
        "fitness_mean": float(fitness.mean()) if fitness.size else 0.0,
        "fitness_std": float(fitness.std()) if fitness.size else 0.0,
        "top_examples": sorted(results, key=lambda row: row["best_fitness"], reverse=True)[:10],
    }
    save_json(evo_dir / "summary.json", summary)
    save_json(evo_dir / "reference_stats.json", reference.to_dict())
    save_json(evo_dir / "results.json", {"results": results})
    return summary


def main() -> None:
    summary = run_evolutionary_search(parse_args())
    print(f"evolution_dir: {summary['evolution_dir']}")
    print(f"optimized_genes: {summary['optimized_genes']}")
    print(f"fitness_mean: {summary['fitness_mean']:.4f}")


if __name__ == "__main__":
    main()
