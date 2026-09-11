from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import pandas as pd
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autoresearch.common import RUNS_DIR, default_run_name, save_json
    from autoresearch.dataset import discover_species_datasets
    from autoresearch.dataset_agent import (
        BRANCH_1_PAXDB_RM_RL,
        BRANCH_2_TRANSCRIPTOME_RM_RL,
        BRANCH_3_CDS_ONLY_EA,
        DatasetAgentRequest,
        run_dataset_agent,
        slugify,
    )
    from autoresearch.evolution import EvolutionConfig, run_evolutionary_search
    from autoresearch.karpathy_worker import launch_karpathy_stage
    from autoresearch.run_policy_init import PolicyInitSpec, run_policy_init
else:
    from .common import RUNS_DIR, default_run_name, save_json
    from .dataset import discover_species_datasets
    from .dataset_agent import (
        BRANCH_1_PAXDB_RM_RL,
        BRANCH_2_TRANSCRIPTOME_RM_RL,
        BRANCH_3_CDS_ONLY_EA,
        DatasetAgentRequest,
        run_dataset_agent,
        slugify,
    )
    from .evolution import EvolutionConfig, run_evolutionary_search
    from .karpathy_worker import launch_karpathy_stage
    from .run_policy_init import PolicyInitSpec, run_policy_init


PACKAGE_ROOT = Path(__file__).resolve().parent
TRAIN_FRACTION = 0.85
RL_UNIREF_TRAIN_PROMPT_FASTA = PACKAGE_ROOT / "assets" / "prompt_library" / "uniref10_fixed_train.fasta"
RL_UNIREF_VALIDATION_PROMPT_FASTA = PACKAGE_ROOT / "assets" / "prompt_library" / "uniref10_fixed_validation.fasta"
RL_MAX_AA_LEN = 512
RL_TRAIN_PROMPT_BATCH_SIZE = 72
RL_GROUP_SIZE = 4
RL_POLICY_MINI_BATCH_SIZE = 64
RL_VALIDATION_PROMPT_BATCH_SIZE = 300
RL_VALIDATION_GROUP_SIZE = 4
VALID_AA_TOKENS = set("ACDEFGHIKLMNPQRSTVWY*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end standalone CDS optimization pipeline: species -> dataset agent -> branch -> training -> inference script.")
    parser.add_argument("--species", nargs="+", default=["saccharomyces_cerevisiae"])
    parser.add_argument("--mode", choices=["auto", "supervised", "rl", "evolution"], default="auto")
    parser.add_argument("--engine", choices=["karpathy"], default="karpathy", help=argparse.SUPPRESS)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default=str(RUNS_DIR))
    parser.add_argument("--max-aa-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--supervised-lr", type=float, default=3e-4)
    parser.add_argument("--reward-alpha", type=float, default=1.0)
    parser.add_argument("--rl-steps", type=int, default=120)
    parser.add_argument("--rl-lr", type=float, default=1e-4)
    parser.add_argument("--rl-best-metric-key", choices=["sample_reward_mean", "sample_reward_best_mean"], default="sample_reward_mean")
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--subset-size", type=int, default=128)
    parser.add_argument("--rm-stage-seconds", type=int, default=28800)
    parser.add_argument("--rl-stage-seconds", type=int, default=14400)
    parser.add_argument("--codex-model", default="")
    parser.add_argument("--worker-base-ref", default="HEAD")
    parser.add_argument("--cleanup-worktrees", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--max-validation-records", type=int, default=None)
    parser.add_argument("--dataset-agent-prompt", default="")
    parser.add_argument("--dataset-agent-prompt-file", default="")
    parser.add_argument("--dataset-agent-seconds", type=int, default=1800)
    parser.add_argument("--rl-train-prompts", default="", help="Existing training protein FASTA; pair with --rl-validation-prompts")
    parser.add_argument("--rl-validation-prompts", default="", help="Existing validation protein FASTA")
    return parser.parse_args()


def resolve_species_list(raw_species: list[str]) -> list[str]:
    if len(raw_species) == 1 and raw_species[0] == "all":
        return sorted(discover_species_datasets())
    return raw_species


def read_dataset_agent_extra_instructions(args: argparse.Namespace) -> str:
    prompt = args.dataset_agent_prompt.strip()
    if args.dataset_agent_prompt_file:
        prompt = Path(args.dataset_agent_prompt_file).read_text(encoding="utf-8").strip()
    return prompt


def collect_dataset_for_species(
    *,
    species: str,
    args: argparse.Namespace,
    species_root: Path,
) -> dict[str, Any]:
    stage_dir = species_root / "dataset_agent"
    request = DatasetAgentRequest(
        species=species,
        stage_dir=str(stage_dir),
        codex_model=args.codex_model,
        timeout_seconds=args.dataset_agent_seconds,
        extra_instructions=read_dataset_agent_extra_instructions(args),
    )
    result = run_dataset_agent(request)
    payload = json.loads(Path(result.branch_json).read_text(encoding="utf-8"))
    return {
        "request": result.request,
        "dataset_csv": result.dataset_csv,
        "branch_json": result.branch_json,
        "requested_species": result.requested_species,
        "canonical_species_name": result.canonical_species_name,
        "canonical_species_slug": result.canonical_species_slug,
        "branch_name": result.branch_name,
        "label_source": result.label_source,
        "has_abundance": result.has_abundance,
        "dataset_schema": result.dataset_schema,
        "codex_returncode": result.codex_returncode,
        "timed_out": result.timed_out,
        "notes": str(payload.get("notes", "")),
    }


def determine_effective_mode(args: argparse.Namespace, *, branch_name: str, has_abundance: bool) -> str:
    if args.mode != "auto":
        return args.mode
    if branch_name in {BRANCH_1_PAXDB_RM_RL, BRANCH_2_TRANSCRIPTOME_RM_RL}:
        return "rl"
    if branch_name == BRANCH_3_CDS_ONLY_EA:
        return "evolution"
    return "rl" if has_abundance else "evolution"


def prepare_fixed_train_validation_split(
    *,
    dataset_path: str,
    species_root: Path,
    seed: int,
) -> dict[str, Any]:
    source_path = Path(dataset_path).expanduser().resolve()
    frame = pd.read_csv(source_path)
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = int(len(shuffled) * TRAIN_FRACTION)
    train_frame = shuffled.iloc[:train_end].reset_index(drop=True)
    validation_frame = shuffled.iloc[train_end:].reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise ValueError(f"Unable to create non-empty train/validation split from {source_path}")

    split_dir = species_root / "fixed_split"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train.csv"
    validation_csv = split_dir / "validation.csv"
    manifest_json = split_dir / "split_manifest.json"

    train_frame.to_csv(train_csv, index=False)
    validation_frame.to_csv(validation_csv, index=False)
    save_json(
        manifest_json,
        {
            "source_dataset": str(source_path),
            "seed": int(seed),
            "train_fraction": float(TRAIN_FRACTION),
            "train_csv": str(train_csv),
            "validation_csv": str(validation_csv),
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
        },
    )
    return {
        "train_csv": str(train_csv),
        "validation_csv": str(validation_csv),
        "manifest_json": str(manifest_json),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
    }


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


def normalize_uniref_prompt(seq: str) -> str:
    text = (seq or "").strip().upper()
    if not text:
        raise ValueError("Empty UniRef prompt sequence")
    invalid = sorted({token for token in text if token not in VALID_AA_TOKENS})
    if invalid:
        raise ValueError(f"Invalid UniRef prompt tokens: {''.join(invalid)}")
    return text


def count_fasta_records(path: Path, *, max_aa_len: int) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Fixed UniRef prompt split not found: {path}")
    total_entries = 0
    valid_entries = 0
    invalid_entries = 0
    too_long_entries = 0
    for _header, seq in iter_fasta_records(path):
        total_entries += 1
        try:
            aas = normalize_uniref_prompt(seq)
        except Exception:
            invalid_entries += 1
            continue
        if len(aas) > max_aa_len:
            too_long_entries += 1
            continue
        valid_entries += 1
    return {
        "total_entries": int(total_entries),
        "valid_entries": int(valid_entries),
        "invalid_entries": int(invalid_entries),
        "too_long_entries": int(too_long_entries),
    }


def prepare_fixed_uniref_prompt_split(
    *,
    species_root: Path,
    max_aa_len: int,
    train_prompt_path: str = "",
    validation_prompt_path: str = "",
) -> dict[str, Any]:
    if bool(train_prompt_path) != bool(validation_prompt_path):
        raise ValueError("Supply both --rl-train-prompts and --rl-validation-prompts")
    train_fasta = Path(train_prompt_path or RL_UNIREF_TRAIN_PROMPT_FASTA).expanduser().resolve()
    validation_fasta = Path(validation_prompt_path or RL_UNIREF_VALIDATION_PROMPT_FASTA).expanduser().resolve()
    if train_fasta == validation_fasta:
        raise ValueError("Training and validation prompt paths must differ")
    if not train_prompt_path and not train_fasta.exists() and not validation_fasta.exists():
        from autoresearch.uniref_prompts import download_prompts
        download_prompts(train_fasta, validation_fasta, max_aa_len=max_aa_len)
    train_meta = count_fasta_records(train_fasta, max_aa_len=max_aa_len)
    validation_meta = count_fasta_records(validation_fasta, max_aa_len=max_aa_len)
    if train_meta["valid_entries"] <= 0:
        raise ValueError(f"Fixed UniRef train split has no valid prompts after filtering: {train_fasta}")
    if validation_meta["valid_entries"] <= 0:
        raise ValueError(f"Fixed UniRef validation split has no valid prompts after filtering: {validation_fasta}")
    split_dir = species_root / "rl_prompt_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = split_dir / "prompt_split_manifest.json"
    save_json(
        manifest_json,
        {
            "source": "fixed_local_uniref_prompt_split",
            "max_aa_len": int(max_aa_len),
            "train_prompt_fasta": str(train_fasta),
            "validation_prompt_fasta": str(validation_fasta),
            "train_prompt_count": int(train_meta["valid_entries"]),
            "validation_prompt_count": int(validation_meta["valid_entries"]),
            "train_prompt_meta": train_meta,
            "validation_prompt_meta": validation_meta,
        },
    )
    return {
        "source": "fixed_local_uniref_prompt_split",
        "manifest_json": str(manifest_json),
        "train_prompt_fasta": str(train_fasta),
        "validation_prompt_fasta": str(validation_fasta),
        "train_prompt_count": int(train_meta["valid_entries"]),
        "validation_prompt_count": int(validation_meta["valid_entries"]),
    }


def run_policy_init_stage(
    *,
    species: str,
    dataset_path: str,
    species_root: Path,
    pipeline_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    policy_init_dir = species_root / "policy_init"
    policy_init_summary = run_policy_init(
        PolicyInitSpec(
            species=species,
            dataset=dataset_path,
            output_dir=str(policy_init_dir),
            run_name=f"{pipeline_name}__policy_init",
            seed=args.seed,
            max_aa_len=args.max_aa_len,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            device=args.device,
            dtype=args.dtype,
            lr=args.supervised_lr,
            max_train_records=args.max_train_records,
            max_test_records=args.max_test_records,
        )
    )
    policy_checkpoint = Path(str(policy_init_summary["best_checkpoint"])).expanduser().resolve()
    if not policy_checkpoint.exists():
        raise RuntimeError(f"Policy-init checkpoint missing for {species}: {policy_checkpoint}")
    return {
        "engine": "local_supervised",
        "summary_json": str(policy_init_dir / "policy_init_summary.json"),
        "best_checkpoint": str(policy_checkpoint),
        "best_test_token_accuracy": policy_init_summary["best_test"]["token_accuracy"],
        "test_token_accuracy": policy_init_summary["test_metrics"]["token_accuracy"],
    }


def write_inference_artifacts(
    *,
    species_root: Path,
    spec_payload: dict[str, Any],
) -> dict[str, str]:
    inference_dir = species_root / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    spec_path = inference_dir / "inference_spec.json"
    save_json(spec_path, spec_payload)

    script_path = inference_dir / "run_aas2cds_inference.sh"
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"python {shlex.quote(str(PACKAGE_ROOT / 'infer.py'))} --spec {shlex.quote(str(spec_path))} \"$@\"",
            "",
        ]
    )
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    return {
        "spec_json": str(spec_path),
        "run_script": str(script_path),
    }


def build_seq2seq_inference_spec(
    *,
    species: str,
    dataset_path: str,
    checkpoint_path: str,
    pipeline_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "method": "seq2seq",
        "species": species,
        "dataset": dataset_path,
        "checkpoint_path": checkpoint_path,
        "run_name": pipeline_name,
        "batch_size": args.eval_batch_size,
        "sample": False,
        "temperature": 1.0,
        "top_k": 0,
        "sample_seed": int(args.seed + 5000),
        "device": args.device,
        "dtype": args.dtype,
        "max_aa_len": args.max_aa_len,
        "head_codons": 10,
    }


def build_evolution_hybrid_inference_spec(
    *,
    species: str,
    dataset_path: str,
    checkpoint_path: str,
    pipeline_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "method": "evolution_hybrid",
        "species": species,
        "dataset": dataset_path,
        "checkpoint_path": checkpoint_path,
        "run_name": pipeline_name,
        "batch_size": args.eval_batch_size,
        "sample": False,
        "temperature": 1.0,
        "top_k": 0,
        "sample_seed": int(args.seed + 5000),
        "device": args.device,
        "dtype": args.dtype,
        "max_aa_len": args.max_aa_len,
        "head_codons": 10,
        "evo_population_size": args.population_size,
        "evo_generations": args.generations,
        "evo_elite_fraction": 0.25,
        "evo_mutation_rate": 0.12,
        "evo_mutations_per_child": 2,
    }


def run_pipeline_for_species(
    *,
    species: str,
    pipeline_name: str,
    args: argparse.Namespace,
    output_root: Path,
    species_root: Path,
) -> dict[str, Any]:
    dataset_info = collect_dataset_for_species(species=species, args=args, species_root=species_root)
    full_dataset_path = str(Path(dataset_info["dataset_csv"]).expanduser().resolve())
    split_info = prepare_fixed_train_validation_split(
        dataset_path=full_dataset_path,
        species_root=species_root,
        seed=args.seed,
    )
    dataset_path = str(Path(split_info["train_csv"]).expanduser().resolve())
    has_abundance = bool(dataset_info["has_abundance"])
    effective_mode = determine_effective_mode(args, branch_name=str(dataset_info["branch_name"]), has_abundance=has_abundance)

    species_summary: dict[str, Any] = {
        "species": species,
        "canonical_species_name": dataset_info["canonical_species_name"],
        "canonical_species_slug": dataset_info["canonical_species_slug"],
        "dataset_path": dataset_path,
        "full_dataset_path": full_dataset_path,
        "dataset_agent": dataset_info,
        "fixed_split": {
            "train_csv": split_info["train_csv"],
            "validation_csv": split_info["validation_csv"],
            "manifest_json": split_info["manifest_json"],
            "train_rows": split_info["train_rows"],
            "validation_rows": split_info["validation_rows"],
        },
        "has_abundance": has_abundance,
        "execution_model": "fixed_pipeline",
        "effective_mode": effective_mode,
    }

    policy_init = run_policy_init_stage(
        species=species,
        dataset_path=dataset_path,
        species_root=species_root,
        pipeline_name=pipeline_name,
        args=args,
    )
    species_summary["policy_init"] = policy_init

    if effective_mode == "supervised":
        species_summary["final_inference"] = write_inference_artifacts(
            species_root=species_root,
            spec_payload=build_seq2seq_inference_spec(
                species=species,
                dataset_path=dataset_path,
                checkpoint_path=str(policy_init["best_checkpoint"]),
                pipeline_name=pipeline_name,
                args=args,
            ),
        )
        return species_summary

    if effective_mode == "evolution":
        evolution_summary = run_evolutionary_search(
            EvolutionConfig(
                species=species,
                dataset=dataset_path,
                run_name=pipeline_name,
                output_root=str(species_root),
                seed=args.seed,
                max_aa_len=args.max_aa_len,
                population_size=args.population_size,
                generations=args.generations,
                subset_size=args.subset_size,
            )
        )
        species_summary["evolution"] = {
            "engine": "local",
            "evolution_dir": evolution_summary["evolution_dir"],
            "optimized_genes": evolution_summary["optimized_genes"],
            "fitness_mean": evolution_summary["fitness_mean"],
        }
        species_summary["final_inference"] = write_inference_artifacts(
            species_root=species_root,
            spec_payload=build_evolution_hybrid_inference_spec(
                species=species,
                dataset_path=dataset_path,
                checkpoint_path=str(policy_init["best_checkpoint"]),
                pipeline_name=pipeline_name,
                args=args,
            ),
        )
        return species_summary

    if not has_abundance:
        raise ValueError(f"{species} branch selected RM/RL, but dataset has no abundance labels")

    rm_stage_dir = species_root / "rm_autoresearch"
    rm_stage = launch_karpathy_stage(
        stage="rm",
        spec_payload={
            "species": species,
            "dataset": dataset_path,
            "validation_dataset": split_info["validation_csv"],
            "output_dir": str(rm_stage_dir),
            "seed": args.seed,
            "max_aa_len": None,
            "reward_alpha": args.reward_alpha,
            "max_train_records": args.max_train_records,
            "max_validation_records": args.max_validation_records,
        },
        stage_dir=rm_stage_dir,
        codex_model=args.codex_model or None,
        stage_timeout_seconds=args.rm_stage_seconds,
        base_ref=args.worker_base_ref,
        keep_worktree=not args.cleanup_worktrees,
    )
    if not rm_stage.timed_out:
        raise RuntimeError(
            "RM karpathy stage ended before the external stage timeout; refusing to start RL "
            f"for {species}. timed_out={rm_stage.timed_out}, codex_returncode={rm_stage.codex_returncode}, "
            f"attempts={rm_stage.codex_attempts}"
        )
    rm_bundle_json = Path(rm_stage.best_reward_bundle_json or "")
    if not rm_bundle_json.exists():
        raise RuntimeError(f"RM karpathy stage for {species} did not materialize a canonical best reward bundle: {rm_bundle_json}")
    species_summary["reward_autoresearch"] = {
        "worker_session": {
            "branch_name": rm_stage.branch_name,
            "worktree_dir": rm_stage.worktree_dir,
            "package_dir": rm_stage.package_dir,
            "unit_dir": rm_stage.unit_dir,
            "program_md": rm_stage.program_md,
            "timed_out": rm_stage.timed_out,
            "codex_returncode": rm_stage.codex_returncode,
            "codex_attempts": rm_stage.codex_attempts,
        },
        "results_tsv": rm_stage.results_tsv,
        "best_summary_json": rm_stage.best_summary_json,
        "best_reward_bundle_json": str(rm_bundle_json),
        "objective_name": "validation_spearman_rho",
    }

    rl_prompt_split = prepare_fixed_uniref_prompt_split(
        species_root=species_root,
        max_aa_len=RL_MAX_AA_LEN,
        train_prompt_path=args.rl_train_prompts,
        validation_prompt_path=args.rl_validation_prompts,
    )
    rl_stage_dir = species_root / "rl_autoresearch"
    rl_stage = launch_karpathy_stage(
        stage="rl",
        spec_payload={
            "species": species,
            "dataset": dataset_path,
            "init_checkpoint": str(policy_init["best_checkpoint"]),
            "reward_bundle_json": str(rm_bundle_json),
            "train_prompt_fasta": rl_prompt_split["train_prompt_fasta"],
            "validation_prompt_fasta": rl_prompt_split["validation_prompt_fasta"],
            "output_dir": str(rl_stage_dir),
            "seed": args.seed,
            "max_aa_len": RL_MAX_AA_LEN,
            "steps": args.rl_steps,
            "prompt_batch_size": RL_TRAIN_PROMPT_BATCH_SIZE,
            "group_size": RL_GROUP_SIZE,
            "policy_mini_batch_size": RL_POLICY_MINI_BATCH_SIZE,
            "validation_prompt_batch_size": RL_VALIDATION_PROMPT_BATCH_SIZE,
            "validation_group_size": RL_VALIDATION_GROUP_SIZE,
            "lr": args.rl_lr,
            "dtype": args.dtype,
            "device": args.device,
            "best_metric_key": args.rl_best_metric_key,
        },
        stage_dir=rl_stage_dir,
        codex_model=args.codex_model or None,
        stage_timeout_seconds=args.rl_stage_seconds,
        base_ref=args.worker_base_ref,
        keep_worktree=not args.cleanup_worktrees,
    )
    rl_best_checkpoint = Path(rl_stage.best_checkpoint or "")
    if not rl_best_checkpoint.exists():
        raise RuntimeError(f"RL karpathy stage for {species} did not materialize a canonical best checkpoint: {rl_best_checkpoint}")
    species_summary["rl_autoresearch"] = {
        "worker_session": {
            "branch_name": rl_stage.branch_name,
            "worktree_dir": rl_stage.worktree_dir,
            "package_dir": rl_stage.package_dir,
            "unit_dir": rl_stage.unit_dir,
            "program_md": rl_stage.program_md,
            "timed_out": rl_stage.timed_out,
            "codex_returncode": rl_stage.codex_returncode,
            "codex_attempts": rl_stage.codex_attempts,
        },
        "results_tsv": rl_stage.results_tsv,
        "best_summary_json": rl_stage.best_summary_json,
        "best_checkpoint": str(rl_best_checkpoint),
        "objective_name": args.rl_best_metric_key,
        "fixed_prompt_split": rl_prompt_split,
    }
    species_summary["final_inference"] = write_inference_artifacts(
        species_root=species_root,
        spec_payload=build_seq2seq_inference_spec(
            species=species,
            dataset_path=dataset_path,
            checkpoint_path=str(rl_best_checkpoint),
            pipeline_name=pipeline_name,
            args=args,
        ),
    )
    return species_summary


def main() -> None:
    args = parse_args()
    species_list = resolve_species_list(args.species)
    pipeline_name = args.run_name or default_run_name("pipeline")
    output_root = Path(args.output_root).expanduser().resolve()
    pipeline_dir = output_root / "_pipelines" / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    species_summaries: list[dict[str, Any]] = []
    pipeline_summary = {
        "pipeline_name": pipeline_name,
        "pipeline_dir": str(pipeline_dir),
        "execution_model": "fixed_pipeline",
        "species_summaries": species_summaries,
    }

    for species in species_list:
        species_root = pipeline_dir / slugify(species)
        species_root.mkdir(parents=True, exist_ok=True)
        summary = run_pipeline_for_species(
            species=species,
            pipeline_name=pipeline_name,
            args=args,
            output_root=output_root,
            species_root=species_root,
        )
        species_summaries.append(summary)

    save_json(pipeline_dir / "summary.json", pipeline_summary)
    print(f"pipeline_dir: {pipeline_summary['pipeline_dir']}")
    for species_summary in species_summaries:
        print(
            f"{species_summary['species']}: "
            f"branch={species_summary['dataset_agent']['branch_name']} "
            f"mode={species_summary['effective_mode']} "
            f"infer={species_summary['final_inference']['run_script']}"
        )


if __name__ == "__main__":
    main()
