from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim

from prepare import (
    TIME_BUDGET,
    autocast_dtype,
    decode_cds_batch,
    evaluate_policy,
    group_advantages,
    load_model_from_checkpoint,
    load_trial_context,
    pad_source_batch,
    repeat_prompts,
    sample_sequences,
    save_json,
    save_trial_summary,
    sequence_logprob_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fixed RL autoresearch trial.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--trial-name", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def run_trial(spec_path: str | Path, trial_name: str) -> dict[str, Any]:
    total_start = time.perf_counter()
    context = load_trial_context(spec_path, trial_name)
    spec = context.spec

    set_seed(spec.seed)
    device = torch.device(spec.device)
    amp_dtype = autocast_dtype(spec.dtype)

    model, model_config, init_payload = load_model_from_checkpoint(spec.init_checkpoint, map_location="cpu")
    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)

    history: list[dict[str, Any]] = []
    best_path = context.trial_dir / "best.pt"
    last_path = context.trial_dir / "last.pt"

    initial_validation = evaluate_policy(
        model,
        context.reward_model,
        context.validation_records,
        prompt_batch_size=spec.validation_prompt_batch_size,
        group_size=spec.validation_group_size,
        temperature=spec.temperature,
        top_k=spec.top_k,
        amp_dtype=amp_dtype,
        seed=spec.seed + 1000,
        device=device,
    )
    best_metric = float(initial_validation[spec.best_metric_key])
    best_step = 0
    no_improve_evals = 0
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": 0,
            "model_config": model_config.to_dict(),
            "rl_config": asdict(spec),
            "dataset_meta": context.dataset_meta,
            "best_metrics": initial_validation,
        },
        best_path,
    )

    rng = np.random.default_rng(spec.seed + 17)
    training_start = time.perf_counter()
    for step in range(1, spec.steps + 1):
        if time.perf_counter() - training_start >= TIME_BUDGET:
            break

        batch_size = min(spec.prompt_batch_size, len(context.train_records))
        batch_idx = rng.choice(len(context.train_records), size=batch_size, replace=False)
        batch_records = [context.train_records[int(index)] for index in batch_idx]
        src_ids, src_padding_mask = pad_source_batch(batch_records)
        src_ids = src_ids.to(device)
        src_padding_mask = src_padding_mask.to(device)

        with torch.no_grad():
            generated_ids = sample_sequences(
                model,
                src_ids=src_ids,
                src_padding_mask=src_padding_mask,
                group_size=spec.group_size,
                temperature=spec.temperature,
                top_k=spec.top_k,
                sample_seed=spec.seed + step * 1009,
                amp_dtype=amp_dtype,
            )

        repeated_src_ids, repeated_src_mask = repeat_prompts(src_ids, src_padding_mask, spec.group_size)
        cds_batch = decode_cds_batch(generated_ids)
        reward_raw = context.reward_model.predict_reward(cds_batch).reshape(len(batch_records), spec.group_size)
        advantage = group_advantages(reward_raw).reshape(-1)

        new_logp, entropy = sequence_logprob_stats(
            model,
            src_ids=repeated_src_ids,
            src_padding_mask=repeated_src_mask,
            sequences=generated_ids,
            amp_dtype=amp_dtype,
            mini_batch_size=spec.policy_mini_batch_size,
        )
        advantage_t = torch.tensor(advantage, dtype=new_logp.dtype, device=new_logp.device)
        loss = -(new_logp * advantage_t).mean() - spec.entropy_coef * entropy.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip_norm)
        optimizer.step()

        history_row = {
            "step": int(step),
            "loss": float(loss.item()),
            "entropy": float(entropy.mean().item()),
            "reward_mean": float(reward_raw.mean()),
            "reward_best_mean": float(reward_raw.max(axis=1).mean()),
            "elapsed_training_seconds": float(time.perf_counter() - training_start),
        }

        if step % spec.eval_every == 0 or step == spec.steps:
            model.eval()
            validation_eval = evaluate_policy(
                model,
                context.reward_model,
                context.validation_records,
                prompt_batch_size=spec.validation_prompt_batch_size,
                group_size=spec.validation_group_size,
                temperature=spec.temperature,
                top_k=spec.top_k,
                amp_dtype=amp_dtype,
                seed=spec.seed + 1000,
                device=device,
            )
            model.train()
            history_row.update({f"validation_{key}": value for key, value in validation_eval.items()})
            current_metric = float(validation_eval[spec.best_metric_key])
            if current_metric > best_metric + 1e-5:
                best_metric = current_metric
                best_step = step
                no_improve_evals = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "step": step,
                        "model_config": model_config.to_dict(),
                        "rl_config": asdict(spec),
                        "dataset_meta": context.dataset_meta,
                        "best_metrics": validation_eval,
                    },
                    best_path,
                )
            else:
                no_improve_evals += 1
            if no_improve_evals >= spec.patience_evals:
                history.append(history_row)
                break

        history.append(history_row)

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": len(history),
            "model_config": model_config.to_dict(),
            "rl_config": asdict(spec),
            "dataset_meta": context.dataset_meta,
            "best_metrics": {spec.best_metric_key: best_metric},
        },
        last_path,
    )

    best_model, _, best_payload = load_model_from_checkpoint(best_path, map_location="cpu")
    best_model = best_model.to(device)
    best_model.eval()
    final_validation = evaluate_policy(
        best_model,
        context.reward_model,
        context.validation_records,
        prompt_batch_size=spec.validation_prompt_batch_size,
        group_size=spec.validation_group_size,
        temperature=spec.temperature,
        top_k=spec.top_k,
        amp_dtype=amp_dtype,
        seed=spec.seed + 1000,
        device=device,
    )
    summary = {
        "trial_name": trial_name,
        "trial_dir": str(context.trial_dir),
        "trial_spec": asdict(spec),
        "dataset_name": context.dataset_meta["dataset_name"],
        "dataset_path": context.dataset_meta["dataset_path"],
        "split_sizes": context.split_sizes,
        "init_checkpoint": str(Path(spec.init_checkpoint).expanduser().resolve()),
        "reward_bundle": str(Path(spec.reward_bundle_json).expanduser().resolve()),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_metric_key": spec.best_metric_key,
        "best_metric_value": float(final_validation[spec.best_metric_key]),
        "best_step": int(best_step),
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "training_seconds": float(time.perf_counter() - training_start),
        "total_seconds": float(time.perf_counter() - total_start),
        "history": history,
        "init_payload_keys": sorted(init_payload.keys()),
        "best_payload_keys": sorted(best_payload.keys()),
        "objective_name": spec.best_metric_key,
        "objective_score": float(final_validation[spec.best_metric_key]),
        "peak_vram_mb": peak_vram_mb(),
        "time_budget_seconds": TIME_BUDGET,
    }
    save_trial_summary(context.trial_dir, summary)
    save_json(context.trial_dir / "history.json", {"history": history})
    return summary


def main() -> None:
    args = parse_args()
    summary = run_trial(args.spec, args.trial_name)
    print("---")
    print(f"score: {summary['objective_score']:.6f}")
    print(f"training_seconds: {summary['training_seconds']:.1f}")
    print(f"total_seconds: {summary['total_seconds']:.1f}")
    print(f"peak_vram_mb: {summary['peak_vram_mb']:.1f}")
    print(f"time_budget_seconds: {summary['time_budget_seconds']}")
    print(f"summary_json: {Path(summary['trial_dir']) / 'summary.json'}")
    print(f"best_checkpoint: {summary['best_checkpoint']}")
    print(f"objective_name: {summary['objective_name']}")


if __name__ == "__main__":
    main()
