from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prepare import (
    OBJECTIVE_NAME,
    RewardBundle,
    RewardTrialSpec,
    TIME_BUDGET,
    evaluate_bundle,
    load_trial_context,
    save_reward_bundle,
    save_trial_summary,
)


@dataclass
class RMTrainConfig:
    alpha: float = 1.0
    feature_std_floor: float = 1e-8
    target_std_floor: float = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fixed RM autoresearch trial.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--trial-name", required=True)
    return parser.parse_args()


def standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray, floor: float) -> np.ndarray:
    safe_std = std.copy()
    safe_std[safe_std < floor] = 1.0
    return (matrix - mean) / safe_std


def fit_ridge_closed_form(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + alpha * penalty, x_aug.T @ y_train)


def predict_linear(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return x_aug @ weights


def peak_vram_mb() -> float:
    try:
        import torch
    except Exception:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def run_trial(spec_path: str | Path, trial_name: str) -> dict[str, Any]:
    total_start = time.perf_counter()
    context = load_trial_context(spec_path, trial_name)
    train_start = time.perf_counter()

    spec = RewardTrialSpec.from_path(spec_path)
    train_config = RMTrainConfig(alpha=float(spec.reward_alpha))

    target_mean = float(context.y_train.mean())
    target_std = float(context.y_train.std())
    if target_std < train_config.target_std_floor:
        target_std = 1.0
    y_train_z = (context.y_train - target_mean) / target_std

    feature_mean = context.x_train.mean(axis=0)
    feature_std = context.x_train.std(axis=0)
    x_train_std = standardize(context.x_train, feature_mean, feature_std, train_config.feature_std_floor)
    x_validation_std = standardize(context.x_validation, feature_mean, feature_std, train_config.feature_std_floor)

    weights = fit_ridge_closed_form(x_train_std, y_train_z, alpha=train_config.alpha)
    train_pred = predict_linear(x_train_std, weights) * target_std + target_mean
    validation_pred = predict_linear(x_validation_std, weights) * target_std + target_mean

    bundle = RewardBundle(
        reference=context.reference.to_dict(),
        feature_names=context.feature_names,
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        weights=weights.tolist(),
        target_mean=target_mean,
        target_std=target_std,
        alpha=float(train_config.alpha),
        train_metrics={
            "pearson_r": float(np.corrcoef(context.y_train, train_pred)[0, 1]) if len(context.y_train) > 1 else 0.0,
            "spearman_rho": 0.0,
            "rmse": float(np.sqrt(np.mean((context.y_train - train_pred) ** 2))),
        },
        validation_metrics={
            "pearson_r": float(np.corrcoef(context.y_validation, validation_pred)[0, 1]) if len(context.y_validation) > 1 else 0.0,
            "spearman_rho": 0.0,
            "rmse": float(np.sqrt(np.mean((context.y_validation - validation_pred) ** 2))),
        },
    )

    bundle_path = context.trial_dir / "reward_bundle.json"
    save_reward_bundle(bundle, bundle_path)

    validation_metrics = evaluate_bundle(bundle_path, context.validation_records)
    bundle.train_metrics = evaluate_bundle(bundle_path, context.train_records)
    bundle.validation_metrics = validation_metrics
    save_reward_bundle(bundle, bundle_path)

    training_seconds = time.perf_counter() - train_start
    summary = {
        "trial_name": trial_name,
        "trial_dir": str(context.trial_dir),
        "dataset_meta": context.dataset_meta,
        "split_sizes": context.split_sizes,
        "trial_spec": asdict(context.spec),
        "train_config": asdict(train_config),
        "reward_bundle": str(bundle_path),
        "bundle_train_metrics": bundle.train_metrics,
        "bundle_validation_metrics": bundle.validation_metrics,
        "validation_metrics": validation_metrics,
        "objective_name": OBJECTIVE_NAME,
        "objective_score": float(validation_metrics["spearman_rho"]),
        "training_seconds": float(training_seconds),
        "total_seconds": float(time.perf_counter() - total_start),
        "peak_vram_mb": peak_vram_mb(),
        "time_budget_seconds": TIME_BUDGET,
    }
    save_trial_summary(context.trial_dir, summary)
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
    print(f"reward_bundle_json: {summary['reward_bundle']}")
    print(f"objective_name: {summary['objective_name']}")


if __name__ == "__main__":
    main()
