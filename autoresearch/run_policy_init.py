from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autoresearch.common import save_json
    from autoresearch.train import TrainConfig, run_supervised_training
else:
    from .common import save_json
    from .train import TrainConfig, run_supervised_training


@dataclass
class PolicyInitSpec:
    species: str | None = None
    dataset: str | None = None
    output_dir: str = ""
    run_name: str = "policy_init"
    seed: int = 42
    max_aa_len: int = 512
    batch_size: int = 128
    eval_batch_size: int = 128
    max_epochs: int = 20
    num_workers: int = 4
    device: str = "cpu"
    dtype: str = "bf16"
    lr: float = 3e-4
    max_train_records: int | None = None
    max_test_records: int | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "PolicyInitSpec":
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed local AA->CDS policy-init training stage.")
    parser.add_argument("--spec", required=True)
    return parser.parse_args()


def run_policy_init(spec: PolicyInitSpec) -> dict[str, Any]:
    output_root = Path(spec.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = run_supervised_training(
        TrainConfig(
            species=spec.species,
            dataset=spec.dataset,
            run_name=spec.run_name,
            output_root=str(output_root),
            seed=spec.seed,
            max_aa_len=spec.max_aa_len,
            batch_size=spec.batch_size,
            eval_batch_size=spec.eval_batch_size,
            max_epochs=spec.max_epochs,
            num_workers=spec.num_workers,
            device=spec.device,
            dtype=spec.dtype,
            lr=spec.lr,
            max_train_records=spec.max_train_records,
            max_test_records=spec.max_test_records,
        )
    )
    payload = {
        "policy_init_spec": asdict(spec),
        **summary,
    }
    save_json(output_root / "policy_init_summary.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    summary = run_policy_init(PolicyInitSpec.from_path(args.spec))
    print("status: ok")
    print(f"best_checkpoint: {summary['best_checkpoint']}")
    print(f"summary_json: {Path(summary['policy_init_spec']['output_dir']) / 'policy_init_summary.json'}")


if __name__ == "__main__":
    main()
