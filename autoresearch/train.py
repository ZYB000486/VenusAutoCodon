from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autoresearch.common import RUNS_DIR, default_run_name, save_json, set_seed
    from autoresearch.dataset import Seq2SeqDataset, collate_examples, load_records, slice_splits, split_records
    from autoresearch.model import Seq2SeqConfig, build_model, count_parameters, load_model_from_checkpoint
    from autoresearch.vocab import TGT_PAD_ID
else:
    from .common import RUNS_DIR, default_run_name, save_json, set_seed
    from .dataset import Seq2SeqDataset, collate_examples, load_records, slice_splits, split_records
    from .model import Seq2SeqConfig, build_model, count_parameters, load_model_from_checkpoint
    from .vocab import TGT_PAD_ID


@dataclass
class TrainConfig:
    species: str | None = None
    dataset: str | None = None
    run_name: str = ""
    output_root: str = str(RUNS_DIR)
    seed: int = 42
    max_aa_len: int = 512
    batch_size: int = 128
    eval_batch_size: int = 128
    max_epochs: int = 20
    patience: int = 4
    min_delta: float = 1e-4
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bf16"
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    disable_codon_constraint: bool = False
    max_train_records: int | None = None
    max_test_records: int | None = None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a small autoresearch AA->CDS Transformer.")
    parser.add_argument("--species", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default=str(RUNS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-aa-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--disable-codon-constraint", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    return TrainConfig(**vars(parser.parse_args()))


def get_autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "fp32":
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def build_grad_scaler(device: torch.device, dtype_name: str):
    enabled = device.type == "cuda" and dtype_name == "fp16"
    amp_module = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_module, "GradScaler", None) if amp_module is not None else None
    if grad_scaler_cls is not None:
        try:
            return grad_scaler_cls("cuda", enabled=enabled)
        except TypeError:
            return grad_scaler_cls(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_scheduler(optimizer: optim.Optimizer, warmup_steps: int, total_steps: int):
    total_steps = max(1, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def batch_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, int]:
    pred = logits.argmax(dim=-1)
    valid = labels.ne(TGT_PAD_ID)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return 0.0, 0.0, 0
    token_correct = float((pred.eq(labels) & valid).sum().item())
    exact = float(((pred.eq(labels) | ~valid).all(dim=1)).float().sum().item())
    return token_correct, exact, valid_count


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, *, device: torch.device, dtype_name: str) -> dict[str, float]:
    model.eval()
    autocast_context = get_autocast_context(device, dtype_name)

    total_loss = 0.0
    total_token_correct = 0.0
    total_exact = 0.0
    total_tokens = 0
    total_examples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with autocast_context:
            logits = model(
                batch["src_ids"],
                batch["tgt_input_ids"],
                src_padding_mask=batch["src_padding_mask"],
                tgt_padding_mask=batch["tgt_padding_mask"],
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch["labels"].reshape(-1),
                ignore_index=TGT_PAD_ID,
            )
        token_correct, exact_count, token_count = batch_metrics(logits, batch["labels"])
        batch_size = int(batch["src_ids"].size(0))
        total_loss += float(loss.item()) * batch_size
        total_token_correct += token_correct
        total_exact += exact_count
        total_tokens += token_count
        total_examples += batch_size

    return {
        "loss": total_loss / max(1, total_examples),
        "token_accuracy": total_token_correct / max(1, total_tokens),
        "exact_match": total_exact / max(1, total_examples),
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    model_config: Seq2SeqConfig,
    train_config: TrainConfig,
    dataset_meta: dict[str, Any],
    best_metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "model_config": model_config.to_dict(),
            "train_config": asdict(train_config),
            "dataset_meta": dataset_meta,
            "best_metrics": best_metrics,
        },
        path,
    )


def run_supervised_training(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)

    dataset_meta, records = load_records(
        species=config.species,
        dataset=config.dataset,
        max_aa_len=config.max_aa_len,
        require_abundance=False,
    )
    split_map = split_records(records, seed=config.seed)
    split_map = slice_splits(
        split_map,
        max_train_records=config.max_train_records,
        max_test_records=config.max_test_records,
    )
    dataset_meta["effective_split_sizes"] = {split_name: len(items) for split_name, items in split_map.items()}
    if not split_map["train"] or not split_map["test"]:
        raise ValueError("Not enough records for train/test after filtering")

    output_root = Path(config.output_root).expanduser().resolve()
    run_name = config.run_name or default_run_name(dataset_meta["dataset_name"])
    run_dir = output_root / dataset_meta["dataset_name"] / run_name / "supervised"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        Seq2SeqDataset(split_map["train"]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_examples,
    )
    test_loader = DataLoader(
        Seq2SeqDataset(split_map["test"]),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_examples,
    )

    model_config = Seq2SeqConfig(
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_aa_len,
        use_codon_constraint=not config.disable_codon_constraint,
    )
    model = build_model(model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * config.max_epochs)
    scheduler = build_scheduler(optimizer, warmup_steps=int(total_steps * config.warmup_ratio), total_steps=total_steps)
    scaler = build_grad_scaler(device, config.dtype)
    autocast_context = get_autocast_context(device, config.dtype)

    history: list[dict[str, Any]] = []
    best_metrics = {"loss": float("inf"), "token_accuracy": 0.0, "exact_match": 0.0, "epoch": 0}
    epochs_without_improvement = 0
    start_time = time.time()

    save_json(run_dir / "dataset_meta.json", dataset_meta)
    save_json(run_dir / "train_config.json", asdict(config))
    save_json(run_dir / "model_config.json", model_config.to_dict())

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_token_correct = 0.0
        train_exact = 0.0
        train_tokens = 0
        train_examples = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                logits = model(
                    batch["src_ids"],
                    batch["tgt_input_ids"],
                    src_padding_mask=batch["src_padding_mask"],
                    tgt_padding_mask=batch["tgt_padding_mask"],
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["labels"].reshape(-1),
                    ignore_index=TGT_PAD_ID,
                    label_smoothing=config.label_smoothing,
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            scheduler.step()
            token_correct, exact_count, token_count = batch_metrics(logits.detach(), batch["labels"])
            batch_size = int(batch["src_ids"].size(0))
            train_loss_sum += float(loss.item()) * batch_size
            train_token_correct += token_correct
            train_exact += exact_count
            train_tokens += token_count
            train_examples += batch_size

        train_metrics = {
            "loss": train_loss_sum / max(1, train_examples),
            "token_accuracy": train_token_correct / max(1, train_tokens),
            "exact_match": train_exact / max(1, train_examples),
        }
        test_metrics = evaluate(model, test_loader, device=device, dtype_name=config.dtype)
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_token_accuracy": train_metrics["token_accuracy"],
            "train_exact_match": train_metrics["exact_match"],
            "test_loss": test_metrics["loss"],
            "test_token_accuracy": test_metrics["token_accuracy"],
            "test_exact_match": test_metrics["exact_match"],
            "epoch_seconds": time.time() - epoch_start,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(history_row)

        improved = (
            test_metrics["token_accuracy"] > best_metrics["token_accuracy"] + config.min_delta
            or (
                abs(test_metrics["token_accuracy"] - best_metrics["token_accuracy"]) <= config.min_delta
                and test_metrics["loss"] < best_metrics["loss"] - config.min_delta
            )
        )
        if improved:
            best_metrics = {**test_metrics, "epoch": epoch}
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                model_config=model_config,
                train_config=config,
                dataset_meta=dataset_meta,
                best_metrics=best_metrics,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    save_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=len(history),
        model_config=model_config,
        train_config=config,
        dataset_meta=dataset_meta,
        best_metrics=best_metrics,
    )
    best_model, _, _ = load_model_from_checkpoint(run_dir / "best.pt", map_location="cpu")
    best_model = best_model.to(device)
    test_metrics = evaluate(best_model, test_loader, device=device, dtype_name=config.dtype)
    summary = {
        "dataset_name": dataset_meta["dataset_name"],
        "dataset_path": dataset_meta["dataset_path"],
        "run_name": run_name,
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "model_config": model_config.to_dict(),
        "parameter_count": int(count_parameters(model)),
        "dataset_meta": dataset_meta,
        "best_test": best_metrics,
        "test_metrics": test_metrics,
        "epochs_ran": len(history),
        "training_seconds": time.time() - start_time,
        "history": history,
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_supervised_training(parse_args())
    print(f"run_dir: {summary['run_dir']}")
    print(f"best_checkpoint: {summary['best_checkpoint']}")
    print(f"best_test_token_accuracy: {summary['best_test']['token_accuracy']:.4f}")
    print(f"test_token_accuracy: {summary['test_metrics']['token_accuracy']:.4f}")


if __name__ == "__main__":
    main()
