# autoresearch

`autoresearch/` is a self-contained CDS optimization pipeline. It takes a species-level CDS dataset, chooses the appropriate branch, optionally runs reward-model and reinforcement-learning autoresearch, and exports an `aas2cds` inference entrypoint.

The folder is intended to be movable as one unit. It should not depend on sibling directories, but it must live inside a git repository when running the Karpathy-style worker because the worker uses `git worktree`.

## What This Folder Contains

- `run_pipeline.py`: the main end-to-end entrypoint.
- `run_timed_pipeline.sh`: outer wall-clock wrapper. The default total timeout is `16h`.
- `dataset_agent.py`: Codex-based dataset collector and branch decider.
- `prepare.py`: local dataset inspection and branch recommendation.
- `train.py`: local supervised `AA -> CDS` policy initialization.
- `run_policy_init.py`: wrapper used by the pipeline before RL.
- `rm_autoresearch/`: reward-model autoresearch unit.
- `rl_autoresearch/`: reinforcement-learning autoresearch unit.
- `karpathy_worker.py`: creates a git worktree, renders a stage-specific `program.md`, and launches `codex exec`.
- `evolution.py`: no-abundance branch optimizer.
- `infer.py`: final inference/export helpers.
- `assets/`: dataset-agent SOP, model registry, and fixed UniRef prompt files.
- `runs/`: generated outputs, pipeline runs, worktrees, logs, checkpoints, and summaries.

## Pipeline Shape

For each species, the pipeline does:

1. Collect or standardize the species dataset with the dataset agent.
2. Decide the branch:
   - PaxDB abundance labels: RM autoresearch, policy init, RL autoresearch.
   - transcriptome TPM labels: RM autoresearch, policy init, RL autoresearch.
   - no abundance labels: local evolution branch, no RM/RL.
3. Export a final inference package under the species run directory.

Only the RM and RL branches invoke the Karpathy-style autoresearch worker.

## Karpathy-Style Worker Contract

Each RM/RL stage is run in an isolated git worktree under:

```text
runs/_pipelines/<run_name>/<species>/<stage>/worktree/
```

The worker generates a fresh runtime `program.md` inside that worktree. The static `program.md` files in the repository are documentation, not the exact prompt used by a live worker.

For each stage:

- `prepare.py` is fixed.
- `train.py` is the only research surface the agent may edit.
- `results.tsv` is appended with trial outcomes.
- Better trials are kept as git commits inside the worktree.
- Failed or worse trials are reset/discarded.

The launcher now relaunches Codex if it exits cleanly before the stage deadline, so a premature normal exit does not end the stage immediately.

## Fixed RM Settings

RM autoresearch learns a reward model from a fixed train split and a fixed validation split.

- Objective: `validation_spearman_rho`.
- Training data: `spec.dataset` (`train.csv`).
- Validation data: `spec.validation_dataset` (`validation.csv`).
- Validation labels may be used for repeated evaluation and tuning, but not for direct training.
- Single-trial soft budget: `TIME_BUDGET = 300s`.
- Single-trial hard timeout: `10m`.
- RM stage default budget: `8h` (`--rm-stage-seconds 28800`).

## Fixed RL Settings

RL autoresearch optimizes a policy initialized from the local supervised `AA -> CDS` checkpoint, using the fixed RM reward model.

The algorithm is constrained to the current GRPO-style family:

- grouped sampling,
- group-relative or normalized advantages,
- policy-gradient optimization against the fixed reward model.

The worker must not replace this with SFT, behavior cloning, PPO, DPO, random initialization, or another reward/checkpoint.

Current fixed RL resource settings are:

- `max_aa_len = 512`
- `steps = 120`
- training `prompt_batch_size = 72`
- training `group_size = 4`
- `policy_mini_batch_size = 64`
- validation `prompt_batch_size = 300`
- validation `group_size = 4`
- single-trial soft budget: `TIME_BUDGET = 900s` on 4090/4090D, `1500s` on 3090
- single-trial hard timeout: `20m` on 4090/4090D, `30m` on 3090
- RL stage default budget: `4h` (`--rl-stage-seconds 14400`)

The worker selects the 3090 trial budget when local `nvidia-smi` reports an RTX 3090.

The fixed UniRef prompt files are:

```text
assets/prompt_library/uniref10_fixed_train.fasta
assets/prompt_library/uniref10_fixed_validation.fasta
```

With `max_aa_len = 512`, the current fixed prompt files load about `8925` training prompts and `279` validation prompts. The validation batch size of `300` therefore evaluates the whole validation split in one chunk.

## Requirements

Python dependencies are listed in `pyproject.toml`.

System tools needed for the full autoresearch path:

- `git`
- `codex`
- `timeout` or `gtimeout`
- CUDA-capable PyTorch for GPU runs

The no-abundance evolution branch also requires `RNAfold`.

## Running

Enter the folder:

```bash
cd VenusAutoCodon/autoresearch
```

Inspect available local species datasets:

```bash
python prepare.py --list-species
```

Inspect one species:

```bash
python prepare.py --species saccharomyces_cerevisiae
```

Run the supervised local policy-init path only:

```bash
python run_pipeline.py \
  --species saccharomyces_cerevisiae \
  --mode supervised \
  --run-name scer_supervised \
  --device cuda
```

Run the full automatic pipeline:

```bash
python run_pipeline.py \
  --species saccharomyces_cerevisiae \
  --mode auto \
  --run-name scer_auto \
  --device cuda
```

Run with the outer total wall-clock wrapper:

```bash
./run_timed_pipeline.sh \
  --species saccharomyces_cerevisiae \
  --mode auto \
  --run-name scer_auto_16h \
  --device cuda
```

Override the wrapper total timeout if needed:

```bash
TOTAL_HOURS=24 ./run_timed_pipeline.sh \
  --species saccharomyces_cerevisiae \
  --mode auto \
  --run-name scer_auto_24h \
  --device cuda
```

Useful runtime knobs:

- `--dataset-agent-seconds`: timeout for dataset collection.
- `--rm-stage-seconds`: total RM autoresearch stage budget.
- `--rl-stage-seconds`: total RL autoresearch stage budget.
- `--rl-steps`: RL update-step target per trial.
- `--codex-model`: Codex model override for worker stages.
- `--cleanup-worktrees`: remove stage worktrees after completion.

## Outputs

Default output root:

```text
runs/
```

Pipeline summary directory:

```text
runs/_pipelines/<run_name>/
```

Typical per-species outputs:

```text
runs/_pipelines/<run_name>/<species>/
  dataset_agent/
  fixed_split/
  policy_init/
  rm_autoresearch/
  rl_autoresearch/
  inference/
```

Typical RM/RL stage artifacts:

- `worker_session.json`
- `results.tsv`
- `best_summary.json`
- `best_reward_bundle.json` for RM
- `best_rl.pt` for RL
- `logs/`
- `trial_runs/`
- `worktree/`

Final inference artifacts are written under:

```text
runs/_pipelines/<run_name>/<species>/inference/
```

## Development Notes

- Commit code changes before launching new autoresearch stages. The worker creates worktrees from git `HEAD`, so uncommitted edits in the source folder are not automatically visible inside new worker worktrees.
- Keep RM and RL resource/time-budget constants synchronized across `run_pipeline.py`, `rl_autoresearch/prepare.py`, and `karpathy_worker.py`.
- Do not treat historical files under `runs/` as source-of-truth configuration; they are run artifacts.
- Avoid editing generated worktrees directly unless debugging a specific stage.
