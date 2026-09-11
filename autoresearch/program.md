# autoresearch program

This file is a static overview for humans. During an RM or RL autoresearch stage, `karpathy_worker.py` renders a fresh stage-specific `program.md` inside that stage's git worktree. The generated file is the prompt actually read by the worker.

## Entry Point

```bash
python run_pipeline.py --species <species> --mode auto
```

## Branches

- Species with PaxDB abundance labels:
  - dataset agent
  - RM autoresearch
  - local policy init
  - RL autoresearch
  - RL `aas2cds` inference export
- Species with transcriptome TPM labels:
  - dataset agent
  - RM autoresearch
  - local policy init
  - RL autoresearch
  - RL `aas2cds` inference export
- Species without abundance labels:
  - dataset agent
  - local evolution branch
  - EA-hybrid `aas2cds` inference export

Only `rm_autoresearch/` and `rl_autoresearch/` invoke the Karpathy-style worker.

## Worker Contract

When `run_pipeline.py` enters RM or RL, the worker:

1. creates an isolated git worktree and branch,
2. writes a stage-specific runtime `program.md`,
3. commits that runtime `program.md`,
4. launches `codex exec`,
5. relaunches Codex if it exits cleanly before the stage deadline.

For both RM and RL:

- `prepare.py` is fixed.
- `train.py` is the only file the agent may edit.
- dataset/spec/prompt/reward/checkpoint files are fixed inputs.
- trial results are appended to `results.tsv`.
- better trials are kept; failed or worse trials are reset/discarded.

## RM Stage

- Objective: `validation_spearman_rho`.
- Training split: `spec.dataset`.
- Validation split: `spec.validation_dataset`.
- The validation split may be used repeatedly for evaluation and tuning, but not for direct training.
- Single-trial soft budget: `TIME_BUDGET = 300s`.
- Single-trial hard timeout: `10m`.
- Stage budget default: `8h` (`--rm-stage-seconds 28800`).

## RL Stage

- Objective: validation reward, selected by `--rl-best-metric-key`.
- Algorithm family: GRPO-style grouped sampling, group-relative or normalized advantages, and policy-gradient optimization of the fixed reward model.
- Init policy: `spec.init_checkpoint`.
- Reward model: `spec.reward_bundle_json`.
- Train prompts: `spec.train_prompt_fasta`.
- Validation prompts: `spec.validation_prompt_fasta`.

Fixed RL resource settings:

- `spec.max_aa_len = 512`
- `spec.steps = 120`
- `spec.prompt_batch_size = 72`
- `spec.group_size = 4`
- `spec.policy_mini_batch_size = 64`
- `spec.validation_prompt_batch_size = 300`
- `spec.validation_group_size = 4`

RL time budgets:

- Single-trial soft budget: `TIME_BUDGET = 900s` on 4090/4090D, `1500s` on 3090.
- Single-trial hard timeout: `20m` on 4090/4090D, `30m` on 3090.
- Stage budget default: `4h` (`--rl-stage-seconds 14400`).

The agent must not replace RL with SFT, behavior cloning, PPO, DPO, random initialization, another reward model, or another checkpoint.

## Outer Wrapper

`run_timed_pipeline.sh` is the outer total wall-clock wrapper. Its default total wall-clock limit is `16h`:

```bash
TOTAL_HOURS="${TOTAL_HOURS:-16}"
```

This wrapper timeout applies to the whole `run_pipeline.py` process, not to an individual RM or RL trial.

## Development Boundary

- Commit source changes before launching a worker stage. New worker worktrees are created from git `HEAD`.
- Do not treat files under `runs/` as source-of-truth configuration; they are historical run artifacts.
- Keep this static overview, `README.md`, `run_pipeline.py`, `karpathy_worker.py`, and the RM/RL `prepare.py` defaults synchronized.
