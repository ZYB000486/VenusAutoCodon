from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .common import save_json


PACKAGE_ROOT = Path(__file__).resolve().parent
RM_TRIAL_TIME_BUDGET_SECONDS = 300
RL_TRIAL_TIME_BUDGET_SECONDS = 900
RL_3090_TRIAL_TIME_BUDGET_SECONDS = 1500
RM_TRIAL_TIMEOUT_MINUTES = 10
RL_TRIAL_TIMEOUT_MINUTES = 20
RL_3090_TRIAL_TIMEOUT_MINUTES = 30
WATCHDOG_INTERVAL_SECONDS = 10


@dataclass
class KarpathyStageResult:
    stage: Literal["rm", "rl"]
    objective_name: str
    branch_name: str
    worktree_dir: str
    package_dir: str
    unit_dir: str
    program_md: str
    stage_dir: str
    spec_json: str
    results_tsv: str
    best_summary_json: str
    best_reward_bundle_json: str | None = None
    best_checkpoint: str | None = None
    codex_returncode: int | None = None
    timed_out: bool = False
    started_at_unix: int = 0
    finished_at_unix: int | None = None
    codex_attempts: list[dict[str, object]] = field(default_factory=list)


def worker_env() -> dict[str, str]:
    return os.environ.copy()


def local_gpu_names() -> list[str]:
    env = worker_env()
    nvidia_smi = shutil.which("nvidia-smi", path=env.get("PATH", ""))
    if nvidia_smi is None:
        return []
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
            env=env,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def trial_budget_for_stage(stage: Literal["rm", "rl"]) -> tuple[int, int]:
    if stage == "rm":
        return RM_TRIAL_TIME_BUDGET_SECONDS, RM_TRIAL_TIMEOUT_MINUTES
    gpu_names = local_gpu_names()
    if any("3090" in name for name in gpu_names):
        return RL_3090_TRIAL_TIME_BUDGET_SECONDS, RL_3090_TRIAL_TIMEOUT_MINUTES
    return RL_TRIAL_TIME_BUDGET_SECONDS, RL_TRIAL_TIMEOUT_MINUTES


def require_executable(name: str, env: dict[str, str]) -> str:
    path = shutil.which(name, path=env.get("PATH", ""))
    if path is None:
        raise FileNotFoundError(f"`{name}` was not found on PATH={env.get('PATH', '')}")
    return path


def resolve_git_root() -> Path:
    env = worker_env()
    git_bin = require_executable("git", env)
    proc = subprocess.run(
        [git_bin, "rev-parse", "--show-toplevel"],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError("Karpathy worker backend requires the autoresearch folder to live inside a git repository.")
    return Path(proc.stdout.strip()).resolve()


def package_relpath(git_root: Path) -> Path:
    try:
        return PACKAGE_ROOT.relative_to(git_root)
    except ValueError as exc:
        raise RuntimeError(f"Package root {PACKAGE_ROOT} is not under git root {git_root}") from exc


def worktree_package_dir(worktree_root: Path, package_path: Path) -> Path:
    return worktree_root if str(package_path) == "." else (worktree_root / package_path)


def run_git(args: list[str], *, cwd: Path) -> None:
    env = worker_env()
    git_bin = require_executable("git", env)
    proc = subprocess.run([git_bin, *args], cwd=cwd, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def create_worktree(
    *,
    git_root: Path,
    stage_dir: Path,
    stage: Literal["rm", "rl"],
    base_ref: str = "HEAD",
) -> tuple[str, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    branch_name = f"autoresearch_{stage}_{timestamp}_{os.getpid()}"
    worktree_dir = stage_dir / "worktree"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    run_git(["worktree", "add", "-b", branch_name, str(worktree_dir), base_ref], cwd=git_root)
    return branch_name, worktree_dir


def remove_worktree(*, git_root: Path, worktree_dir: Path, branch_name: str) -> None:
    env = worker_env()
    git_bin = require_executable("git", env)
    subprocess.run([git_bin, "worktree", "remove", "--force", str(worktree_dir)], cwd=git_root, check=False, env=env)
    subprocess.run([git_bin, "branch", "-D", branch_name], cwd=git_root, check=False, env=env)


def quoted(value: Path | str) -> str:
    return shlex.quote(str(value))


def process_table() -> list[dict[str, object]]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,etimes=,args="],
        text=True,
        capture_output=True,
        check=False,
    )
    rows: list[dict[str, object]] = []
    if proc.returncode != 0:
        return rows
    for raw_line in proc.stdout.splitlines():
        parts = raw_line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "ppid": int(parts[1]),
                    "etimes": int(parts[2]),
                    "args": parts[3],
                }
            )
        except ValueError:
            continue
    return rows


def descendant_pids(root_pid: int, rows: list[dict[str, object]]) -> set[int]:
    children: dict[int, list[int]] = {}
    for row in rows:
        children.setdefault(int(row["ppid"]), []).append(int(row["pid"]))
    stack = list(children.get(root_pid, []))
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return seen


def terminate_tree(root_pid: int) -> None:
    rows = process_table()
    targets = list(descendant_pids(root_pid, rows))
    targets.append(root_pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in targets:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        time.sleep(1.0)


def append_watchdog_event(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def enforce_trial_watchdog(
    *,
    root_pid: int,
    hard_timeout_seconds: int,
    unit_dir: Path,
    spec_json: Path,
    log_path: Path,
) -> None:
    rows = process_table()
    descendants = descendant_pids(root_pid, rows)
    if not descendants:
        return
    spec_text = str(spec_json)
    unit_text = str(unit_dir)
    for row in rows:
        pid = int(row["pid"])
        if pid not in descendants:
            continue
        args = str(row["args"])
        elapsed = int(row["etimes"])
        is_train_process = "train.py" in args and (" --spec " in args or spec_text in args or unit_text in args)
        if not is_train_process or elapsed <= hard_timeout_seconds:
            continue
        append_watchdog_event(
            log_path,
            {
                "event": "kill_over_budget_trial",
                "pid": pid,
                "elapsed_seconds": elapsed,
                "hard_timeout_seconds": hard_timeout_seconds,
                "args": args,
                "time_unix": int(time.time()),
            },
        )
        terminate_tree(pid)


def render_program(
    *,
    stage: Literal["rm", "rl"],
    objective_name: str,
    unit_relpath: Path,
    unit_dir: Path,
    prepare_script: Path,
    train_script: Path,
    spec_json: Path,
    results_tsv: Path,
    best_summary_json: Path,
    best_bundle_json: Path | None,
    best_checkpoint: Path | None,
    logs_dir: Path,
) -> str:
    objective_desc = "validation Spearman (`validation_spearman_rho`)" if stage == "rm" else f"validation reward (`{objective_name}`)"
    trial_time_budget_seconds, trial_timeout_minutes = trial_budget_for_stage(stage)
    trial_time_budget_minutes = trial_time_budget_seconds // 60
    optional_paths = ""
    if best_bundle_json is not None:
        optional_paths += f"- best reward bundle JSON: `{best_bundle_json}`\n"
    if best_checkpoint is not None:
        optional_paths += f"- best RL checkpoint: `{best_checkpoint}`\n"

    stage_rules = ""
    data_rules = ""
    if stage == "rm":
        data_rules = """
Dataset split rules:

- `spec.dataset` points to the fixed training split (`train.csv`) prepared before this stage.
- `spec.validation_dataset` points to the only fixed held-out split for this stage (`validation.csv`).
- Do not train on validation examples or copy validation labels into train-derived artifacts.
- You may evaluate and tune against this fixed validation split repeatedly; there is no separate test split in this RM stage.
- Treat `validation_spearman_rho` as the score to maximize.
"""
        stage_rules = """
Stage-specific modeling constraints:

- The reward model representation must come only from the raw sequence itself and sequence-derived statistics available inside this run.
- Do not use embeddings, hidden states, logits, scores, or fused features from any external or pretrained model.
- Do not call external model services or add dependencies on outside checkpoints to build sequence representations.
"""
    if stage == "rl":
        data_rules = """
Prompt split rules:

- `spec.train_prompt_fasta` is the fixed UniRef training prompt split prepared before this stage.
- `spec.validation_prompt_fasta` is the fixed UniRef validation prompt split prepared before this stage.
- Train/update the policy only on `spec.train_prompt_fasta`.
- Do not train, update gradients, or replay optimization batches from validation prompts.
- You may evaluate and tune against the fixed validation prompts repeatedly; there is no separate test prompt split in this RL stage.
- The final score for this stage is the fixed validation-prompt score printed by `train.py`.
- Do not modify either fixed prompt split file.
"""
        stage_rules = """
Stage-specific optimization constraints:

- You must train from the provided init policy checkpoint in `spec.init_checkpoint`; do not replace it with random initialization or another starting checkpoint.
- Keep the training algorithm in the current GRPO-style family: grouped sampling, group-relative/normalized advantages, and policy-gradient optimization of the provided reward model.
- Do not replace this stage with supervised finetuning, pure behavioral cloning, PPO, DPO, or another optimization objective family.
- The reward model at `spec.reward_bundle_json` is fixed input for this stage and must be treated as read-only.
- Treat these RL resource settings as fixed: `spec.max_aa_len = 512`, `spec.prompt_batch_size = 72`, `spec.group_size = 4`, `spec.policy_mini_batch_size = 64`, `spec.validation_prompt_batch_size = 300`, and `spec.validation_group_size = 4`.
- Do not bypass or reinterpret those fixed resource settings inside `train.py`.
"""

    trial_runs_dir = spec_json.parent / "trial_runs"
    extra_artifacts = ""
    if stage == "rm":
        extra_artifacts += f"""The RM train script also writes:

```text
{trial_runs_dir}/<trial_name>/reward_bundle.json
```
"""
    if stage == "rl":
        extra_artifacts += f"""The RL train script also writes:

```text
{trial_runs_dir}/<trial_name>/best.pt
```
"""

    promotion_block = f"""```bash
TRIAL_DIR={quoted(trial_runs_dir)}/${{TRIAL_NAME}}
cp "$TRIAL_DIR/summary.json" {quoted(best_summary_json)}
```
"""
    if best_bundle_json is not None:
        promotion_block += f"""
```bash
mkdir -p {quoted(best_bundle_json.parent)}
cp "$TRIAL_DIR/reward_bundle.json" {quoted(best_bundle_json)}
```
"""
    if best_checkpoint is not None:
        promotion_block += f"""
```bash
cp "$TRIAL_DIR/best.pt" {quoted(best_checkpoint)}
```
"""

    return f"""# Karpathy-Style {stage.upper()} Autoresearch

This unit follows the original autoresearch split:

- `program.md` is human-owned for this run
- `prepare.py` is fixed and must not be edited
- `train.py` is the only file you may edit

Goal:

- maximize {objective_desc}

What you CAN do:

- Modify only `{unit_relpath.as_posix()}/train.py`

What you CANNOT do:

- Modify `program.md`
- Modify `prepare.py`
- Modify any parent directory file
- Modify any sibling file
- Modify dataset files
- Modify the external spec JSON

{data_rules}

{stage_rules}

Fixed paths for this run:

- unit dir: `{unit_dir}`
- fixed prepare script: `{prepare_script}`
- train entrypoint: `{train_script}`
- spec JSON: `{spec_json}`
- results TSV: `{results_tsv}`
- best summary JSON: `{best_summary_json}`
- logs dir: `{logs_dir}`
{optional_paths}

The fixed evaluation/time budget lives in `prepare.py`.
`TIME_BUDGET = {trial_time_budget_seconds}` means the training code itself should stop after about {trial_time_budget_minutes} minutes.
If a run exceeds about {trial_timeout_minutes} minutes total, kill it and treat it as a failure.

Hard SOP:

- Follow this document exactly; it is the experiment protocol, not a suggestion.
- The fixed train/validation split, fixed prompt split, reward model, spec JSON, and metric are the only valid evaluation contract.
- Do not create a new validation split or hidden scratch split.
- Do not train on validation records or validation prompts.
- Do not run private grid search, random search, scratch validation search, or multi-trial sweeps outside the loop below.
- Do not call `prepare.py` with modified arguments to change the dataset, split, metric, budget, batch sizes, or prompt sets.
- Do not change `results.tsv` except appending one row per completed trial.
- Each trial must correspond to exactly one committed change to `train.py`, or to the baseline before any change.
- Every trial must use the fixed command pattern below. The parent worker also runs an external watchdog and will terminate over-budget `train.py` descendants.
- Trials must run in the foreground as direct descendants of the current `codex exec` process. Do not use background jobs, `&`, `nohup`, `disown`, `setsid`, daemon processes, detached shells, or any wrapper that lets `train.py` continue after `codex exec` exits.
- If you need to monitor a long trial, wait in the same foreground shell command until it finishes or is killed by the timeout; do not launch it asynchronously and poll it from later commands.

Setup rules:

1. Read `program.md`, `prepare.py`, and `train.py`.
2. Verify `results.tsv` exists. If not, create it with exactly:

```text
timestamp_utc\tcommit\tscore\tmemory_gb\tstatus\tdescription
```

3. The first run in this branch must be the baseline with the current `train.py`.

Fixed trial command pattern:

```bash
cd {quoted(unit_dir)}
mkdir -p {quoted(logs_dir)}
TIMEOUT_BIN=timeout
if ! command -v "$TIMEOUT_BIN" >/dev/null 2>&1; then
  if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN=gtimeout
  else
    echo "Neither timeout nor gtimeout is available on PATH." >&2
    exit 1
  fi
fi
TRIAL_NAME="<your_trial_name>"
LOG_PATH={quoted(logs_dir)}/"${{TRIAL_NAME}}.log"
TRIAL_EXIT_CODE=0
"$TIMEOUT_BIN" {trial_timeout_minutes}m python train.py --spec {quoted(spec_json)} --trial-name "${{TRIAL_NAME}}" > "${{LOG_PATH}}" 2>&1 || TRIAL_EXIT_CODE=$?
```

The train script prints summary lines like:

```text
score: 0.123456
peak_vram_mb: 1234.5
```

The train script writes:

```text
{trial_runs_dir}/<trial_name>/summary.json
```

The trial command must stay in the foreground until `train.py` exits. A trial that is backgrounded or detached is invalid, even if it later writes a score.

{extra_artifacts}

Logging rules:

- `timestamp_utc` must be the UTC update time in ISO 8601 format, e.g. `2026-04-21T09:30:00Z`
- `score` is higher-is-better
- `memory_gb` is `peak_vram_mb / 1024`, rounded to one decimal
- allowed `status`: `keep`, `discard`, `crash`
- do not commit `results.tsv`
- append TSV rows with 6 tab-separated columns:
  `timestamp_utc commit score memory_gb status description`

Experiment loop:

1. Look at the current git state and current best score only from `results.tsv`.
2. Set `START_COMMIT=$(git rev-parse --short HEAD)`.
3. For the first row, run the baseline without changing `train.py`; after that, make one focused change in `train.py`.
4. `git add -- {unit_relpath.as_posix()}/train.py` and make one non-interactive commit for each non-baseline trial.
5. Run exactly one trial and parse:
   - `TIMESTAMP_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")`
   - `SCORE=$(grep '^score:' "$LOG_PATH" | tail -n1 | awk '{{print $2}}')`
   - `PEAK_VRAM_MB=$(grep '^peak_vram_mb:' "$LOG_PATH" | tail -n1 | awk '{{print $2}}')`
   - `MEMORY_GB=$(python -c "import sys; print(f'{{float(sys.argv[1]) / 1024.0:.1f}}')" "${{PEAK_VRAM_MB:-0}}")`
   - `COMMIT=$(git rev-parse --short HEAD)`
6. If the run crashed, timed out, or `score` is missing:
   - inspect `tail -n 50 "$LOG_PATH"`
   - log a `crash`
   - `git reset --hard "$START_COMMIT"`
   - `git clean -fd`
7. If the score improved over the current best:
   - keep the commit
   - copy the canonical artifacts to the fixed best paths
   - log `keep`
8. Otherwise:
   - log `discard`
   - `git reset --hard "$START_COMMIT"`
   - `git clean -fd`

Logging example:

```bash
printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "$TIMESTAMP_UTC" "$COMMIT" "$SCORE" "$MEMORY_GB" "$STATUS" "$DESC" >> {quoted(results_tsv)}
```

Canonical promotion commands:

{promotion_block}

Never stop on your own. Continue until externally stopped.
"""


def commit_program(package_dir: Path, stage: Literal["rm", "rl"]) -> None:
    run_git(["add", "--", "program.md"], cwd=package_dir)
    proc_env = worker_env()
    git_bin = require_executable("git", proc_env)
    proc = subprocess.run(
        [git_bin, "-c", "user.name=Codex", "-c", "user.email=codex@openai.com", "commit", "-m", f"Configure {stage.upper()} worker program"],
        cwd=package_dir,
        text=True,
        capture_output=True,
        check=False,
        env=proc_env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to commit program.md in {package_dir}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def launch_codex(
    package_dir: Path,
    prompt_text: str,
    timeout_seconds: int | None,
    codex_model: str | None,
    *,
    unit_dir: Path,
    spec_json: Path,
    trial_timeout_minutes: int,
    watchdog_log: Path,
) -> tuple[int | None, bool]:
    env = worker_env()
    codex_bin = require_executable("codex", env)
    cmd = [codex_bin, "exec", "--dangerously-bypass-approvals-and-sandbox", "-C", str(package_dir), "-"]
    if codex_model:
        cmd[2:2] = ["-m", codex_model]
    proc = subprocess.Popen(cmd, cwd=package_dir, text=True, stdin=subprocess.PIPE, env=env)
    timed_out = False
    if proc.stdin is not None:
        try:
            proc.stdin.write(prompt_text)
            proc.stdin.close()
        except BrokenPipeError:
            pass
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    last_watchdog = 0.0
    hard_timeout_seconds = int(trial_timeout_minutes * 60)
    while proc.poll() is None:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            timed_out = True
            append_watchdog_event(
                watchdog_log,
                {
                    "event": "kill_stage_timeout",
                    "pid": proc.pid,
                    "stage_timeout_seconds": timeout_seconds,
                    "time_unix": int(time.time()),
                },
            )
            terminate_tree(proc.pid)
            break
        if now - last_watchdog >= WATCHDOG_INTERVAL_SECONDS:
            enforce_trial_watchdog(
                root_pid=proc.pid,
                hard_timeout_seconds=hard_timeout_seconds,
                unit_dir=unit_dir,
                spec_json=spec_json,
                log_path=watchdog_log,
            )
            last_watchdog = now
        time.sleep(1.0)
    proc.wait()
    return proc.returncode, timed_out


def launch_karpathy_stage(
    *,
    stage: Literal["rm", "rl"],
    spec_payload: dict[str, object],
    stage_dir: Path,
    codex_model: str | None = None,
    stage_timeout_seconds: int | None = None,
    base_ref: str = "HEAD",
    keep_worktree: bool = True,
) -> KarpathyStageResult:
    git_root = resolve_git_root()
    rel_package = package_relpath(git_root)
    branch_name, worktree_root = create_worktree(git_root=git_root, stage_dir=stage_dir, stage=stage, base_ref=base_ref)
    package_dir = worktree_package_dir(worktree_root, rel_package)

    unit_relpath = Path("rm_autoresearch") if stage == "rm" else Path("rl_autoresearch")
    unit_dir = package_dir / unit_relpath
    spec_json = stage_dir / f"{stage}_spec.json"
    results_tsv = stage_dir / "results.tsv"
    best_summary_json = stage_dir / "best_summary.json"
    best_bundle_json = stage_dir / "best_reward_bundle.json"
    best_checkpoint = stage_dir / "best_rl.pt"
    logs_dir = stage_dir / "logs"
    _trial_time_budget_seconds, trial_timeout_minutes = trial_budget_for_stage(stage)
    watchdog_log = stage_dir / "watchdog_events.jsonl"
    objective_name = "validation_spearman_rho" if stage == "rm" else str(spec_payload.get("best_metric_key", "sample_reward_mean"))

    stage_dir.mkdir(parents=True, exist_ok=True)
    save_json(spec_json, spec_payload)

    program_md = package_dir / "program.md"
    program_md.write_text(
        render_program(
            stage=stage,
            objective_name=objective_name,
            unit_relpath=unit_relpath,
            unit_dir=unit_dir,
            prepare_script=unit_dir / "prepare.py",
            train_script=unit_dir / "train.py",
            spec_json=spec_json,
            results_tsv=results_tsv,
            best_summary_json=best_summary_json,
            best_bundle_json=best_bundle_json if stage == "rm" else None,
            best_checkpoint=best_checkpoint if stage == "rl" else None,
            logs_dir=logs_dir,
        ),
        encoding="utf-8",
    )
    commit_program(package_dir=package_dir, stage=stage)

    started_at = int(time.time())
    attempts: list[dict[str, object]] = []
    returncode: int | None = None
    timed_out = False
    deadline = time.time() + stage_timeout_seconds if stage_timeout_seconds is not None else None
    attempt_index = 0
    while True:
        attempt_index += 1
        attempt_started = int(time.time())
        if deadline is None:
            attempt_timeout = None
        else:
            remaining = max(1, int(deadline - time.time()))
            if remaining <= 1 and time.time() >= deadline:
                timed_out = True
                break
            attempt_timeout = remaining
        prompt_text = "Read program.md in the current directory and start the autonomous research loop now."
        if attempt_index > 1:
            prompt_text = (
                "Read program.md in the current directory and continue the autonomous research loop from the "
                "current worktree. First inspect existing results.tsv, logs, and trial_runs; reconcile any "
                "completed but unlogged trial, promote the best valid artifact if needed, then continue one "
                "focused trial at a time until externally stopped."
            )
        returncode, attempt_timed_out = launch_codex(
            package_dir=package_dir,
            prompt_text=prompt_text,
            timeout_seconds=attempt_timeout,
            codex_model=codex_model,
            unit_dir=unit_dir,
            spec_json=spec_json,
            trial_timeout_minutes=trial_timeout_minutes,
            watchdog_log=watchdog_log,
        )
        attempts.append(
            {
                "attempt": attempt_index,
                "started_at_unix": attempt_started,
                "finished_at_unix": int(time.time()),
                "codex_returncode": returncode,
                "timed_out": attempt_timed_out,
            }
        )
        if attempt_timed_out:
            timed_out = True
            break
        if deadline is None:
            break
        if time.time() >= deadline:
            timed_out = True
            break
        if returncode != 0:
            append_watchdog_event(
                watchdog_log,
                {
                    "event": "restart_codex_after_nonzero_exit",
                    "attempt": attempt_index,
                    "codex_returncode": returncode,
                    "time_unix": int(time.time()),
                },
            )
        sleep_seconds = 60 if returncode != 0 else 5
        remaining_after_attempt = max(0.0, deadline - time.time())
        time.sleep(min(float(sleep_seconds), remaining_after_attempt))
    finished_at = int(time.time())

    result = KarpathyStageResult(
        stage=stage,
        objective_name=objective_name,
        branch_name=branch_name,
        worktree_dir=str(worktree_root),
        package_dir=str(package_dir),
        unit_dir=str(unit_dir),
        program_md=str(program_md),
        stage_dir=str(stage_dir),
        spec_json=str(spec_json),
        results_tsv=str(results_tsv),
        best_summary_json=str(best_summary_json),
        best_reward_bundle_json=str(best_bundle_json) if stage == "rm" else None,
        best_checkpoint=str(best_checkpoint) if stage == "rl" else None,
        codex_returncode=returncode,
        timed_out=timed_out,
        started_at_unix=started_at,
        finished_at_unix=finished_at,
        codex_attempts=attempts,
    )
    save_json(stage_dir / "worker_session.json", asdict(result))
    if not keep_worktree:
        remove_worktree(git_root=git_root, worktree_dir=worktree_root, branch_name=branch_name)
    return result
