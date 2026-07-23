# Executor Slimming + HPC/Workstation Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vizfold install` and `vizfold execute-run` work on an HPC cluster *or* a beefy workstation with live, unbuffered terminal output throughout, then remove ~2,400 lines of layering that does not pay for itself.

**Architecture:** One binary, one SQLite file. Cluster work runs through blocking `srun --pty` (never detached `sbatch`) so every step streams to the user's terminal. The `repositories/` pass-through layer is deleted in favour of SeaORM's active-record API. Run provenance moves onto the `runs` row as an immutable JSON snapshot. Seven migrations collapse to one baseline.

**Tech Stack:** Rust 2024 edition, SeaORM 1 (sqlx-sqlite), clap 4, tokio 1, axum 0.8 (retained for Project B), bash (the `install/` scripts).

## Global Constraints

- **Never commit to `main`.** All work is on branch `executor-slimming`. `main` is pull-only.
- **Never add `Co-Authored-By` lines** to commit messages.
- **Never merge PRs.** Human review only.
- Language in code, comments, and commit messages stays professional and neutral — these repos are public.
- Baseline before any change: `cargo test` = **103 passed, 0 failed**. Every task must end at or above this, with zero failures.
- Working directory for all `cargo` commands: `science-gateway/apps/executor`.
- `install/` bash is load-bearing cluster knowledge. Change only what a task names explicitly.
- Rust edition 2024 let-chains are already in use (`config.rs:11`); match the surrounding style.
- Comments stay minimal and match existing density. Do not add explanatory comments to code that does not have them.

---

## Execution order and spec mapping

Tasks are ordered by value and dependency, which differs from the spec's narrative PR numbering.

| Task | Spec PR | Why here |
|---|---|---|
| 1. Install streams live | PR2 (bash) | The primary goal. Pure bash, no Rust dependency. |
| 2. `srun` seam for folds | PR2 (Rust) | Completes "HPC or workstation" in behaviour. |
| 3. GPU preflight + defaults | PR1 | Workstation becomes first-class. |
| 4. Stream fold output | PR1 | Trait change; isolated so it can be reverted alone. |
| 5. Delete dead surface | PR3 | **Must precede Task 8** (`examples/` references `repositories::`). |
| 6. Baseline schema | PR7 | Introduces the column Task 7 needs. |
| 7. Provenance snapshot | PR5 | Depends on Task 6. |
| 8. Delete `repositories/` | PR4 | Depends on Task 5. |
| 9. Delete execution ceremony | PR6 | Last; touches the most call sites. |

## File Structure

**Created:**
- `install/tests/launch_args.sh` — bash assertions for the scheduler argv builder (Task 1)
- `src/core/migrations/m20260723_000001_create_schema.rs` — the single baseline migration (Task 6)

**Deleted:**
- `examples/` — all 4 files (Task 5)
- `src/core/repositories/` — all 8 files (Task 8)
- `src/core/execution.rs` (Task 9)
- `src/core/migrations/m2026070*.rs`, `m2026071*.rs` — all 7 (Task 6)

**Modified (primary responsibility):**
- `install/slurm.sh` — scheduler dispatch; gains `slurm::launch_args` (Task 1)
- `src/core/config.rs` — resolves the GPU launch prefix from `vizfold.json` (Task 2)
- `src/core/services/openfold_execution.rs` — applies srun + micromamba wrappers (Tasks 2, 9)
- `src/core/model_runners/openfold.rs` — preflight checks (Task 3)
- `src/core/commands.rs` — `CommandSpec`/`CommandRunner`, streaming (Task 4)
- `src/adapters/cli.rs` — arg defaults, provenance capture, inlined reads (Tasks 3, 7, 8)

---

## Task 1: Install streams live on a cluster

**Spec:** PR2 (bash half). **This is the project's primary goal.**

Today `slurm::run` submits with `sbatch --output=$PREFIX/install-%j.log` and returns immediately, so the user sees nothing. It becomes a blocking `srun --pty` that streams every step of `setup.sh`.

**Files:**
- Modify: `install/slurm.sh:50-84`
- Create: `install/tests/launch_args.sh`

**Interfaces:**
- Produces: `slurm::launch_args <account> <partition> <pty>` — echoes the scheduler argv, one argument per line. `<pty>` is the literal string `--pty` or empty.

**Why `--pty` matters:** without it `srun` hands the task a pipe, and `curl` suppresses its progress meter entirely, `micromamba` stops rendering progress, and `python` switches to 4–8 KB block buffering. The terminal would sit silent and then burst — exactly what this task exists to prevent. `-u` additionally disables srun's own line buffering.

- [ ] **Step 1: Write the failing test**

Create `install/tests/launch_args.sh`:

```bash
#!/bin/bash
# Assertions for slurm::launch_args. Run: bash install/tests/launch_args.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$(cd .. && pwd); export REPO
. ./slurm.sh

fail=0
check() {
    local want=$1 got=$2 name=$3
    if [ "$want" = "$got" ]; then
        echo "ok   $name"
    else
        echo "FAIL $name"; echo "  want: $want"; echo "  got:  $got"; fail=1
    fi
}

# Already inside an srun step: run in place, never nest srun.
got=$(SLURM_STEP_ID=0 SLURM_JOB_ID=1 slurm::launch_args acct part --pty | tr '\n' ' ')
check "bash " "$got" "step id means bash"

# Holding an allocation but on the submit host: a bare step is enough.
got=$(SLURM_JOB_ID=1 slurm::launch_args acct part --pty | tr '\n' ' ')
check "srun --ntasks=1 " "$got" "job id means plain srun"

# No allocation: full srun with resources, pty requested.
got=$(env -u SLURM_STEP_ID -u SLURM_JOB_ID slurm::launch_args acct part --pty | tr '\n' ' ')
want="srun -u --pty --job-name=vizfold-install --account=acct --partition=part --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --time=02:00:00 "
check "$want" "$got" "no allocation means full srun with pty"

# Not a terminal: identical but without --pty.
got=$(env -u SLURM_STEP_ID -u SLURM_JOB_ID slurm::launch_args acct part "" | tr '\n' ' ')
want="srun -u --job-name=vizfold-install --account=acct --partition=part --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --time=02:00:00 "
check "$want" "$got" "no tty means no pty"

# sbatch must be gone entirely.
grep -q sbatch ./slurm.sh && { echo "FAIL sbatch still referenced"; fail=1; } || echo "ok   no sbatch"

exit $fail
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `bash install/tests/launch_args.sh`
Expected: FAIL — `slurm::launch_args: command not found`, and `FAIL sbatch still referenced`.

- [ ] **Step 3: Add `slurm::launch_args` and rewrite `slurm::run`**

In `install/slurm.sh`, replace `slurm::run` (lines 50-84) with:

```bash
# Build the scheduler argv for setup.sh, one argument per line.
# $1 account, $2 partition, $3 the literal --pty (or empty when stdout is not a terminal).
slurm::launch_args() {
    if [ -n "${SLURM_STEP_ID:-}" ]; then
        printf '%s\n' bash                                   # already on the node
        return
    fi
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        printf '%s\n' srun --ntasks=1                        # salloc leaves you off it
        return
    fi
    printf '%s\n' srun -u
    [ -n "$3" ] && printf '%s\n' "$3"
    printf '%s\n' --job-name=vizfold-install "--account=$1" "--partition=$2" \
        --nodes=1 --ntasks=1 "--cpus-per-task=${OPENFOLD_BUILD_CPUS:-8}" \
        "--mem=${OPENFOLD_BUILD_MEM:-24G}" "--time=${OPENFOLD_BUILD_TIME:-02:00:00}"
    [ -n "${OPENFOLD_BUILD_GRES:-}" ] && printf '%s\n' "--gres=$OPENFOLD_BUILD_GRES"
    return 0
}

# Run the assembled hooks, then run setup.sh on the scheduler (or here when there is none).
slurm::run() {
    if [ -z "${SLURM_JOB_ID:-}" ] && ! command -v srun >/dev/null 2>&1; then
        exec bash "$REPO/install/setup.sh"          # no scheduler: install here
    fi
    local PREFIX ACCOUNT PARTITION SETUP PTY
    # OPENFOLD_PREFIX/ACCOUNT come pre-resolved: inline env, or <site>.json templates expanded off slurm::discover's vars.
    PREFIX=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "${OPENFOLD_PREFIX:-}")
    [ -n "$PREFIX" ] || die "no install prefix; set OPENFOLD_PREFIX or its <site>.json"
    ACCOUNT=$(interactive::resolve OPENFOLD_ACCOUNT "slurm account" "${OPENFOLD_ACCOUNT:-$(slurm::default_account)}")
    export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-${ACCOUNT:+$ACCOUNT${OPENFOLD_GPU_ACCOUNT_SUFFIX:-}}}
    export OPENFOLD_PREFIX=$PREFIX OPENFOLD_HOME=$REPO
    SETUP=$REPO/install/setup.sh
    mkdir -p "$PREFIX"

    if [ -z "${SLURM_STEP_ID:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
        [ -n "$ACCOUNT" ] || die "no slurm account; set OPENFOLD_ACCOUNT"
        PARTITION=$(interactive::resolve OPENFOLD_PARTITION "slurm partition" "${OPENFOLD_PARTITION:-}")
        [ -n "$PARTITION" ] || die "no build partition; set OPENFOLD_PARTITION or its <site>.json"
    fi

    # -t 1 must be tested here, not inside launch_args: command substitution makes stdout a pipe.
    PTY=; [ -t 1 ] && PTY=--pty
    local LAUNCH=()
    while IFS= read -r arg; do LAUNCH+=("$arg"); done < <(slurm::launch_args "$ACCOUNT" "${PARTITION:-}" "$PTY")
    echo "${LAUNCH[0]} $SETUP"
    exec "${LAUNCH[@]}" "$SETUP"
}
```

Three deliberate changes beyond the launcher: `--output=` and `--export=ALL` are dropped (both are `sbatch` concepts; `srun` streams to the terminal and propagates the environment by default), the no-scheduler probe at the top now tests for `srun` rather than `sbatch`, and `exec` is preserved so the exit code is exact.

- [ ] **Step 4: Run the test to verify it passes**

Run: `bash install/tests/launch_args.sh`
Expected: 5 lines beginning `ok`, exit 0.

- [ ] **Step 5: Confirm the no-scheduler path still works**

Run: `env -u SLURM_JOB_ID PATH=/usr/bin:/bin bash -c '. install/slurm.sh 2>/dev/null; type slurm::run' `
Expected: prints the function body — confirms the file still sources cleanly with no `srun` on PATH.

- [ ] **Step 6: Update the README**

In `README.md`, under the install section, add:

```markdown
`vizfold install` holds your terminal and streams every step of the install as it happens. On a
cluster it runs as a blocking `srun` job, so a queue wait shows as
`srun: job N queued and waiting for resources`. Use `tmux` or `screen` for long installs — if the
connection drops, re-run `vizfold install` and it continues from the last completed step.

To keep a log, wrap the whole command rather than piping it:

    script -q -e -c 'vizfold install' install.log

Do not pipe to `tee` — that replaces the terminal with a pipe, which suppresses download progress
meters and makes the output arrive in delayed bursts.
```

- [ ] **Step 7: Commit**

```bash
git add install/slurm.sh install/tests/launch_args.sh README.md
git commit -m "install: stream the install live via blocking srun

Replaces the detached sbatch submission, whose output went to a log file the
user never saw, with a blocking srun that holds the terminal. --pty gives the
remote task a pseudo-terminal so curl progress meters render and python stays
line-buffered rather than block-buffering into delayed bursts.

The tty test lives in slurm::run rather than slurm::launch_args because
command substitution makes stdout a pipe, which would suppress --pty always."
```

---

## Task 2: `srun` seam for folds

**Spec:** PR2 (Rust half).

`vizfold execute-run` currently always runs the fold as a local subprocess. It gains the same four-way scheduler context detection the bash uses.

**Files:**
- Modify: `src/core/config.rs` (add after `prefix()`, ~line 96)
- Modify: `src/core/services/openfold_execution.rs:51-68`

**Interfaces:**
- Consumes: `config::resolved()` (private, same module)
- Produces:
  - `config::SlurmContext { InStep, InAllocation, None }`
  - `config::gpu_launch(ctx: SlurmContext, partition: Option<&str>, account: Option<&str>, gres: Option<&str>, resources: Option<&str>, time: Option<&str>) -> Vec<String>` — pure, testable
  - `config::gpu_launch_args() -> Vec<String>` — thin env-reading wrapper
  - `openfold_execution::srun_command(command: CommandSpec, launch: &[String]) -> CommandSpec`

**Two gotchas this task must respect:**
1. `OPENFOLD_GPU_RESOURCES` holds *several* space-separated flags (`--cpus-per-task=8 --mem=32G`) and is deliberately word-split unquoted at `setup.sh:212`. Pushing it as one `String` gives `srun` a single bogus argument.
2. The srun wrapper goes **outside** `activate_env_command`, never inside. Micromamba activation must happen on the compute node; wrapping the other way activates the env on the submit host and ships an already-`exec`'d bash.

- [ ] **Step 1: Write the failing tests**

Append to `src/core/config.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::{SlurmContext, gpu_launch};

    #[test]
    fn in_step_runs_bare() {
        assert!(gpu_launch(SlurmContext::InStep, Some("gpuA100x4"), None, None, None, None).is_empty());
    }

    #[test]
    fn in_allocation_uses_a_plain_step() {
        assert_eq!(
            gpu_launch(SlurmContext::InAllocation, Some("gpuA100x4"), None, None, None, None),
            vec!["srun", "--ntasks=1"]
        );
    }

    #[test]
    fn no_partition_runs_bare() {
        assert!(gpu_launch(SlurmContext::None, None, Some("acct"), None, None, None).is_empty());
    }

    #[test]
    fn empty_partition_runs_bare() {
        assert!(gpu_launch(SlurmContext::None, Some(""), None, None, None, None).is_empty());
    }

    #[test]
    fn resources_word_split_into_separate_arguments() {
        assert_eq!(
            gpu_launch(
                SlurmContext::None,
                Some("gpuA100x4"),
                Some("bbkg-delta-gpu"),
                Some("gpu:a100:1"),
                Some("--cpus-per-task=8 --mem=32G"),
                Some("04:00:00"),
            ),
            vec![
                "srun", "-A", "bbkg-delta-gpu", "-p", "gpuA100x4", "--gres=gpu:a100:1",
                "--cpus-per-task=8", "--mem=32G", "-t", "04:00:00",
            ]
        );
    }

    #[test]
    fn defaults_match_the_installer() {
        assert_eq!(
            gpu_launch(SlurmContext::None, Some("gpu"), None, None, None, None),
            vec!["srun", "-p", "gpu", "--gres=gpu:1", "--cpus-per-task=8", "--mem=32G", "-t", "02:00:00"]
        );
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib config::tests`
Expected: FAIL — `cannot find function gpu_launch in this scope`.

- [ ] **Step 3: Implement in `src/core/config.rs`**

Add after `prefix()`:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlurmContext {
    InStep,
    InAllocation,
    None,
}

impl SlurmContext {
    pub fn detect() -> Self {
        if std::env::var_os("SLURM_STEP_ID").is_some() {
            Self::InStep
        } else if std::env::var_os("SLURM_JOB_ID").is_some() {
            Self::InAllocation
        } else {
            Self::None
        }
    }
}

/// SLURM launch prefix for a fold, mirroring `install/setup.sh:212`. Empty means run bare —
/// either we are already on the node, or no GPU partition is configured (the workstation case).
pub fn gpu_launch(
    context: SlurmContext,
    partition: Option<&str>,
    account: Option<&str>,
    gres: Option<&str>,
    resources: Option<&str>,
    time: Option<&str>,
) -> Vec<String> {
    match context {
        SlurmContext::InStep => return Vec::new(),
        SlurmContext::InAllocation => return vec!["srun".to_owned(), "--ntasks=1".to_owned()],
        SlurmContext::None => {}
    }
    let partition = partition.filter(|p| !p.is_empty());
    let Some(partition) = partition else {
        return Vec::new();
    };
    let mut args = vec!["srun".to_owned()];
    if let Some(account) = account.filter(|a| !a.is_empty()) {
        args.push("-A".to_owned());
        args.push(account.to_owned());
    }
    args.push("-p".to_owned());
    args.push(partition.to_owned());
    args.push(format!("--gres={}", gres.unwrap_or("gpu:1")));
    // Holds several space-separated flags and must split, as setup.sh:212 relies on word splitting.
    args.extend(
        resources
            .unwrap_or("--cpus-per-task=8 --mem=32G")
            .split_whitespace()
            .map(str::to_owned),
    );
    args.push("-t".to_owned());
    args.push(time.unwrap_or("02:00:00").to_owned());
    args
}

pub fn gpu_launch_args() -> Vec<String> {
    gpu_launch(
        SlurmContext::detect(),
        resolved("OPENFOLD_GPU_PARTITION").as_deref(),
        resolved("OPENFOLD_GPU_ACCOUNT").as_deref(),
        resolved("OPENFOLD_GPU_GRES").as_deref(),
        resolved("OPENFOLD_GPU_RESOURCES").as_deref(),
        resolved("OPENFOLD_GPU_TIME").as_deref(),
    )
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --lib config::tests`
Expected: PASS, 6 tests.

- [ ] **Step 5: Write the wrapper test**

In `src/core/services/openfold_execution.rs`, add to the existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn srun_command_wraps_the_whole_activated_command() {
    let inner = CommandSpec {
        program: "bash".into(),
        args: vec!["-c".into(), "script".into()],
        current_dir: Some(PathBuf::from("/repo")),
        ..Default::default()
    };
    let wrapped = super::srun_command(inner, &["srun".to_owned(), "-p".to_owned(), "gpu".to_owned()]);

    assert_eq!(wrapped.program, "srun");
    assert_eq!(wrapped.args, vec!["-p", "gpu", "bash", "-c", "script"]);
    assert_eq!(wrapped.current_dir, Some(PathBuf::from("/repo")));
}

#[test]
fn srun_command_is_a_no_op_without_a_launch_prefix() {
    let inner = CommandSpec { program: "python3".into(), ..Default::default() };
    assert_eq!(super::srun_command(inner.clone(), &[]), inner);
}
```

- [ ] **Step 6: Run to verify it fails**

Run: `cargo test --lib openfold_execution::tests::srun_command`
Expected: FAIL — `cannot find function srun_command`.

- [ ] **Step 7: Implement and apply the wrapper**

Add to `src/core/services/openfold_execution.rs`, next to `activate_env_command`:

```rust
/// Prefix a command with the SLURM launcher. Applied *outside* the micromamba wrapper so the
/// environment is activated on the compute node, not the submit host.
fn srun_command(command: CommandSpec, launch: &[String]) -> CommandSpec {
    let Some((program, prefix)) = launch.split_first() else {
        return command;
    };
    let mut args = prefix.to_vec();
    args.push(command.program);
    args.extend(command.args);
    CommandSpec {
        program: program.clone(),
        args,
        current_dir: command.current_dir,
        env: command.env,
    }
}
```

Then change the execution block at `openfold_execution.rs:62-67` from:

```rust
        let exec_command = if prefix.join("bin/micromamba").is_file() {
            activate_env_command(&command, &prefix, &config::openfold_env_prefix())
        } else {
            command.clone()
        };
```

to:

```rust
        let exec_command = if prefix.join("bin/micromamba").is_file() {
            activate_env_command(&command, &prefix, &config::openfold_env_prefix())
        } else {
            command.clone()
        };
        let exec_command = srun_command(exec_command, &config::gpu_launch_args());
```

- [ ] **Step 8: Run the full suite**

Run: `cargo test`
Expected: PASS, 0 failed (count rises by ~8 from the 103 baseline).

- [ ] **Step 9: Commit**

```bash
git add src/core/config.rs src/core/services/openfold_execution.rs
git commit -m "executor: run folds through srun when a GPU partition is configured

Mirrors install/slurm.sh's context detection: bare inside an existing step,
a plain step when holding an allocation, a full srun with resources otherwise,
and bare when no GPU partition is configured, which is the workstation case.

The prefix wraps outside the micromamba activation so the environment is
activated on the compute node. GPU resources are word-split because the
setting holds several flags and setup.sh relies on that splitting."
```

---

## Task 3: GPU preflight and workstation defaults

**Spec:** PR1 (checks and defaults).

**Files:**
- Modify: `src/core/model_runners/openfold.rs:87-175` (`preflight_openfold`)
- Modify: `src/adapters/cli.rs:123-126` (clap defaults)

**Interfaces:**
- Produces: `openfold::gpu_check(detected: Option<&str>) -> PreflightCheck`, `openfold::detect_gpu() -> Option<String>`, `cli::default_model_device() -> String`, `cli::default_cpus() -> i64`

`run/fold.sh:44-46` is the reference: `nvidia-smi --query-gpu=name --format=csv,noheader || die`. The Rust check **warns** instead of failing, because `model_device` now degrades to `cpu` — a CPU-only workstation should run slowly, not error.

- [ ] **Step 1: Write the failing tests**

Add to `src/core/model_runners/openfold.rs`'s test module:

```rust
#[test]
fn gpu_check_passes_when_a_gpu_is_visible() {
    let check = super::gpu_check(Some("NVIDIA A100-SXM4-40GB"));
    assert_eq!(check.status, PreflightStatus::Passed);
    assert!(check.message.unwrap().contains("A100"));
}

#[test]
fn gpu_check_warns_when_no_gpu_is_visible() {
    let check = super::gpu_check(None);
    assert_eq!(check.status, PreflightStatus::Warning);
    assert!(check.message.unwrap().contains("no GPU visible"));
}
```

Add to `src/adapters/cli.rs`'s test module:

```rust
#[test]
fn model_device_defaults_to_cpu_without_a_gpu() {
    assert_eq!(super::model_device_for(None), "cpu");
}

#[test]
fn model_device_defaults_to_cuda_with_a_gpu() {
    assert_eq!(super::model_device_for(Some("NVIDIA A100")), "cuda:0");
}

#[test]
fn cpus_default_follows_available_parallelism() {
    let expected = std::thread::available_parallelism().map_or(1, |n| n.get() as i64);
    assert_eq!(super::default_cpus(), expected);
    assert!(super::default_cpus() > 1, "a dev machine should report more than one core");
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib gpu_check ; cargo test --lib model_device`
Expected: FAIL — `cannot find function gpu_check` / `model_device_for`.

- [ ] **Step 3: Implement the check**

In `src/core/model_runners/openfold.rs`, add:

```rust
/// Mirrors run/fold.sh's `nvidia-smi --query-gpu=name --format=csv,noheader` probe.
pub fn detect_gpu() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    let name = String::from_utf8_lossy(&output.stdout).lines().next()?.trim().to_owned();
    (output.status.success() && !name.is_empty()).then_some(name)
}

pub fn gpu_check(detected: Option<&str>) -> PreflightCheck {
    match detected {
        Some(name) => PreflightCheck::passed("gpu", format!("GPU visible: {name}")),
        None => PreflightCheck::warning("gpu", "no GPU visible; the run will fall back to CPU"),
    }
}
```

Then add it to `preflight_openfold`'s check list, immediately after the `let mut checks = Vec::new();` at line 96:

```rust
    checks.push(gpu_check(detect_gpu().as_deref()));
```

- [ ] **Step 4: Implement the defaults**

In `src/adapters/cli.rs`, add:

```rust
fn model_device_for(detected: Option<&str>) -> String {
    if detected.is_some() { "cuda:0".to_owned() } else { "cpu".to_owned() }
}

fn default_model_device() -> String {
    model_device_for(crate::core::model_runners::openfold::detect_gpu().as_deref())
}

fn default_cpus() -> i64 {
    std::thread::available_parallelism().map_or(1, |n| n.get() as i64)
}
```

Change the clap attributes at `cli.rs:123-126` from:

```rust
    #[arg(long, default_value = "cuda:0")]
    model_device: String,
    #[arg(long, default_value_t = 1)]
    cpus: i64,
```

to:

```rust
    #[arg(long, default_value_t = default_model_device())]
    model_device: String,
    #[arg(long, default_value_t = default_cpus())]
    cpus: i64,
```

- [ ] **Step 5: Run to verify they pass**

Run: `cargo test`
Expected: PASS, 0 failed.

- [ ] **Step 6: Verify the CLI help renders the resolved defaults**

Run: `cargo run -q --bin vizfold -- queue-run openfold --help | grep -E 'model-device|cpus'`
Expected: shows `[default: cpu]` (or `cuda:0` on a GPU host) and `[default: <core count>]`.

- [ ] **Step 7: Commit**

```bash
git add src/core/model_runners/openfold.rs src/adapters/cli.rs
git commit -m "executor: detect the GPU and default the device and cpu count from the host

preflight gains an nvidia-smi probe mirroring run/fold.sh. It warns rather
than failing, because model_device now degrades to cpu when no GPU is visible,
so a CPU-only workstation runs slowly instead of erroring.

cpus defaulted to 1, which single-threaded every run on a workstation; it now
follows available_parallelism."
```

---

## Task 4: Stream fold output

**Spec:** PR1 (streaming).

A multi-hour fold currently prints nothing until it exits, because `commands.rs:39` buffers into `String`s and `cli.rs:560-565` prints only afterwards.

**Files:**
- Modify: `src/core/commands.rs:5-52`
- Modify: `src/core/services/openfold_execution.rs` (set the flag)

**Interfaces:**
- Produces: `CommandSpec.stream: bool` (defaults `false`, so every existing construction is unchanged)

**Accepted trade-off:** when streaming, stderr is not captured, so the failure message falls back to the existing `"OpenFold command exited with code {n}"` at `openfold_execution.rs:96`. The user saw the error live; exit code is all `openfold_execution.rs:82` branches on.

- [ ] **Step 1: Write the failing test**

Add to `src/core/commands.rs`'s test module:

```rust
#[tokio::test]
async fn streaming_returns_the_exit_code_without_capturing_output() {
    let runner = LocalCommandRunner;
    #[cfg(unix)]
    let mut spec = shell_command("printf visible; exit 3");
    #[cfg(windows)]
    let mut spec = shell_command("echo visible & exit /B 3");
    spec.stream = true;

    let output = runner.run(spec).await.expect("command should run");

    assert_eq!(output.exit_code, 3);
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib commands::tests::streaming`
Expected: FAIL — `no field stream on type CommandSpec`.

- [ ] **Step 3: Implement**

In `src/core/commands.rs`, add the field to `CommandSpec`:

```rust
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommandSpec {
    pub program: String,
    pub args: Vec<String>,
    pub current_dir: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    /// Inherit the parent's stdio instead of capturing it, so a long run reports progress live.
    pub stream: bool,
}
```

Replace the body of `LocalCommandRunner::run` (lines 30-51):

```rust
    async fn run(&self, spec: CommandSpec) -> Result<CommandOutput, DbErr> {
        let mut command = tokio::process::Command::new(&spec.program);
        command.args(&spec.args);
        command.envs(&spec.env);

        if let Some(current_dir) = &spec.current_dir {
            command.current_dir(current_dir);
        }

        let spawn_error = |error: std::io::Error| {
            DbErr::Custom(format!("failed to spawn command '{}': {}", spec.program, error))
        };

        if spec.stream {
            let status = command.status().await.map_err(spawn_error)?;
            return Ok(CommandOutput {
                exit_code: status.code().unwrap_or(-1),
                stdout: String::new(),
                stderr: String::new(),
            });
        }

        let output = command.output().await.map_err(spawn_error)?;

        Ok(CommandOutput {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
```

- [ ] **Step 4: Set the flag for real runs**

In `src/core/services/openfold_execution.rs`, immediately after the `srun_command` line added in Task 2:

```rust
        let exec_command = CommandSpec { stream: true, ..exec_command };
```

- [ ] **Step 5: Run to verify**

Run: `cargo test`
Expected: PASS, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add src/core/commands.rs src/core/services/openfold_execution.rs
git commit -m "executor: stream fold output instead of buffering it until exit

A multi-hour fold printed nothing until the process ended, then dumped
everything at once. Streaming runs inherit stdio, matching what run_install
already does.

Streaming does not capture stderr, so the failure message falls back to the
exit code that openfold_execution already branches on. The user has seen the
error live by that point."
```

---

## Task 5: Delete the dead surface

**Spec:** PR3. **Must land before Task 8.**

**Files:**
- Delete: `examples/run_openfold_workflow.rs`, `examples/plan_openfold_command.rs`, `examples/register_openfold_demo_artifacts.rs`, `examples/run_local_command.rs`
- Modify: `src/core/preflight.rs`, `src/core/config.rs`, `src/core/seed.rs`, `src/core/tests.rs`, `Cargo.toml`
- Modify: `science-gateway/DEMO.md`, `science-gateway/openfold-demo/ENVIRONMENT.md`

- [ ] **Step 1: Delete the examples and their documentation**

```bash
git rm -r examples/
```

Then remove these documentation sections, which reference the deleted binaries:
- `science-gateway/DEMO.md` — sections `## 3. Run the Rust executor example` (line 67) through the end of `## 4. What the example does` (line 109)
- `science-gateway/openfold-demo/ENVIRONMENT.md` — `## 5. Run the OpenFold workflow example` (heading line 59, through line 80) and `## 7. Register known demo artifacts` (heading line 112, through line 122)

Renumber the remaining headings in each file.

- [ ] **Step 2: Delete the dead accessors**

In `src/core/preflight.rs`, delete `PreflightReport::passed()` (lines 63-68) and `PreflightReport::warnings()` (lines 70-75).

**Do not touch `PreflightCheck::passed()` at line 18** — that is a live constructor with the same name on a different type.

Then fix the two tests that call the deleted accessors, at `preflight.rs:113-127` and `:129-137`:

```rust
    #[test]
    fn helpers_return_checks_matching_their_status() {
        let report = PreflightReport::new(vec![
            PreflightCheck::passed("workspace", "ready"),
            PreflightCheck::warning("cuda", "unavailable"),
            PreflightCheck::failed("python", "not found"),
        ]);

        assert_eq!(report.failures().len(), 1);
        assert_eq!(report.failures()[0].status, PreflightStatus::Failed);
        assert!(report.has_failures());
    }

    #[test]
    fn empty_report_has_no_failures_or_checks() {
        let report = PreflightReport::default();

        assert!(!report.has_failures());
        assert!(report.failures().is_empty());
        assert!(report.checks.is_empty());
    }
```

- [ ] **Step 3: Trim config and Cargo.toml**

In `src/core/config.rs`: delete `pub const DEFAULT_DATABASE_URL` (line 5), and change `pub fn repository_root()` (line 132) to `fn repository_root()`. **Do not delete the function** — it is the `unwrap_or_else` fallback for `openfold_home()` at line 57.

In `Cargo.toml`, add to `[package]`:

```toml
default-run = "vizfold"
```

- [ ] **Step 4: Remove the unreachable mock seed rows**

In `src/core/seed.rs`, delete the `local-mock` execution target block (lines 163-181) and the `mock` invocation profile block (lines 205-235). These are unreachable: nothing names `local-mock` in production, and `openfold.rs:461` rejects any profile whose `invocation_kind` is not `local_subprocess`.

Then update the tests that assert their presence — `src/core/tests.rs:87, :127, :148, :175, :195` — and rename the test at `tests.rs:80` from `seeds_local_openfold_target_and_profile_without_removing_mock_seed` to `seeds_local_openfold_target_and_profile`, dropping its mock assertions.

**Do not touch `src/core/output_locations.rs:54`**, which uses `invocation_kind: "mock"` in its own unrelated inline fixture.

- [ ] **Step 5: Verify**

Run: `cargo test`
Expected: PASS, 0 failed. Test count drops (the seed assertions shrink).

Run: `cargo run -q --bin vizfold -- --help | head -3`
Expected: prints the vizfold CLI help, not the axum health stub — confirms `default-run`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "executor: delete dead examples, accessors, and unreachable seed rows

The four cargo examples were referenced only by two documentation files and
never built by CI. PreflightReport::passed and ::warnings had no non-test
callers; PreflightCheck::passed, which shares a name on a different type,
stays. The local-mock target and mock profile could never execute, because the
planner rejects any invocation_kind other than local_subprocess.

Adds default-run so bare cargo run starts the CLI rather than the health stub."
```

---

## Task 6: Baseline schema

**Spec:** PR7.

Seven migrations become one. The project is pre-production, so there is no history to preserve: `m20260717_000005` already drops and rebuilds the artifacts table that `m20260707_000003` created, and every `down()` is dead because rolling back means deleting the SQLite file.

**Files:**
- Create: `src/core/migrations/m20260723_000001_create_schema.rs`
- Delete: all 7 existing `m2026*.rs`
- Modify: `src/core/migrations/mod.rs`
- Modify: `src/core/entities/runs.rs`

The baseline is the schema verified from a migrated database, **plus** `runs.provenance_json` for Task 7.

- [ ] **Step 1: Write the failing test**

Add to `src/core/tests.rs`:

```rust
#[tokio::test]
async fn baseline_schema_creates_every_table_including_provenance() {
    let db = db::connect_with_url("sqlite::memory:").await.expect("db");

    let tables: Vec<String> = db
        .query_all(Statement::from_string(
            DatabaseBackend::Sqlite,
            "select name from sqlite_master where type='table' order by name".to_owned(),
        ))
        .await
        .expect("query")
        .iter()
        .map(|row| row.try_get::<String>("", "name").expect("name"))
        .collect();

    for expected in [
        "artifact_types", "artifacts", "execution_targets",
        "model_backends", "model_invocation_profiles", "runs",
    ] {
        assert!(tables.iter().any(|t| t == expected), "missing table {expected}");
    }

    let columns: Vec<String> = db
        .query_all(Statement::from_string(
            DatabaseBackend::Sqlite,
            "select name from pragma_table_info('runs')".to_owned(),
        ))
        .await
        .expect("query")
        .iter()
        .map(|row| row.try_get::<String>("", "name").expect("name"))
        .collect();

    assert!(columns.iter().any(|c| c == "provenance_json"));
}
```

Add the imports this needs at the top of `tests.rs`: `use sea_orm::{ConnectionTrait, DatabaseBackend, Statement};`

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib baseline_schema`
Expected: FAIL — assertion on `provenance_json`.

- [ ] **Step 3: Write the baseline migration**

Create `src/core/migrations/m20260723_000001_create_schema.rs`. It must produce exactly this schema (verified by dumping a migrated database), with `provenance_json` added to `runs`:

```
model_backends(id PK AI, slug varchar NOT NULL UNIQUE, label varchar NOT NULL, version varchar,
    description text, artifact_capabilities_json text NOT NULL, parameter_schema_json text NOT NULL,
    created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP)

execution_targets(id PK AI, slug varchar NOT NULL UNIQUE, target_type varchar NOT NULL,
    description text, available_resources_json text NOT NULL,
    created_at ..., updated_at ...)

model_invocation_profiles(id PK AI, model_backend_id int NOT NULL, execution_target_id int NOT NULL,
    invocation_kind varchar NOT NULL, config_json text NOT NULL, created_at ..., updated_at ...,
    FK model_backend_id -> model_backends(id) ON DELETE RESTRICT ON UPDATE CASCADE,
    FK execution_target_id -> execution_targets(id) ON DELETE RESTRICT ON UPDATE CASCADE)

runs(id PK AI, model_backend_id int NOT NULL, execution_target_id int NOT NULL,
    invocation_profile_id int NOT NULL, status varchar NOT NULL, input_id varchar NOT NULL DEFAULT '',
    input_sequence text NOT NULL, model_parameters_json text NOT NULL,
    execution_parameters_json text NOT NULL, provenance_json text NULL,
    submitted_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP, started_at timestamp NULL,
    completed_at timestamp NULL, error_message text NULL,
    FK model_backend_id -> model_backends(id) ON DELETE RESTRICT ON UPDATE CASCADE,
    FK execution_target_id -> execution_targets(id) ON DELETE RESTRICT ON UPDATE CASCADE,
    FK invocation_profile_id -> model_invocation_profiles(id) ON DELETE RESTRICT ON UPDATE CASCADE)

artifact_types(id PK AI, slug varchar NOT NULL UNIQUE, label varchar NOT NULL,
    default_format varchar NOT NULL, display_mode varchar NOT NULL, viewer_kind varchar NOT NULL,
    description text NOT NULL, metadata_schema_json text NOT NULL DEFAULT '{}',
    created_at ..., updated_at ...)

artifacts(id PK AI, run_id int NOT NULL, artifact_type_id int NOT NULL, format varchar NOT NULL,
    storage_uri text NOT NULL, metadata_json text NOT NULL, created_at ...,
    FK run_id -> runs(id) ON DELETE CASCADE ON UPDATE CASCADE,
    FK artifact_type_id -> artifact_types(id) ON DELETE RESTRICT ON UPDATE CASCADE)
```

Create tables in FK order: `model_backends`, `execution_targets`, `model_invocation_profiles`, `runs`, `artifact_types`, `artifacts`. Copy the `DeriveIden` enums and `ColumnDef` idiom from the existing `m20260707_000001_create_model_backends_table.rs` so the style matches. `down()` drops the six tables in reverse order.

`display_mode`, `viewer_kind`, and `target_type` are retained deliberately although nothing reads them yet — Project B consumes the first two.

- [ ] **Step 4: Replace the migration registry**

`src/core/migrations/mod.rs` — replace the whole `migrations()` vec with the single baseline. The old vec order is *not* filename order, so replace it wholesale rather than editing entries:

```rust
fn migrations() -> Vec<Box<dyn MigrationTrait>> {
    vec![Box::new(m20260723_000001_create_schema::Migration)]
}
```

Update the `mod` declarations to match, then:

```bash
git rm src/core/migrations/m20260707_000001_create_model_backends_table.rs \
       src/core/migrations/m20260707_000002_create_runs_table.rs \
       src/core/migrations/m20260707_000003_create_artifacts_table.rs \
       src/core/migrations/m20260710_000002_create_execution_targets_table.rs \
       src/core/migrations/m20260710_000003_create_model_invocation_profiles_table.rs \
       src/core/migrations/m20260716_000004_add_input_id_to_runs_table.rs \
       src/core/migrations/m20260717_000005_add_artifact_types.rs
```

- [ ] **Step 5: Add the entity field**

In `src/core/entities/runs.rs`, add to `Model`:

```rust
    pub provenance_json: Option<String>,
```

- [ ] **Step 6: Verify against a real database**

Run: `cargo test`
Expected: PASS, 0 failed.

Run:
```bash
D=$CLAUDE_JOB_DIR/tmp/baseline.db && rm -f "$D" && \
  DATABASE_URL="sqlite://$D?mode=rwc" cargo run -q --bin vizfold -- seed && \
  sqlite3 "$D" ".schema runs" | grep -c provenance_json
```
Expected: `Seeded default executor records.` then `1`.

- [ ] **Step 7: Document the reset**

In `science-gateway/README.md`, in the "Resetting an Existing Development Database" section, add:

```markdown
The migration history was collapsed into a single baseline on 2026-07-23. An executor database
created before that will not match and is not migrated forward — delete the SQLite file and let
the executor recreate it. Seeding is existence-guarded, so the default records repopulate.
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "executor: collapse seven migrations into one baseline schema

The project is pre-production, so there is no migration history worth
preserving: the last migration already dropped and rebuilt the artifacts table
an earlier one created, and every down() was dead because rolling back a local
SQLite file means deleting it.

The baseline adds runs.provenance_json. Databases created before this are not
migrated forward; delete the file and let it recreate."
```

---

## Task 7: Provenance snapshot

**Spec:** PR5. Depends on Task 6.

`update_config` mutates the very profile row completed runs point at, and `resolve_output_location` reads `output_location` out of that row at artifact-registration time. So config drift between a run finishing and its artifacts being registered silently relocates where the executor looks for that run's outputs. A snapshot on the run fixes this.

**Files:**
- Modify: `src/adapters/cli.rs` (`queue_openfold_run`, ~line 588-610)
- Modify: `src/core/services/runs.rs` (`SubmitRunInput`)
- Modify: `src/core/output_locations.rs`

**Interfaces:**
- Produces: `runs::provenance_snapshot(backend, target, profile) -> String`, `output_locations::resolve_output_location_with_provenance(run, profile) -> Result<PathBuf, DbErr>`

- [ ] **Step 1: Write the failing tests**

Add to `src/core/services/runs.rs`'s test module:

```rust
#[test]
fn snapshot_records_every_catalog_payload_and_the_resolved_paths() {
    let snapshot = super::provenance_snapshot(
        "openfold", Some("v2.1"), "local-openfold", "local_subprocess",
        r#"{"output_location":"/work/runs"}"#,
    );
    let value: serde_json::Value = serde_json::from_str(&snapshot).expect("valid json");

    assert_eq!(value["backend"]["slug"], "openfold");
    assert_eq!(value["backend"]["version"], "v2.1");
    assert_eq!(value["target"]["slug"], "local-openfold");
    assert_eq!(value["profile"]["invocation_kind"], "local_subprocess");
    assert_eq!(value["profile"]["config"]["output_location"], "/work/runs");
}
```

Add to `src/core/output_locations.rs`'s test module:

```rust
#[test]
fn provenance_snapshot_wins_over_a_mutated_profile() {
    let profile_config = r#"{"output_location":"/work/relocated"}"#;
    let snapshot = Some(r#"{"profile":{"config":{"output_location":"/work/original"}}}"#.to_owned());

    let resolved = super::output_location_from(snapshot.as_deref(), profile_config).expect("resolved");

    assert_eq!(resolved, "/work/original");
}

#[test]
fn falls_back_to_the_profile_when_a_run_has_no_snapshot() {
    let resolved = super::output_location_from(None, r#"{"output_location":"/work/live"}"#).expect("resolved");
    assert_eq!(resolved, "/work/live");
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib provenance`
Expected: FAIL — `cannot find function provenance_snapshot` / `output_location_from`.

- [ ] **Step 3: Implement the snapshot**

In `src/core/services/runs.rs`:

```rust
/// Immutable record of what produced a run. Catalog rows can be edited later; this cannot.
pub fn provenance_snapshot(
    backend_slug: &str,
    backend_version: Option<&str>,
    target_slug: &str,
    invocation_kind: &str,
    profile_config_json: &str,
) -> String {
    let config: serde_json::Value =
        serde_json::from_str(profile_config_json).unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "backend": { "slug": backend_slug, "version": backend_version },
        "target": { "slug": target_slug },
        "profile": { "invocation_kind": invocation_kind, "config": config },
        "resolved": {
            "openfold_home": config::openfold_home().display().to_string(),
            "prefix": config::prefix().display().to_string(),
            "env_prefix": config::openfold_env_prefix().display().to_string(),
        },
    })
    .to_string()
}
```

Add `provenance_json: Option<String>` to `SubmitRunInput` and set it on the `ActiveModel` in the create path.

- [ ] **Step 4: Implement the resolution preference**

In `src/core/output_locations.rs`, extract the existing parse into a pure helper and prefer the snapshot:

```rust
/// Prefer the run's immutable snapshot; fall back to the live profile for runs queued before
/// snapshots existed.
pub fn output_location_from(
    provenance_json: Option<&str>,
    profile_config_json: &str,
) -> Result<String, DbErr> {
    if let Some(raw) = provenance_json
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(raw)
        && let Some(location) = value["profile"]["config"]["output_location"].as_str()
    {
        return Ok(location.to_owned());
    }
    let config: serde_json::Value = serde_json::from_str(profile_config_json)
        .map_err(|error| DbErr::Custom(format!("invalid profile config_json: {error}")))?;
    config["output_location"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| DbErr::Custom("profile config_json has no output_location".into()))
}
```

Route `resolve_output_location` through it, passing `run.provenance_json.as_deref()`.

- [ ] **Step 5: Capture the snapshot at queue time**

In `src/adapters/cli.rs`'s `queue_openfold_run`, after the backend/target/profile are resolved (~line 594), build the snapshot and pass it into `SubmitRunInput`.

- [ ] **Step 6: Verify**

Run: `cargo test`
Expected: PASS, 0 failed.

Run:
```bash
D=$CLAUDE_JOB_DIR/tmp/prov.db && rm -f "$D" && export DATABASE_URL="sqlite://$D?mode=rwc" && \
  cargo run -q --bin vizfold -- seed >/dev/null && \
  cargo run -q --bin vizfold -- queue-run openfold --input-id 6KWC_1 --input-sequence AAA >/dev/null && \
  sqlite3 "$D" "select json_extract(provenance_json,'\$.backend.slug') from runs limit 1"
```
Expected: `openfold`

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "executor: snapshot run provenance instead of trusting mutable catalog rows

update_config mutates the profile row completed runs point at, and
resolve_output_location reads output_location from that row when artifacts are
registered. Config drift between a run finishing and its artifacts being
recorded therefore relocated where the executor looked for its outputs.

Runs now carry an immutable snapshot of the backend, target, profile, and
resolved paths that produced them, and prefer it when resolving output
locations. Runs queued before this fall back to the live profile."
```

---

## Task 8: Delete `repositories/`

**Spec:** PR4. Depends on Task 5.

269 LOC across 8 files, almost all one-line forwards to SeaORM's active-record API — which already *is* the repository. There are **43 call sites**, not the ~15 the original audit estimated.

**Files:**
- Delete: `src/core/repositories/` (all 8 files)
- Modify: 9 files containing call sites, plus `src/core/mod.rs`

**Mechanical mapping** — apply to all trivial forwards:

| Repository fn | Inline replacement |
|---|---|
| `list(db)` | `Entity::find().all(db).await` |
| `find_by_id(db, id)` | `Entity::find_by_id(id).one(db).await` |
| `find_by_slug(db, slug)` | `Entity::find().filter(Column::Slug.eq(slug)).one(db).await` |
| `artifacts::list_by_run(db, id)` | `Entity::find().filter(Column::RunId.eq(id)).all(db).await` |
| `create(db, input)` | `ActiveModel { …fields…, ..Default::default() }.insert(db).await` |

**Two that are NOT trivial — inlining these naively corrupts data:**

1. `runs::update_status` (`repositories/runs.rs:32-58`) is a read-modify-write with **double-Option** semantics. `UpdateRunStatusInput.{started_at, completed_at, error_message}` are `Option<Option<T>>` where outer `None` means *leave the column untouched* and `Some(None)` means *write NULL*. A naive `Set(update.started_at)` collapses them and silently NULLs columns on partial updates. Move it verbatim into `services/runs.rs` as a free function. It must **not** stamp `updated_at` — `runs` has no such column.
2. `model_invocation_profiles::update_config` (`repositories/model_invocation_profiles.rs:37-49`) is a read-modify-write that **does** stamp `updated_at = Set(Utc::now())`. Move it verbatim into `services/model_invocation_profiles.rs`, replacing its internal `find_by_id` call with inline SeaORM. Preserve the asymmetry with the above exactly.

**Three traps:**
- **Grep trap.** All 9 `cli.rs` sites are *unprefixed* (`model_backends::find_by_id(…)`) because of the import at `cli.rs:11`, so `grep -rn 'repositories::'` finds none of them. The correct sites are `cli.rs:467, :477, :517, :588, :591, :594, :739, :758, :777`. The same `use` block imports `services::{…}`, so `artifacts::list_artifacts_for_run` (`:485`) and every `runs::*` in `cli.rs` are **service** calls — do not rewrite those.
- **No service equivalent.** Only 3 of the 9 `cli.rs` calls have one (the three `list`s). The other 6 must become inline SeaORM. `services::runs` has no `find_by_id`; the nearest is `get_run_with_artifacts`, which also loads artifacts.
- **Name collision.** Importing `entities::{execution_targets, model_backends, model_invocation_profiles}` clashes with the fully-qualified `crate::core::entities::model_invocation_profiles::Model` at `cli.rs:692` and with the identically-named service modules. Choose aliases deliberately.

- [ ] **Step 1: Move the two non-trivial functions**

Move `update_status` into `services/runs.rs` and `update_config` into `services/model_invocation_profiles.rs`, verbatim except for the internal `find_by_id` call becoming inline SeaORM.

- [ ] **Step 2: Verify the move in isolation**

Run: `cargo test --lib runs`
Expected: PASS, 0 failed — behaviour is unchanged at this point.

- [ ] **Step 3: Inline the 30 `services/` call sites**

Work file by file: `services/runs.rs` (8 sites), `services/artifacts.rs` (3), `services/artifact_types.rs` (3), `services/execution_targets.rs` (2), `services/model_backends.rs` (2), `services/model_invocation_profiles.rs` (4), `services/openfold_artifacts.rs` (3), `services/openfold_execution.rs` (4 + 3 in tests). Each `create` must copy its field list verbatim from the corresponding `Register*Input`.

- [ ] **Step 4: Inline the 9 `cli.rs` sites and the seed site**

Replace the `repositories::{…}` import at `cli.rs:11` with aliased entity imports. Then `seed.rs:251`'s `update_config` call points at the relocated service function.

- [ ] **Step 5: Delete the module**

```bash
git rm -r src/core/repositories/
```

Remove `pub mod repositories;` from `src/core/mod.rs:10`, and drop `#![allow(dead_code)]` from `src/core/services/mod.rs:1` so orphaned service functions become visible.

- [ ] **Step 6: Verify**

Run: `cargo build 2>&1 | grep -c '^error'`
Expected: `0`

Run: `cargo test`
Expected: PASS, 0 failed.

Run: `cargo clippy --all-targets 2>&1 | grep -E '^warning: unused|never used' | head`
Expected: review each — these are the service functions the removed `allow(dead_code)` now reveals. Delete any with no remaining callers.

- [ ] **Step 7: Remove the redundant round-trip tests**

`src/core/tests.rs` has no `repositories::` references, so nothing there breaks — but ~9 of its 15 tests are now plain SeaORM insert/select round-trips that test the framework rather than our behaviour. Delete those; **keep the 4 `rejects_*` validation tests**.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "executor: delete the repositories layer in favour of SeaORM directly

Every function was a one-line forward to the active-record API, which already
is the repository, and the boundary was fictional: the CLI imported
repositories directly and skipped services for nine of its calls.

update_status and update_config moved rather than inlined. update_status
carries double-Option semantics where an outer None means leave the column
alone, and update_config stamps updated_at while update_status must not,
because runs has no such column."
```

---

## Task 9: Delete the execution ceremony

**Spec:** PR6.

**Files:**
- Delete: `src/core/execution.rs`
- Modify: `src/core/preflight.rs`, `src/core/model_runners/openfold.rs`, `src/core/services/openfold_execution.rs`, `src/adapters/cli.rs`, `src/core/mod.rs`, `src/adapters/rest.rs`

**Keep `ExecutionCore`.** `rest.rs::serve` is its only caller, and Project B needs it. Move it into `src/core/db.rs` when `execution.rs` is deleted.

- [ ] **Step 1: Delete the single-implementation trait**

Delete the `PreflightRunner` trait (`preflight.rs:48-50`), the `OpenFoldPreflightRunner` struct and its impl (`openfold.rs:177-187`) — its body is one call to the free function `preflight_openfold` — and the 3 delegation tests at `openfold.rs:1868-1925`.

- [ ] **Step 2: Inline the workflow**

In `openfold_execution.rs`, replace the `execute_command_workflow(…)` call with the ~8 lines the caller actually wants:

```rust
        let report = preflight_openfold(&command, &invocation_profile, &run)?;
        if report.has_failures() {
            return Ok(ExecutionOutcome { report, output: None });
        }
        let output = runner.run(exec_command).await?;
        Ok(ExecutionOutcome { report, output: Some(output) })
```

Delete `preflight_failure_message` (`openfold_execution.rs:132-153`), a 22-line helper that reconstructs a reason the caller already has. Replace `ExecutionWorkflowResult`'s three-Option shape with a two-field `ExecutionOutcome`, and update the CLI consumer at `cli.rs:531-573`.

- [ ] **Step 3: Trim the unfalsifiable checks**

In `validate_entity_consistency` (`openfold.rs:428-469`), delete the 4 id-equality assertions. The sole caller fetches each entity *by the run's own FK*, so `run.model_backend_id == model_backend.id` cannot fail. Keep only the `invocation_kind == "local_subprocess"` guard.

- [ ] **Step 4: Merge the pass-through command**

Merge `execute_run` (`cli.rs:368-385`) into `execute_openfold` (`cli.rs:387-430`) — 18 LOC that resolve a run and check a backend slug the callee immediately re-resolves.

- [ ] **Step 5: Move `ExecutionCore` and delete the file**

Move `ExecutionCore` (`execution.rs:49-68`) into `src/core/db.rs`, update the import in `rest.rs:5`, then:

```bash
git rm src/core/execution.rs
```

Remove `pub mod execution;` from `src/core/mod.rs:5`.

- [ ] **Step 6: Consolidate the tests**

Trim `commands.rs`'s 187 test LOC to those testing our code rather than tokio — keep exit-code capture, the spawn-failure message, and the new streaming test; drop the env-passing and cwd-application cases. Lift the `TestLayout` helper duplicated between `openfold.rs` and `openfold_execution.rs:266-309` into one module. Table-drive the surviving `plan_openfold_command` flag cases into a single loop over `(params, expected_flag, expected_value)`.

- [ ] **Step 7: Verify**

Run: `cargo test`
Expected: PASS, 0 failed.

Run: `cargo clippy --all-targets 2>&1 | grep -c '^warning: unused'`
Expected: `0`

- [ ] **Step 8: Final verification against the spec's success criteria**

```bash
cargo build && cargo test && cargo clippy --all-targets
find src -name '*.rs' | xargs wc -l | tail -1
```
Expected: all green; total ~5,400 LOC (from 7,858).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "executor: inline the execution workflow and drop its ceremony

execute_command_workflow was generic over the runner plus an optional trait
object for a single caller, and returned three Options that forced consumers to
re-derive what had happened. The caller wants eight lines.

PreflightRunner had one real implementation, a struct whose body forwarded to
the free function. validate_entity_consistency's id-equality checks were
unfalsifiable, because the caller fetches each entity by the run's own key.

ExecutionCore moves to db.rs; the REST adapter still needs it."
```

---

## Self-Review

**Spec coverage:** PR1 → Tasks 3-4. PR2 → Tasks 1-2. PR3 → Task 5. PR4 → Task 8. PR5 → Task 7. PR6 → Task 9. PR7 → Task 6. All seven covered.

**Deliberately deferred, and why:** collapsing the 4 registry services (~139 LOC) is *not* a task here. The spec assigned them the provenance write path, which the snapshot approach removed, so they revert to being seed's helpers. Cutting them is optional cleanup and would expand this plan's scope; Task 8 Step 6 surfaces them via the removed `allow(dead_code)` if they are genuinely orphaned.

**Type consistency:** `CommandSpec.stream` (Task 4) is used in Tasks 4 and 9. `SlurmContext`/`gpu_launch`/`gpu_launch_args` (Task 2) are used only in Task 2. `srun_command` (Task 2) is referenced by Task 4's flag-setting step. `provenance_snapshot`/`output_location_from` (Task 7) depend on `runs.provenance_json` from Task 6. `ExecutionOutcome` (Task 9) replaces `ExecutionWorkflowResult` at both its consumers.

**Known estimate risk:** the LOC targets come from `wc -l` and are ±10%. The test-count expectations in Tasks 2-4 assume no other task has run in between; if tasks are reordered, verify against "0 failed" rather than the absolute count.
