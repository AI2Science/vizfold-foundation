# Executor slimming + HPC/workstation execution

**Date:** 2026-07-23
**Status:** design approved, ready for implementation planning
**Scope:** Project A of two (see [Scope](#scope))

## Problem

The `executor` crate is 7,858 Rust LOC implementing a multi-model/multi-cluster gateway
catalog — 6 SQLite tables, an entity → repository → service layer cake, and a schema-interpreter
command planner — around what is at runtime exactly one command line: the one `run/fold.sh:52-70`
assembles in 19 lines of bash.

Two things are wrong, and they are not the two you would guess:

1. **The layering does not pay for itself.** `src/core/repositories/` (269 LOC, 7 files) is 100%
   pass-through to SeaORM's active-record API, and the boundary it claims to draw is fictional —
   `cli.rs:11` imports `repositories::` directly, skipping the service layer for 8 of its calls.
2. **The SLURM support does not exist.** `grep -riE 'slurm|sbatch|srun|squeue|scontrol'` over
   `src/` and `examples/` returns zero hits. `LocalCommandRunner` (`commands.rs:26-52`) spawning a
   local subprocess is the only implemented execution path. The scheduler lives entirely in
   `install/` bash.

The premise "make it run on a beefy workstation" therefore inverts: the workstation path is
~45 LOC of polish, and **the HPC path is the one that is missing**.

## Scope

**Project A (this spec)** — slim the executor, make both execution targets first-class.

**Project B (separate spec, after A)** — wire the Next.js workbench to real data via a read API.
Project A deliberately *preserves* `axum`, `rest.rs`, `ExecutionCore`, the `artifacts` table, and
all 13 `artifact_types` rows so B has something to consume. B is out of scope here because it
depends on A's schema having settled; bundling them would mean writing a frontend against a
moving target.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Keep** the 3-table catalog and the schema interpreter | Explicit call. The audit recommended cutting both (~1,385 LOC); preserved instead for decision 2. |
| 2 | The catalog's job is a **provenance log** | Rows are written by `vizfold install` from resolved config, **append-on-change, never mutated**. Each run FKs the rows current at queue time, so run #47 stays reproducible after a reinstall changes the config. This defeats the "it is only ever 1-2 rows" critique: provenance rows accumulate. |
| 3 | Workbench gets wired up (Project B) | Hence `axum` stays and becomes a real API rather than being deleted and re-added. |
| 4 | HPC execution is **blocking `srun`**, never detached `sbatch` | Fits `CommandRunner`'s fire-and-wait shape with no schema change. Detached batch would need `runs.job_id`, a `running` status, a poll command, and crash reconciliation. |
| 5 | Sequencing is **value-first** | Workstation + HPC land in PR1-2 (~70 LOC, provably independent of all layer surgery). Everything after is subtraction against a working system. |
| 6 | **`vizfold install` also uses blocking `srun` with streamed stdio** | Consistency: no stage of the system detaches. Covered in PR2. |

### Rejected

- **Cutting the catalog to a Rust `const`** (audit's recommendation, ~765 LOC). Rejected in favour
  of decision 2. Cost: roughly half the available reduction.
- **Replacing the schema interpreter with a literal arg builder** (~620 LOC). Rejected with the
  catalog it serves.
- **A `ModelRunner` trait with one implementation.** Deferred until model #2 actually lands, so it
  can be designed against two real implementations rather than one imagined one.
- **`sbatch` + poll.** See decision 4. Worth revisiting only when losing a terminal actually bites.

### Known consequence of decision 1

The catalog is currently **read-only in practice**: no `register`/`add` command exists in the CLI,
rows come only from `seed.rs`, `queue_openfold_run` hardcodes the slug lookups `"openfold"` and
`"local-openfold"` (`cli.rs:588-591`), and the planner rejects any profile whose
`invocation_kind != "local_subprocess"` (`openfold.rs:461`). Adding boltz would still mean editing
Rust. Decision 2 makes the tables earn their keep as provenance regardless; making models
genuinely data-addable is explicitly **not** in scope.

## Target architecture

One binary, one SQLite file, layered `core/` — one layer removed, one seam added.

```
adapters/cli.rs      product surface (install, queue-run, execute-run, list, show, ...)
adapters/rest.rs     /health today; Project B grows it into the read API   [kept, untouched]
core/entities/       6 tables, unchanged
core/repositories/   -- DELETED (SeaORM active-record is the repository)
core/services/       runs, artifacts, openfold_*, validation,
                     + the 4 registry services, repurposed as the provenance write path
core/model_runners/  openfold.rs -- schema interpreter KEPT (decision 1)
core/commands.rs     + streaming mode
core/preflight.rs    + GPU check
core/config.rs       + srun-prefix composition
```

Explicitly untouched: the schema interpreter, the 3 catalog tables, `artifacts`/`artifact_types`,
`axum`, and `install/`'s 683 LOC of bash (whose comments encode real cluster failures — NVRTC/PTX
pinning, `CONDA_OVERRIDE_CUDA` on driverless build nodes, `env -i` against leaked site toolchains).

## Implementation

Seven PRs. Each is independently shippable and revertible.

### PR1 — Workstation first-class (+45 LOC)

- **GPU preflight.** Add a `PreflightCheck` running `nvidia-smi --query-gpu=name --format=csv,noheader`,
  mirroring `run/fold.sh:44-46`. **Warns** rather than fails when no GPU is visible, because
  `model_device` now degrades to `cpu` (below). Still fails loudly on a cluster outside an
  allocation, matching `fold.sh`.
- **`model_device` default.** `cli.rs:123-124` is `#[arg(long, default_value = "cuda:0")]`. Derive
  the default from GPU detection instead: `cuda:0` when a GPU is visible, `cpu` otherwise.
- **`cpus` default.** `cli.rs:125-126` is `default_value_t = 1`, so a beefy workstation
  single-threads by default. Use `std::thread::available_parallelism()`.
- **Streamed output.** `commands.rs:39` uses `command.output().await`, buffering everything;
  `cli.rs:560-565` prints it only after exit. A multi-hour fold shows a blank terminal, then a
  dump. Add `stream: bool` to `CommandSpec`; when set, `LocalCommandRunner` uses `.status()` with
  inherited stdio — the pattern `run_npm` and `run_install` (`cli.rs:204`) already use. Set it for
  `execute-run`.

**Accepted trade-off:** streaming means stderr is no longer captured, so the failure message at
`openfold_execution.rs:95-99` degrades to its existing fallback, `"OpenFold command exited with
code {n}"` (`openfold_execution.rs:96`). This is a net win — the user watched the error scroll past
live instead of staring at nothing for hours. Exit code is all `openfold_execution.rs:82` branches on.

### PR2 — Cluster execution is blocking `srun` with streamed stdio (+25 LOC, bash + Rust)

**Install path (bash).** The chain already streams end to end except one branch:

```
vizfold install  -> .status() inherits stdio          streams   (cli.rs:204)
  -> init.sh -> slurm::run                                      (slurm.sh:51-84)
       SLURM_STEP_ID set -> bash                      streams   (already on node)
       SLURM_JOB_ID set  -> srun --ntasks=1           streams   (salloc)
       neither           -> sbatch                    DETACHES  (slurm.sh:73-80)
```

Replace the `sbatch` branch at `slurm.sh:73-80` with `srun`:

```bash
LAUNCH=(
    srun --job-name=vizfold-install
    --account="$ACCOUNT" --partition="$PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="${OPENFOLD_BUILD_CPUS:-8}"
    --mem="${OPENFOLD_BUILD_MEM:-24G}" --time="${OPENFOLD_BUILD_TIME:-02:00:00}"
    ${OPENFOLD_BUILD_GRES:+--gres="$OPENFOLD_BUILD_GRES"}
)
"${LAUNCH[@]}" "$SETUP" 2>&1 | tee "$PREFIX/install.log"
exit "${PIPESTATUS[0]}"
```

`--output=` and `--export=ALL` drop — both are `sbatch` concepts; `srun` streams to the terminal
and propagates the environment by default. `exec` is replaced by the tee pipeline, with
`PIPESTATUS[0]` preserving the real exit code. This collapses branches 2 and 3 into one tool.

Losing the terminal is survivable: `install/setup.sh:14` documents that *"a step seals only after
finishing, so an interrupted create/clone/download is redone next run, not skipped"* — re-running
`vizfold install` continues. README gains a `tmux` recommendation for long installs. Nothing reads
the dropped `install-%j.log`; the path appears exactly once in the repo, at `slurm.sh:79`.

**Fold path (Rust).** Compose the `srun` prefix in `config.rs` from the four persisted keys
`OPENFOLD_GPU_{ACCOUNT,PARTITION,RESOURCES,GRES}`, mirroring `setup.sh:212`. Note the composed
`LAUNCH` string is a *local* in `setup.sh:212` and is **not** persisted — only its resolved
components are (`setup.sh:220-230`). Walltime is likewise not persisted: `setup.sh:212` hardcodes
`-t 00:30:00`, correct for the install's example fold and far too short for a real one. Add
`OPENFOLD_GPU_TIME`, defaulting to `02:00:00`.

Apply the prefix in `openfold_execution.rs` beside `activate_env_command` (`:159-183`), at the same
gate as the micromamba wrapper (`:63`, `prefix.join("bin/micromamba").is_file()`). Mirror the
bash's context detection exactly — a 4-way, not a 2-way:

| Context | Fold command |
|---|---|
| `SLURM_STEP_ID` set | bare — already in a step; nesting `srun` would fail |
| `SLURM_JOB_ID` set | `srun --ntasks=1` — `salloc` leaves you off the node |
| `OPENFOLD_GPU_PARTITION` set | full `srun -A … -p … --gres=… -t $OPENFOLD_GPU_TIME` |
| neither | bare — beefy workstation |

**After PR2 the headline claim is true: one executor, HPC or workstation.** Everything below is
structural cleanup.

### PR3 — Dead surface (−800 LOC)

- `examples/` — all 4 files, 692 LOC. Not build-load-bearing (nothing in `Cargo.toml`, CI, or
  `install/` references them). Cited in 3 docs, which update in the same PR: `DEMO.md:90`,
  `openfold-demo/ENVIRONMENT.md:77`, `openfold-demo/ENVIRONMENT.md:117`.
- `PreflightReport::passed()` (`preflight.rs:63`) and `PreflightReport::warnings()`
  (`preflight.rs:70`) — zero non-test callers. **Trap:** `PreflightCheck::passed(name, message)` at
  `preflight.rs:18` is a *different, live* constructor. Do not delete it. `failures()` (`:77`) and
  `has_failures()` (`:57`) are also live.
- `pub const DEFAULT_DATABASE_URL` (`config.rs:5`) — zero callers.
- `repository_root()` (`config.rs:119`) — `pub` but called only within `config.rs`; make private.
- The `local-mock` execution target and `mock` invocation profile seed blocks. Dead twice over:
  nothing names `local-mock` in production, and `openfold.rs:461` rejects its `invocation_kind`.
  The 3 tests referencing it register their own fixture inline, as `openfold_execution.rs:339`
  already does. **This removes two seeded rows, not the tables** — decision 1 keeps every table;
  decision 2 gives them a job. Unreachable rows are not part of that job.
- `default-run = "vizfold"` in `Cargo.toml`. Because the package is named `executor` and `main.rs`
  exists, `cargo run` currently starts the `/health` stub instead of the CLI.

### PR4 — Delete `repositories/` (−620 LOC: 269 impl + ~350 tests that go with it)

Every function is a one-line forward: `list` = `Entity::find().all(db)`, `find_by_id` =
`Entity::find_by_id(id).one(db)`. `mod.rs` already carries `#![allow(dead_code)]`.

Inline SeaORM at the ~15 call sites. The two functions with real bodies — `runs::update_status`
(field-merge) and `model_invocation_profiles::update_config` — become free functions in
`services/runs.rs`. Point `cli.rs` at services and entities, eliminating the two-parallel-access-paths
problem (`cli.rs:11`, and calls at `:323, :333, :444, :447, :450, :595, :614, :633`).

Roughly 9 of `tests.rs`'s 15 tests delete with this — they are SeaORM insert/select round-trips
testing the framework. Keep the 4 `rejects_*` validation tests.

### PR5 — Provenance (−20 LOC net)

`vizfold install` writes the catalog rows from resolved config, replacing the hardcoded
`seed.rs` blocks. Semantics: **compare the resolved config against the newest existing row of that
kind; if any compared field differs, insert a new row; never `UPDATE`.** Runs FK the rows current
at queue time (they already do), so historical runs keep pointing at what actually produced them.

The compared fields, stated explicitly so "changed" is not a judgement call:

| Row | Compared on |
|---|---|
| `model_backends` | `slug`, `version` |
| `execution_targets` | `slug`, `target_type`, `available_resources_json` |
| `model_invocation_profiles` | `slug`, `invocation_kind`, `config_json` |

"Newest" means highest `id` among rows sharing that `slug`. Lookups that currently resolve a slug
to a single row (`cli.rs:588-591`) must become "newest row for this slug" — otherwise a second
provenance row makes the query ambiguous. This is the one place PR5 changes read behaviour, and it
is the step most likely to break `queue-run` if missed.

The 4 registry services become the write path and are thereby justified — they are the only place
`require_json_object` validation runs.

### PR6 — Execution ceremony + test cleanup (−585 LOC: ~385 ceremony + ~200 test consolidation)

- Delete the `PreflightRunner` trait (`preflight.rs:48-50`) and its single forwarding
  implementation `OpenFoldPreflightRunner` (`openfold.rs:177-187`), whose body is one call to the
  free function `preflight_openfold`. Every other implementation is a test double. Delete the 3
  delegation tests at `openfold.rs:1868-1925`.
- Inline `execute_command_workflow` (`execution.rs:17-47`) into `execute_openfold_run` as ~8 lines:
  `if report.has_failures() { mark_failed(..); return }`. Delete `ExecutionWorkflowResult`'s
  3-Option soup and `preflight_failure_message` (`openfold_execution.rs:132-153`), a 22-line helper
  that exists only to reconstruct a reason the caller already knew. Its 187 tests cover 6
  permutations of one branch; keep one.
- **Keep `ExecutionCore`** (`execution.rs:49-68`) — `rest.rs` bootstraps from it and Project B
  needs it. Only the workflow function and its result type go.
- Trim `validate_entity_consistency` (`openfold.rs:428-469`) to its one live check. The 4 id-equality
  assertions are structurally unfalsifiable: the sole caller (`openfold_execution.rs:29-39`) fetches
  each entity *by the run's own FK*, so `run.model_backend_id == model_backend.id` cannot fail.
  Keep the `invocation_kind == "local_subprocess"` guard.
- Merge `execute_run` (`cli.rs:368-385`) into `execute_openfold` (`cli.rs:387-430`) — 18 LOC of
  pass-through that re-fetches what the callee immediately re-fetches.
- Test consolidation (~200 LOC): trim `commands.rs`'s 187 test LOC to the two that test our code
  rather than tokio (exit-code capture, spawn-failure message), lift the `TestLayout` helper
  duplicated across `openfold.rs` and `openfold_execution.rs:266-309` into one module, and
  table-drive the surviving `plan_openfold_command` flag cases into a single loop over
  `(params, expected_flag, expected_value)`.

### PR7 — Squash migrations (−560 LOC)

7 migrations (710 LOC) into 1 create-schema migration (~150). `m20260717_000005` drops and rebuilds
the artifacts table that `m20260707_000003` created two weeks earlier, and carries a
`create_legacy_artifacts_table()` helper. Every `down()` is dead: rolling back a single-user local
SQLite file means deleting it, which is exactly what `science-gateway/README.md:179-224` prescribes
while stating no production-safe migration path exists.

**Do this last**, once the entity set has stopped moving, so the squash happens once.

### Reconciliation

| PR | Δ LOC |
|---|---|
| PR1 workstation | +45 |
| PR2 `srun` + streaming | +25 |
| PR3 dead surface | −800 |
| PR4 `repositories/` | −620 |
| PR5 provenance | −20 |
| PR6 ceremony + tests | −585 |
| PR7 migration squash | −560 |
| **Net** | **−2,515** |

7,858 − 2,515 = **5,343** (−32%). PR4 and PR6 figures include the tests that delete alongside
their implementation, which is where most of the test reduction comes from. Treat all counts as
±10%: they are `wc -l` on the current tree, spot-verified, not a compiler's opinion.

## Data flow

```
vizfold install ──> vizfold.json (resolved config)
                └─> catalog rows (backend/target/profile)   append-on-change, never mutate
                          │
queue-run ──> runs row, FK'd to the rows current at queue time
                          │
execute-run ──> planner reads profile schema ──> CommandSpec
                          └─> [srun prefix, per 4-way context] ──> [micromamba activate]
                                    └──> streamed exec ──> status + artifacts
```

## Error handling

Preflight keeps its pass/warn/fail model and still blocks execution on failure. Changes:

- **No GPU visible** → warn, not fail, since `model_device` degrades to `cpu`. A CPU-only
  workstation runs (slowly) rather than erroring.
- **On a cluster outside an allocation** → still fails, matching `fold.sh:44-46`.
- **Streamed run fails** → message is `"OpenFold command exited with code {n}"`; the error itself
  already scrolled past live.
- **Install interrupted** → re-run `vizfold install`; `setup.sh` re-does the unsealed step.
- `DbErr::Custom` remains the CLI's general-purpose error type. It is ugly — npm failures surface
  as database errors — but the user-visible strings are already correct and changing ~16 sites
  plus every helper signature buys nothing behavioural. Recorded so it is not rediscovered.

## Testing

One runnable check per non-trivial change.

**Deleted for free** (side effects of the cuts, not deliberate test deletion): ~9 SeaORM
round-trips in `tests.rs` with PR4; 3 preflight-delegation tests and 5 of 6 workflow permutations
with PR6; the `local-mock` fixtures with PR3.

Note `assert_eq!(artifact_types.len(), 13)` in `tests.rs` **stays valid** — decision 3 keeps all 13
rows. Only the execution-target and invocation-profile count assertions move, because PR3 removes
the `local-mock`/`mock` pair.

**New:**

| Test | Asserts |
|---|---|
| GPU preflight, absent | check warns; `model_device` resolves to `cpu` |
| GPU preflight, present | check passes; `model_device` resolves to `cuda:0` |
| `srun` composition | composed string matches `setup.sh:212`'s shape for the same inputs |
| 4-way context detection | each of `SLURM_STEP_ID` / `SLURM_JOB_ID` / partition / neither yields the right prefix |
| Streaming | exit code propagates when stdio is inherited |
| Provenance append | changed config inserts a row; unchanged config inserts none |
| Provenance immutability | an existing row is never updated |

Test LOC: 2,977 → **~1,900**. Less aggressive than the audit's ~700 because the schema
interpreter's permutation tests stay with the interpreter (decision 1).

## Success criteria

1. `vizfold install` on a login node runs via blocking `srun`, streams live, and writes
   `$PREFIX/install.log`; the real exit code propagates.
2. `vizfold execute-run` folds on a beefy workstation with no SLURM present, defaulting to all
   cores and streaming progress.
3. The same command on a cluster prefixes `srun` correctly in all four contexts.
4. Re-running `vizfold install` after a config change appends a catalog row; an unchanged config
   appends none; no existing row is ever mutated.
5. `cargo build`, `cargo test`, `cargo clippy --all-targets` clean.
6. Rust LOC 7,858 → ~5,300 (−32%); 6 tables intact.

## Risks

| Risk | Mitigation |
|---|---|
| `srun` holds the terminal through a queue wait | `srun` prints `job N queued and waiting for resources`; README recommends `tmux`. Install is resumable (`setup.sh:14`). |
| Streaming loses captured stderr | Accepted; the fallback message already exists and live output is strictly more useful. |
| PR5 changes bootstrap behaviour | Seeds are existence-guarded today; append-on-change is additive. Verify on a fresh DB and an existing one. |
| PR7 squash on an existing dev DB | Already the documented workflow: delete the SQLite file and re-create (`science-gateway/README.md:179-224`). |
| Estimates assume the audit's LOC counts | Counts were taken from `wc -l` on the current tree and spot-verified; treat as ±10%. |

## Follow-ups (not this project)

- **Project B** — read API + workbench off mock data.
- Detached `sbatch` + `vizfold status`, if losing a terminal on a multi-hour fold becomes a real
  complaint (needs `runs.job_id`, a `running` status, crash reconciliation).
- A `ModelRunner` trait, when model #2 lands.
- Making the catalog genuinely writable (`vizfold register …`), if models should be addable without
  a recompile.
