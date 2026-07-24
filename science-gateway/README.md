# VizFold Gateway

Prototype workspace for the VizFold science gateway.

This subtree currently has two distinct tracks:

- `apps/executor`: the active Rust core for persistence and execution-domain work.
- `apps/workbench`: a disconnected Next.js frontend prototype that still runs on mock data.

The frontend is useful for UX iteration, but the Rust executor is the main implementation path for core data model, database, and execution workflow development.

## Repository Structure

- `apps/`: runnable gateway applications.
- `apps/executor/`: Rust service and core library. Contains SeaORM entities, migrations, services, seed setup, and the current Axum adapter.
- `apps/workbench/`: Next.js workbench prototype for browsing concepts and mock flows. Not wired to the Rust executor yet.
- `docs/`: gateway-specific notes, architecture sketches, future UX ideas, and backlog material.
- `docs/architecture.md`: high-level architecture notes for the gateway direction.
- `docs/future-ux.md`: product and interaction ideas that are not implemented yet.
- `docs/todo.md`: working backlog and rough implementation notes.
- `docs/img/`: diagrams and supporting images used by the docs.
- `CONTRIBUTING.md`: lightweight branching and contribution guidance for this fork.

## Current Development Status

- The Rust executor is the primary active implementation path.
- The workbench is still a UI prototype with static mock data.
- The workbench does not own persistence and is not connected to the executor yet.
- SeaORM migrations are part of the Rust executor startup flow.

## Local Development

### Prerequisites

- Node.js 20 LTS or later
- npm
- Git
- Rust toolchain with `cargo` and `rustc`

Recommended version checks:

```bash
node -v
npm -v
cargo -V
rustc -V
```

### Clone the repository

```bash
git clone <repo-url>
cd vizfold-foundation/science-gateway
```

## Workbench Development

Run the frontend prototype from `science-gateway/apps/workbench`:

```bash
cd apps/workbench
npm install
npm run dev
```

The workbench will be available at [http://localhost:3000](http://localhost:3000).

Notes:

- This app currently uses mock data only.
- No database setup is required for the current workbench prototype.
- The older generic Next.js README language about alternate package managers or Vercel deployment is not important for current gateway development.

## Executor Development

Run the Rust executor from `science-gateway/apps/executor`. `cargo run` alone now runs the `vizfold` CLI (see below), so name the REST binary explicitly:

```bash
cd apps/executor
cargo run --bin executor
```

The current HTTP health endpoint is available at [http://127.0.0.1:3001/health](http://127.0.0.1:3001/health).

### Executor CLI

The `vizfold` CLI provides a development workflow for inspecting and operating persisted OpenFold runs. Run it from `science-gateway/apps/executor`:

```bash
cargo run --bin vizfold -- seed
cargo run --bin vizfold -- list models
cargo run --bin vizfold -- list targets
cargo run --bin vizfold -- list profiles
cargo run --bin vizfold -- list runs
```

Queueing is model-specific because a run does not exist yet. Once queued, operations are run-centric:

```bash
cargo run --bin vizfold -- queue-run openfold ...
cargo run --bin vizfold -- execute-run <run-id>
cargo run --bin vizfold -- register-artifacts <run-id>
cargo run --bin vizfold -- show run <run-id>
```

`seed` is safe to repeat and ensures the local OpenFold backend, `local-openfold` target, and matching invocation profile are available. The CLI uses `DATABASE_URL` when set and otherwise uses the SQLite default described below. For the complete local OpenFold setup and an end-to-end CLI workflow, see [DEMO.md](DEMO.md).

### Installing the CLI

End users install the prebuilt release binary via the bootstrap in the repo-root
[README](../README.md#install) (`curl … install.sh | bash`). For development, build from source
in `science-gateway/apps/executor`:

```bash
cargo build --bin vizfold
./target/debug/vizfold --help
```

On PowerShell, run the built binary with:

```powershell
.\target\debug\vizfold.exe --help
```

To install only the CLI binary into Cargo's bin directory (typically `~/.cargo/bin`) so it can be invoked directly, use:

```bash
cargo install --path . --bin vizfold
vizfold --help
vizfold seed
```

Use `cargo install --path . --bin vizfold --force` to update an existing installation. This is currently a development/demo CLI: the seeded local OpenFold profile assumes the checked-out repository layout, so build and run it against that checkout rather than treating it as a general standalone installed application.

### Database and SeaORM Migrations

The executor uses SQLite and SeaORM migrations.

Current behavior:

- Database URL resolution order (`config::database_url()`): `DATABASE_URL` env var, then `VIZFOLD_DB` (env var or install config), then `<OPENFOLD_PREFIX>/vizfold.db`, then `$XDG_DATA_HOME/vizfold/vizfold.db` (`~/.local/share/vizfold/vizfold.db` by default).
- Parent directories are created automatically if they do not exist.
- SeaORM migrations run automatically whenever the executor or CLI connects to the database.
- Default seed records are inserted by `vizfold seed`.

Create the database and apply migrations:

```bash
cd apps/executor
cargo run --bin vizfold -- seed
```

That command will:

1. open or create the SQLite database file,
2. enable SQLite foreign keys,
3. run SeaORM migrations,
4. seed default model backend and execution target records.

To use a different SQLite file, set `DATABASE_URL` before running the CLI.

PowerShell:

```powershell
$env:DATABASE_URL = "sqlite://data/vizfold-dev.db?mode=rwc"
cargo run --bin vizfold -- seed
```

Bash:

```bash
export DATABASE_URL="sqlite://data/vizfold-dev.db?mode=rwc"
cargo run --bin vizfold -- seed
```

There is not currently a migrations-only CLI command; use either `cargo run --bin executor` (REST startup) or `vizfold seed` for local development setup.

### Resetting an Existing Development Database

If you already ran the executor against a database from before the migrations were collapsed into a single baseline schema, the CLI/executor will fail with an actionable error naming the exact file to delete, e.g.:

```text
this executor database predates the 2026-07-23 baseline schema; delete <path> and re-run
```

If you want to find the file yourself instead of waiting for that error, it is whatever `DATABASE_URL` → `VIZFOLD_DB` → `<OPENFOLD_PREFIX>/vizfold.db` → `$XDG_DATA_HOME/vizfold/vizfold.db` resolves to (see above) -- not a fixed `apps/executor/data/vizfold.db` path.

Current expectation:

- this reset guidance is appropriate for local development
- no production-safe migration path exists yet for carrying an older executor DB forward automatically

The migration history was collapsed into a single baseline on 2026-07-23. An executor database
created before that will not match and is not migrated forward — delete the SQLite file and let
the executor recreate it. Seeding is existence-guarded, so the default records repopulate.

### Tests

Run the Rust executor tests from `science-gateway/apps/executor`:

```bash
cargo test
```

These tests exercise the in-memory SQLite path, SeaORM migrations, and the core registration/run/artifact services.

## What May Be Obsolete

The previous gateway README referenced directories such as `packages/schemas`, `packages/adapters`, and `examples` as if they were part of `science-gateway`. Those entries do not exist in the current `science-gateway` subtree and should not be treated as active local structure here.

Likewise, the workbench should not be described as integrated with the backend yet. The current accurate state is:

- frontend prototype in `apps/workbench`
- Rust core and persistence work in `apps/executor`
- no real executor-to-workbench wiring yet
