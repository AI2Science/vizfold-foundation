# VizFold Gateway

Prototype for the VizFold Science Gateway.

## Structure

- `apps/workbench`: browser-based VizFold interface, runnable locally and deployable later as a science gateway frontend.
- `apps/executor`: Rust backend executor service currently exposing an Axum HTTP adapter.
- `packages/schemas`: shared data contracts.
- `packages/adapters`: model adapter interfaces and implementations.
- `examples`: sample inputs and outputs.
- `docs`: architecture and implementation notes.

## Local Development

### Prerequisites

- Node.js 20 LTS or later
- npm
- Git
- Rust toolchain (`cargo`, `rustc`)

> **Recommended:** Use `nvm` (Linux/macOS/WSL) or `nvm-windows` (Windows) to manage Node.js versions.

Verify your installation:

```bash
node -v
npm -v
cargo -V
```

---

### Clone the repository

```bash
git clone ...
```

---

### Install dependencies

```bash
cd science-gateway/apps/workbench
npm install
```

---

### Start the workbench prototype

```bash
npm run dev
```

The workbench prototype will be available at:

```
http://localhost:3000
```

The workbench is currently work-in-progress and uses static mock data only. It does not own persistence and is not wired to the executor yet.

## Executor Development

The Rust executor is the primary active implementation path.

Run the Axum HTTP adapter locally from the gateway root:

```bash
cd apps/executor
cargo run
```

The health check will be available at:

```
http://127.0.0.1:3001/health
```

By default, the executor uses SQLite at `apps/executor/data/vizfold.db` and applies SeaORM migrations automatically on startup. To override the location, set `DATABASE_URL`, for example:

```powershell
$env:DATABASE_URL = "sqlite://data/vizfold.db?mode=rwc"
cargo run
```
