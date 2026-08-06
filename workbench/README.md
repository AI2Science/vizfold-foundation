# VizFold Workbench

The dashboard for the vizfold executor: fold proteins, then read what the model computed — the
predicted structure, the attention behind it, and the activations the run stored.

It is a Bun program end to end. `Bun.serve` bundles the React client and answers the API in one
process; `bun:sqlite` reads the executor's database read-only; the CLI is shelled out to for
everything that writes (`list proteins`, `run <proteins…> --no-exec`, `run <id>`). The UI is built
from [Base UI](https://base-ui.com) primitives styled by `src/client/app.css` — one design system,
light and dark, no component-library theme underneath it.

## Running it

```bash
vizfold serve                     # every installed backend, http://localhost:3000
vizfold serve openfold esmfold    # only these two
vizfold serve --port 4000
```

`serve`:
- stages this directory to `<OPENFOLD_PREFIX>/workbench` (in place when the prefix is
  `OPENFOLD_HOME`); it never clones a missing checkout
- provisions Bun if the host has none (>= 1.2; the static musl build, so an old glibc is fine)
- runs `bun install` on first use, then `bun run start`
- exports `VIZFOLD_BIN`, `OPENFOLD_PREFIX`, `VIZFOLD_DB`, `VIZFOLD_BACKENDS` (the served slugs,
  comma-separated: the run list is filtered to them, and the fold form picks between them when there
  is more than one) and `PORT`. Unset — `bun run dev` by hand — filters nothing and lets the CLI
  choose.

`bun run dev` by hand needs that env passed in — the workbench reads `process.env` only, never the
vizfold config, so without it the run list is silently empty and folding fails:

```bash
# both values are in `vizfold status`, as OPENFOLD_PREFIX and database
OPENFOLD_PREFIX=<OPENFOLD_PREFIX> VIZFOLD_DB=<database> bun run dev
```

```bash
bun test          # attention parsing, run-file resolution
bun run typecheck
```

## What it shows

Only what a run actually produced. Every file a run writes is registered under a true kind —
protein structure, attention map, attention or activation tensor, trace summary, run metadata, the
alignments it searched for itself — classified from its path by the executor, and the dashboard
takes its viewer and display mode from that kind. A tab appears when the files behind it are on
disk, and not before: no placeholder structures, no sample attention, no demo run. A fold that wrote nothing says
so and names the directory it looked in.

- **Structure** — the relaxed prediction per target in a 3Dmol viewer (cartoon/stick/line/sphere,
  chain-spectrum/residue/chain/pLDDT colouring), with every other structure the target wrote linked
  beside it.
- **Attention** — arc diagrams drawn in the browser from the run's own attention dump. The server
  parses `msa_row_attn_layer<L>.txt` and `triangle_start_attn_layer<L>_residue_idx_<R>.txt` — the
  files `save_attention_topk` writes — into per-head edge lists, and the client draws them: forward
  pairs above the residue axis, backward below, arc height by separation, colour and width by
  weight. Target, attention type, layer, query residue, head and edges-per-head are all pickers, and
  the same data is one click away as a table. Nothing is pre-rendered; a `.npz` beside a dump is
  offered as a download.
- **Activations** — ESMFold's `trace/summary.json` as per-layer magnitude and attention-statistic
  bars, its `trace/index.json` as a tensor inventory (shape, dtype, size, download), its `meta.json`
  as the run's model/device/dtype header, plus every dense array on disk no index claims —
  OpenFold's `.npz` dumps and `_output_dict.pkl`.
- **Files** — everything under the run directory, filterable, each one downloadable, and named by
  the kind the executor classified it as rather than by its extension.

## Layout

```
src/
  server/     Bun.serve, bun:sqlite reads, CLI calls, attention parsing, trace reading
  client/     React 19 + Base UI, one page each for the dashboard, the run list and a run
  shared/     the types both sides speak
```

The server never trusts a path from the browser: file and attention requests resolve inside the
run's own directory (`resolveInside`), and a fold only ever hands the CLI ids the CLI itself listed.
