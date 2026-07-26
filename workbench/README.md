# VizFold Workbench

Next.js dashboard for the vizfold executor. Reads the executor's SQLite read-only via `node:sqlite`
and shells out to the `vizfold` binary (`list examples`, `queue openfold`, `run <id>`).

## Running it

```bash
vizfold serve                # http://localhost:3000
vizfold serve --port 4000
```

`serve`:
- stages this directory to `<OPENFOLD_PREFIX>/workbench` (in place when the prefix is
  `OPENFOLD_HOME`); it never clones a missing checkout
- provisions Node if the host has none (>= 22.13, for `node:sqlite`)
- runs `npm install` on first use, then `npm run dev`
- exports `VIZFOLD_BIN`, `OPENFOLD_PREFIX` and `VIZFOLD_DB`
- symlinks `<OPENFOLD_PREFIX>/runs` to `public/runs`, so the 3D viewer and attention images resolve

`npm run dev` by hand needs that env passed in — the workbench reads `process.env` only, never the
vizfold config, so without it the run list is silently empty and folding 500s:

```bash
# both values are in `vizfold status`, as OPENFOLD_PREFIX and database
OPENFOLD_PREFIX=<OPENFOLD_PREFIX> VIZFOLD_DB=<database> npm run dev
```
