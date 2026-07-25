# VizFold Workbench

This is the Next.js workbench app for the VizFold science gateway prototype.

It reads the executor's SQLite directly (read-only, via `node:sqlite`) and shells out to the
`vizfold` binary to list examples and queue/execute runs.

## Running it

```bash
vizfold serve            # http://localhost:3000
```

`serve` stages the app, provisions Node if the host has none, exports `VIZFOLD_BIN`, `VIZFOLD_DB`
and `OPENFOLD_PREFIX`, and symlinks `<OPENFOLD_PREFIX>/runs` to `public/runs` so the 3D viewer and
attention images resolve. `npm run dev` here works too, but falls back to
`<OPENFOLD_PREFIX>/vizfold.db` and has no `public/runs` link. Needs Node >= 22.13 for `node:sqlite`.


