# Working on the workbench

The runtime is **Bun**, not Node: `Bun.serve` with an HTML entry point does the bundling, and
`bun:sqlite` reads the executor's database. There is no Next.js, no bundler config and no build
step — `bun src/server/index.ts` is the whole server. Do not reach for `node:sqlite`, a React
framework, or a CSS framework; check `node_modules/@base-ui-components/react` before writing a
component, since Base UI is unstyled by design and every look here comes from `src/client/app.css`.

Two rules the dashboard is built on:

1. **Nothing is mocked.** Every number, file and diagram comes from the executor — its SQLite rows,
   its run directories, or the `vizfold` binary. A panel that has no data does not render; it says
   what is missing and where it looked. Never check a sample output into the repo to make a view
   look populated.
2. **What a file is, is the executor's answer, not a guess here.** Every file a run produces is
   registered with a kind (`cli/src/core/artifact_kinds.rs`), and the artifact rows carry the
   viewer, the display mode and the coordinates read off the name — target, layer, attention type.
   Read those. The extension table in `runfiles.ts` is the fallback for a run still going, which
   has registered nothing yet; it is not the rule.
3. **Diagrams are computed from the run, not stored beside it.** Attention arcs are drawn in the
   browser from the text dumps the backends write. If a view needs a new derived value, parse it on
   the server from what the run wrote.

Colours come from the validated data-viz palette already in `app.css` (`--seq-low`/`--seq-high` for
magnitude, the status tokens for run state). Both modes are selected, not flipped; a chart reads its
colours off the document so a theme switch repaints it.
