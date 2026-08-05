import { join } from "node:path";

/** The CLI owns the executor; the dashboard never parses FASTA or writes the database.
 *  `vizfold serve` exports VIZFOLD_BIN, so this resolves without relying on ~/.local/bin. */
export const BIN = process.env.VIZFOLD_BIN ?? "vizfold";

export const PREFIX = process.env.OPENFOLD_PREFIX ?? "";

/** Where the executor writes each run: `<RUNS_DIR>/<run id>`, alongside its submit log. */
export const RUNS_DIR = PREFIX ? join(PREFIX, "runs") : "";

/** The Rust executor owns this file; the dashboard only reads it. `vizfold serve` exports
 *  VIZFOLD_DB as a plain path. The fallback covers running `bun dev` by hand. */
export const DB_PATH = process.env.VIZFOLD_DB ?? (PREFIX ? join(PREFIX, "vizfold.db") : "");

/** Unset means `bun dev` by hand: filter nothing, since filtering by a set nobody chose would hide
 *  every run. Set-but-empty is a real answer — `serve` found no backend installed. */
const served = process.env.VIZFOLD_BACKENDS;

export const BACKENDS: string[] | null =
  served === undefined ? null : served.split(",").filter(Boolean);

/** What the fold form offers. Unset, offer both and let the CLI's prereq gate refuse an
 *  uninstalled one. */
export const FOLDABLE: string[] = BACKENDS ?? ["openfold", "esmfold"];

export const PORT = Number(process.env.PORT ?? process.env.VIZFOLD_PORT ?? 3000);
