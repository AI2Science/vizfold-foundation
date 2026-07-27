import { execFile, spawn } from "node:child_process";
import { mkdirSync, openSync } from "node:fs";
import { promisify } from "node:util";

// The CLI owns the executor; the dashboard never parses FASTA or writes the database.
// `vizfold serve` exports VIZFOLD_BIN, so this resolves without relying on ~/.local/bin.
const BIN = process.env.VIZFOLD_BIN ?? "vizfold";
const PREFIX = process.env.OPENFOLD_PREFIX ?? "";

// Unset means `next dev` by hand: filter nothing, since filtering by a set nobody chose would hide
// every run. Set-but-empty is a real answer — `serve` found no backend installed.
const served = process.env.VIZFOLD_BACKENDS;
export const BACKENDS: string[] | null =
  served === undefined ? null : served.split(",").filter(Boolean);

/** What the Fold card offers. Unset, offer both and let the CLI's prereq gate refuse an
 *  uninstalled one. */
export const FOLDABLE = BACKENDS ?? ["openfold", "esmfold"];

/** Where the executor writes each run: `<RUNS_DIR>/<run id>`, alongside its submit log. */
export const RUNS_DIR = `${PREFIX}/runs`;

const run = promisify(execFile);

export type Protein = {
  id: string;
  residues: number;
  description: string;
  sequence: string;
  /** `alignments/<id>` is there to reuse; false pays for the full MSA search. */
  alignments: boolean;
};

export async function listProteins(): Promise<Protein[]> {
  const { stdout } = await run(BIN, ["list", "proteins", "--json"]);
  return JSON.parse(stdout);
}

/** One run for the whole selection — the CLI folds every target in a single execution, with the
 *  model loaded once. `--no-exec` writes only the run row, so this is near-instant. */
export async function queueRun(
  ids: string[],
  attn: boolean,
  backend?: string,
): Promise<number> {
  // --attn takes a value and defaults to true, so pass it either way; the CLI reads each id's
  // sequence out of its FASTA itself.
  const args = ["run", ...ids, `--attn=${attn}`, "--no-exec", "--json"];
  if (backend) args.push("--backend", backend);
  const { stdout } = await run(BIN, args);
  const { run_id: id } = JSON.parse(stdout);
  if (typeof id !== "number") throw new Error(`no run id in run output: ${stdout.trim()}`);
  return id;
}

/** Detached: a fold runs for minutes, far longer than a request may be held open. The page polls
 *  from there, and `fold` registers the artifacts itself once it lands. */
export function foldInBackground(runId: number): void {
  // Unset, RUNS_DIR is "/runs" — mkdir at the filesystem root, after the row is written.
  if (!PREFIX) throw new Error("OPENFOLD_PREFIX is unset; start the dashboard with `vizfold serve`");
  mkdirSync(RUNS_DIR, { recursive: true });
  const log = openSync(`${RUNS_DIR}/${runId}.submit.log`, "a");
  const child = spawn(BIN, ["run", String(runId)], {
    detached: true,
    stdio: ["ignore", log, log],
  });
  child.unref();
}
