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

const run = promisify(execFile);

export type Example = {
  id: string;
  residues: number;
  description: string;
  sequence: string;
};

export async function listExamples(): Promise<Example[]> {
  const { stdout } = await run(BIN, ["list", "examples", "--json"]);
  return JSON.parse(stdout);
}

/** `--no-exec` writes only the run row, so this is near-instant. */
export async function queueRun(
  example: Example,
  attn: boolean,
  backend?: string,
): Promise<number> {
  // --attn takes a value and defaults to true, so pass it either way; the CLI reads the id and
  // sequence out of the example's FASTA itself.
  const args = ["run", example.id, `--attn=${attn}`, "--no-exec"];
  if (backend) args.push("--backend", backend);
  const { stdout } = await run(BIN, args);
  // Each backend queues under its own label — match the line's shape, not the name in it.
  const id = stdout.match(/Queued \S+ run (\d+)/)?.[1];
  if (!id) throw new Error(`no run id in run output: ${stdout.trim()}`);
  return Number(id);
}

/** Detached: a fold runs for minutes, far longer than a request may be held open. The page polls
 *  from there, and `fold` registers the artifacts itself once it lands. */
export function foldInBackground(runId: number): void {
  // Unset, `${PREFIX}/runs` is "/runs" — mkdir at the filesystem root, after the row is written.
  if (!PREFIX) throw new Error("OPENFOLD_PREFIX is unset; start the dashboard with `vizfold serve`");
  const logs = `${PREFIX}/runs`;
  mkdirSync(logs, { recursive: true });
  const log = openSync(`${logs}/${runId}.submit.log`, "a");
  const child = spawn(BIN, ["run", String(runId)], {
    detached: true,
    stdio: ["ignore", log, log],
  });
  child.unref();
}
