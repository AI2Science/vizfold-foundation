import { execFile, spawn } from "node:child_process";
import { mkdirSync, openSync } from "node:fs";
import { promisify } from "node:util";

// The CLI owns the executor; the dashboard never parses FASTA or writes the database.
// `vizfold serve` exports VIZFOLD_BIN so this resolves without relying on ~/.local/bin.
const BIN = process.env.VIZFOLD_BIN ?? "vizfold";
const PREFIX = process.env.OPENFOLD_PREFIX ?? "";

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

/** Record the run and return its id. `--no-exec` only writes the row, so this is near-instant. */
export async function queueRun(example: Example, attn: boolean): Promise<number> {
  // --attn takes a value and defaults to true, so it has to be passed either way; the CLI reads
  // the id and the sequence out of the example's FASTA itself.
  const args = ["run", example.id, "--backend", "openfold", `--attn=${attn}`, "--no-exec"];
  const { stdout } = await run(BIN, args);
  const id = stdout.match(/Queued OpenFold run (\d+)/)?.[1];
  if (!id) throw new Error(`no run id in queue output: ${stdout.trim()}`);
  return Number(id);
}

/** Detached: a fold runs for minutes, far longer than a request may be held open. The page polls
 *  from there, and `fold` registers the artifacts itself once it lands. */
export function foldInBackground(runId: number): void {
  // Without a prefix this used to resolve to "/runs" and try to mkdir at the filesystem root,
  // 500ing after the run row was already written.
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
