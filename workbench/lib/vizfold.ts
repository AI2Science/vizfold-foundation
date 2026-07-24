import { execFile, spawn } from "node:child_process";
import { mkdirSync, openSync } from "node:fs";
import { promisify } from "node:util";

// The CLI owns the executor; the dashboard never parses FASTA or writes the database itself.
// `vizfold serve` exports VIZFOLD_BIN so this resolves without depending on ~/.local/bin.
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

/** Record the run and return its id. Queueing only writes the row, so this is near-instant. */
export async function queueRun(example: Example, attn: boolean): Promise<number> {
  const args = [
    "queue-run",
    "openfold",
    "--input-id",
    example.id,
    "--input-sequence",
    example.sequence,
    ...(attn ? ["--demo-attn"] : []),
  ];
  const { stdout } = await run(BIN, args);
  const id = stdout.match(/Queued OpenFold run (\d+)/)?.[1];
  if (!id) throw new Error(`no run id in queue-run output: ${stdout.trim()}`);
  return Number(id);
}

/**
 * Fold in the background. A fold runs for minutes, far longer than a request should be held open,
 * so the response returns as soon as the run exists and the page polls the status from there.
 * `execute-run` registers the artifacts itself once the fold lands.
 */
export function foldInBackground(runId: number): void {
  const logs = `${PREFIX}/runs`;
  mkdirSync(logs, { recursive: true });
  const log = openSync(`${logs}/${runId}.submit.log`, "a");
  const child = spawn(BIN, ["execute-run", String(runId)], {
    detached: true,
    stdio: ["ignore", log, log],
  });
  child.unref();
}
