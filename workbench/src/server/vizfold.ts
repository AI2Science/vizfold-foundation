import { spawn } from "node:child_process";
import { mkdirSync, openSync } from "node:fs";

import { BIN, PREFIX, RUNS_DIR } from "./env.ts";
import type { Protein } from "../shared/types.ts";

/** Run the CLI and capture its stdout. Anything non-zero carries the CLI's own message, which is
 *  what the dashboard shows: it knows why a fold cannot start, and the dashboard does not. */
async function capture(args: string[]): Promise<string> {
  const child = Bun.spawn([BIN, ...args], { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (code !== 0) {
    throw new Error((stderr.trim() || stdout.trim() || `${BIN} ${args.join(" ")} failed`).trim());
  }
  return stdout;
}

export async function listProteins(): Promise<Protein[]> {
  return JSON.parse(await capture(["list", "proteins", "--json"])) as Protein[];
}

/** One run for the whole selection — the CLI folds every target in a single execution, with the
 *  model loaded once. `--no-exec` writes only the run row, so this is near-instant. */
export async function queueRun(ids: string[], attn: boolean, backend?: string): Promise<number> {
  // --attn takes a value and defaults to true, so pass it either way; the CLI reads each id's
  // sequence out of its FASTA itself.
  const args = ["run", ...ids, `--attn=${attn}`, "--no-exec", "--json"];
  if (backend) args.push("--backend", backend);
  const stdout = await capture(args);
  const { run_id: id } = JSON.parse(stdout) as { run_id?: unknown };
  if (typeof id !== "number") throw new Error(`no run id in run output: ${stdout.trim()}`);
  return id;
}

/** Detached: a fold runs for minutes, far longer than a request may be held open. The client polls
 *  from there, and `fold` registers the artifacts itself once it lands. */
export function foldInBackground(runId: number): void {
  // Unset, RUNS_DIR is "" — the log would land in the working directory, after the row is written.
  if (!PREFIX) {
    throw new Error("OPENFOLD_PREFIX is unset; start the dashboard with `vizfold serve`");
  }
  mkdirSync(RUNS_DIR, { recursive: true });
  const log = openSync(`${RUNS_DIR}/${runId}.submit.log`, "a");
  // node:child_process, not Bun.spawn: `detached` is what outlives this server, and a fold has to.
  const child = spawn(BIN, ["run", String(runId)], {
    detached: true,
    stdio: ["ignore", log, log],
  });
  child.unref();
}
