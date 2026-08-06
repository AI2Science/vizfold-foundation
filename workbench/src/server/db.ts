import { Database } from "bun:sqlite";
import { existsSync } from "node:fs";

import { BACKENDS, DB_PATH, RUNS_DIR } from "./env.ts";
import { join } from "node:path";
import type { Artifact, RunRow } from "../shared/types.ts";

/** Before a run lands there are no artifacts, and where the executor is writing is the answer. */
function guessRunRoot(runId: number): string | null {
  if (!RUNS_DIR) return null;
  const guess = join(RUNS_DIR, String(runId));
  return existsSync(guess) ? guess : null;
}

export const databasePresent = (): boolean => Boolean(DB_PATH) && existsSync(DB_PATH);

// A fresh install has no database until the first run; not created yet means nothing to show.
function query<T>(read: (db: Database) => T, whenAbsent: T): T {
  if (!databasePresent()) return whenAbsent;
  const db = new Database(DB_PATH, { readonly: true });
  try {
    return read(db);
  } finally {
    db.close();
  }
}

// Join the FK tables here so the API never hands out bare model_backend_id/execution_target_id.
const RUN_SELECT = `SELECT r.id, r.status, r.input_id, r.input_sequence, r.submitted_at,
    r.started_at, r.completed_at, r.error_message,
    b.slug AS model_slug, t.slug AS target_slug
  FROM runs r
  JOIN model_backends b ON b.id = r.model_backend_id
  JOIN execution_targets t ON t.id = r.execution_target_id`;

const ARTIFACT_SELECT = `SELECT a.id, a.storage_uri, a.format, a.metadata_json,
    at.slug AS type_slug, at.label AS type_label, at.display_mode, at.viewer_kind
  FROM artifacts a
  JOIN artifact_types at ON at.id = a.artifact_type_id
  WHERE a.run_id = ? ORDER BY a.id`;

// Runs from a backend this dashboard does not serve are hidden, not deleted.
const SERVED = BACKENDS?.length ? ` WHERE b.slug IN (${BACKENDS.map(() => "?").join(",")})` : "";

export const listRuns = (): RunRow[] =>
  // `IN ()` is not valid SQL, so serving nothing short-circuits before the database.
  BACKENDS?.length === 0
    ? []
    : query(
        (db) =>
          db
            .query(`${RUN_SELECT}${SERVED} ORDER BY r.submitted_at DESC, r.id DESC`)
            .all(...(BACKENDS ?? [])) as RunRow[],
        [],
      );

export const getRun = (id: number): RunRow | null =>
  query((db) => (db.query(`${RUN_SELECT} WHERE r.id = ?`).get(id) as RunRow | null) ?? null, null);

/** Where a run writes. One row rather than the run's every artifact, because every file and
 *  attention request asks this before it can resolve a path. */
export const runRootOf = (runId: number): string | null =>
  query(
    (db) =>
      (db
        .query(
          `SELECT a.storage_uri FROM artifacts a
             JOIN artifact_types at ON at.id = a.artifact_type_id
             WHERE a.run_id = ? AND at.slug = 'run_output_directory' LIMIT 1`,
        )
        .get(runId) as { storage_uri: string } | null)?.storage_uri ?? null,
    null,
  ) ?? guessRunRoot(runId);

export const listArtifacts = (runId: number): Artifact[] =>
  query((db) => db.query(ARTIFACT_SELECT).all(runId) as Artifact[], []);
