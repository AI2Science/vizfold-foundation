import { Database } from "bun:sqlite";
import { existsSync } from "node:fs";

import { BACKENDS, DB_PATH } from "./env.ts";
import type { Artifact, RunRow } from "../shared/types.ts";

// A fresh install has no database until the first run; not created yet means nothing to show.
function query<T>(read: (db: Database) => T, whenAbsent: T): T {
  if (!DB_PATH || !existsSync(DB_PATH)) return whenAbsent;
  const db = new Database(DB_PATH, { readonly: true });
  try {
    return read(db);
  } finally {
    db.close();
  }
}

export const databasePresent = (): boolean => Boolean(DB_PATH) && existsSync(DB_PATH);

// Join the FK tables here so the API never hands out bare model_backend_id/execution_target_id.
const RUN_SELECT = `SELECT r.id, r.status, r.input_id, r.input_sequence, r.submitted_at,
    r.started_at, r.completed_at, r.error_message,
    b.slug AS model_slug, t.slug AS target_slug
  FROM runs r
  JOIN model_backends b ON b.id = r.model_backend_id
  JOIN execution_targets t ON t.id = r.execution_target_id`;

const ARTIFACT_SELECT = `SELECT a.id, a.format, a.storage_uri,
    at.slug AS type_slug, at.label AS type_label, at.viewer_kind, at.display_mode
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

export const listArtifacts = (runId: number): Artifact[] =>
  query((db) => db.query(ARTIFACT_SELECT).all(runId) as Artifact[], []);
