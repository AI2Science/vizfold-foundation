import { DatabaseSync } from "node:sqlite";
import { existsSync } from "node:fs";
import { BACKENDS } from "@/lib/vizfold";

// The Rust executor owns this file; the dashboard only reads it. `vizfold serve` exports VIZFOLD_DB
// as a plain path — node:sqlite cannot open the CLI's `sqlite://...?mode=rwc` form. The fallback
// covers running `next dev` by hand.
const dbPath =
  process.env.VIZFOLD_DB ?? `${process.env.OPENFOLD_PREFIX ?? ""}/vizfold.db`;

export type RunRow = {
  id: number;
  status: string;
  input_id: string;
  input_sequence: string;
  model_slug: string;
  target_slug: string;
  submitted_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
};

export type ArtifactRow = {
  id: number;
  format: string;
  storage_uri: string;
  type_slug: string;
  type_label: string;
  viewer_kind: string;
  display_mode: string;
};

// A fresh install has no db until the first run; not created yet means nothing to show.
function query<T>(read: (db: DatabaseSync) => T, whenAbsent: T): T {
  if (!existsSync(dbPath)) return whenAbsent;
  const db = new DatabaseSync(dbPath, { readOnly: true });
  try {
    return read(db);
  } finally {
    db.close();
  }
}

// Join the FK tables here so pages never see bare model_backend_id/execution_target_id.
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
            .prepare(`${RUN_SELECT}${SERVED} ORDER BY r.submitted_at DESC`)
            .all(...(BACKENDS ?? [])) as RunRow[],
        [],
      );

export const getRun = (id: number): RunRow | null =>
  query((db) => (db.prepare(`${RUN_SELECT} WHERE r.id = ?`).get(id) as RunRow) ?? null, null);

export const listArtifacts = (runId: number): ArtifactRow[] =>
  query((db) => db.prepare(ARTIFACT_SELECT).all(runId) as ArtifactRow[], []);
