import { DatabaseSync } from "node:sqlite";
import { existsSync } from "node:fs";

// The Rust executor owns this file; the dashboard reads it directly, read-only. `vizfold serve`
// exports VIZFOLD_DB (the plain sqlite path); the fallback covers running `next dev` by hand.
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

// A fresh install has no db until the first run; treat "not created yet" as "nothing to show".
function open(): DatabaseSync | null {
  return existsSync(dbPath) ? new DatabaseSync(dbPath, { readOnly: true }) : null;
}

// Join the FK tables here so pages never see bare model_backend_id/execution_target_id.
const RUN_SELECT = `SELECT r.id, r.status, r.input_id, r.input_sequence, r.submitted_at,
    r.started_at, r.completed_at, r.error_message,
    b.slug AS model_slug, t.slug AS target_slug
  FROM runs r
  JOIN model_backends b ON b.id = r.model_backend_id
  JOIN execution_targets t ON t.id = r.execution_target_id`;

export function listRuns(): RunRow[] {
  const db = open();
  if (!db) return [];
  try {
    return db.prepare(`${RUN_SELECT} ORDER BY r.submitted_at DESC`).all() as RunRow[];
  } finally {
    db.close();
  }
}

export function getRun(id: number): RunRow | null {
  const db = open();
  if (!db) return null;
  try {
    return (db.prepare(`${RUN_SELECT} WHERE r.id = ?`).get(id) as RunRow) ?? null;
  } finally {
    db.close();
  }
}

export function listArtifacts(runId: number): ArtifactRow[] {
  const db = open();
  if (!db) return [];
  try {
    return db
      .prepare(
        `SELECT a.id, a.format, a.storage_uri,
            at.slug AS type_slug, at.label AS type_label, at.viewer_kind, at.display_mode
          FROM artifacts a
          JOIN artifact_types at ON at.id = a.artifact_type_id
          WHERE a.run_id = ? ORDER BY a.id`,
      )
      .all(runId) as ArtifactRow[];
  } finally {
    db.close();
  }
}
