import type { ReactNode } from "react";
import Link from "next/link";
import { notFound } from "next/navigation";
import path from "node:path";
import { readdirSync } from "node:fs";
import { getRun, listArtifacts, type ArtifactRow } from "@/lib/db";
import { RUNS_DIR } from "@/lib/vizfold";
import StructureViewer from "@/app/StructureViewer";
import Poller from "@/app/Poller";

export const dynamic = "force-dynamic";

const IS_IMAGE = /\.(png|jpe?g|gif|svg|webp)$/i;
const IS_STRUCTURE = /\.(pdb|cif|ent)$/i;

type FileEntry = { name: string; url: string; isImage: boolean; isStructure: boolean };

// <prefix>/runs/<id>. Its parent is the public/runs symlink target, so a file's browser URL is
// `/runs/` + its path relative to that parent.
function runRoot(artifacts: ArtifactRow[], id: number): string | null {
  const own = artifacts.find((a) => a.type_slug === "run_output_directory");
  if (own) return own.storage_uri;
  const marker = `/runs/${id}`;
  for (const a of artifacts) {
    const i = a.storage_uri.indexOf(marker);
    if (i >= 0) return a.storage_uri.slice(0, i + marker.length);
  }
  return null;
}

const entry = (file: string, runsRoot: string, name = path.basename(file)): FileEntry => ({
  name,
  url: `/runs/${path.relative(runsRoot, file)}`,
  isImage: IS_IMAGE.test(file),
  isStructure: IS_STRUCTURE.test(file),
});

function browse(dir: string, runsRoot: string): FileEntry[] {
  try {
    return readdirSync(dir, { recursive: true, withFileTypes: true })
      .filter((e) => e.isFile())
      .map((e) => path.join(e.parentPath, e.name))
      .map((file) => entry(file, runsRoot, path.relative(dir, file)))
      .sort((a, b) => a.name.localeCompare(b.name));
  } catch {
    return [];
  }
}

export default async function RunPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  const id = Number(runId);
  const run = getRun(id);
  if (!run) notFound();

  const artifacts = listArtifacts(id);
  // Artifacts register only once the run lands, so fall back to where the executor is writing —
  // structures show up per protein while the fold is still going.
  const own = runRoot(artifacts, id) ?? path.join(RUNS_DIR, String(id));
  const runsRoot = path.dirname(own);
  const structures = browse(own, runsRoot).filter((file) => file.isStructure);

  // A batch writes one structure per FASTA tag, and the run records those tags joined with `+`.
  // The tag ends at `_model_`: a bare prefix test would give 1G1J_1 every 1G1J_10 file too.
  const tagOf = (name: string) => name.split("_model_")[0];

  const tags = run.input_id.split("+");
  const folds = tags.map((tag) => {
    // A lone target owns every file: ESMFold writes `structure/predicted.pdb`, without the tag.
    const landed =
      tags.length === 1
        ? structures
        : structures.filter((file) => tagOf(path.basename(file.name)) === tag);
    // The relaxed structure is the one to look at; the rest stay linked under Artifacts.
    return { tag, file: landed.find((file) => !file.name.includes("unrelaxed")) ?? landed[0] };
  });

  return (
    <main className="page-shell">
      <Poller statuses={[run.status]} />
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">
            <Link href="/">← All runs</Link>
          </p>
          <h1 className="brand-title">Run {run.id}</h1>
          <p className="subtitle">{tags.join(", ")}</p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Details</h2>
        </div>
        <div className="result-card">
          <Row label="Status">
            <span className={`status status-${run.status}`}>{run.status}</span>
          </Row>
          <Row label="Model">{run.model_slug}</Row>
          <Row label="Target">{run.target_slug}</Row>
          <Row label="Submitted">{run.submitted_at}</Row>
          <Row label="Started">{run.started_at ?? "—"}</Row>
          <Row label="Completed">{run.completed_at ?? "—"}</Row>
          {run.error_message ? <Row label="Error">{run.error_message}</Row> : null}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Structures</h2>
          <p>
            {folds.filter((fold) => fold.file).length} of {tags.length} landed.
          </p>
        </div>

        {folds.map((fold) => (
          <div key={fold.tag} className="artifact-block">
            <h3>
              {fold.tag}{" "}
              <span className="field-note">
                {fold.file
                  ? fold.file.name
                  : run.status === "failed"
                    ? "no structure written"
                    : "folding…"}
              </span>
            </h3>
            {fold.file ? <StructureViewer url={fold.file.url} name={fold.file.name} /> : null}
          </div>
        ))}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Artifacts</h2>
          <p>
            {artifacts.length} registered.
          </p>
        </div>

        {artifacts.length === 0 ? (
          <div className="empty-state">
            <p>No artifacts registered for this run.</p>
            <p>They register when the run lands.</p>
          </div>
        ) : (
          artifacts.map((artifact) => {
            // A file artifact is just a one-entry listing, so both kinds render the same way.
            const files =
              artifact.format === "directory"
                ? browse(artifact.storage_uri, runsRoot)
                : [entry(artifact.storage_uri, runsRoot)];
            return (
              <div key={artifact.id} className="artifact-block">
                <h3>
                  {artifact.type_label}{" "}
                  <span className="field-note">
                    {artifact.format === "directory"
                      ? `(${files.length} file${files.length === 1 ? "" : "s"})`
                      : `(${artifact.format})`}
                  </span>
                </h3>
                {files.length === 0 ? (
                  <p className="field-note">Empty.</p>
                ) : (
                  <ul className="file-list">
                    {files.map((file) => (
                      <li key={file.url}>
                        <a href={file.url}>
                          {file.isImage ? (
                            <img src={file.url} alt={file.name} className="file-thumb" />
                          ) : null}
                          {file.name}
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })
        )}
      </section>
    </main>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="result-row">
      <span>{label}</span>
      <strong>{children}</strong>
    </div>
  );
}
