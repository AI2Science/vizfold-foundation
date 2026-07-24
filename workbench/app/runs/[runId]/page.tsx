import type { ReactNode } from "react";
import Link from "next/link";
import { notFound } from "next/navigation";
import path from "node:path";
import { readdirSync } from "node:fs";
import { getRun, listArtifacts, type ArtifactRow } from "@/lib/db";
import StructureViewer from "@/app/StructureViewer";

export const dynamic = "force-dynamic";

const IS_IMAGE = /\.(png|jpe?g|gif|svg|webp)$/i;
const IS_STRUCTURE = /\.(pdb|cif|ent)$/i;

type FileEntry = { name: string; url: string; isImage: boolean; isStructure: boolean };

// The run's own directory, <prefix>/runs/<id>. Its parent is the public/runs symlink target, so
// a file's browser URL is `/runs/` + its path relative to that parent.
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

function browse(dir: string, runsRoot: string): FileEntry[] {
  let abs: string[];
  try {
    abs = readdirSync(dir, { recursive: true, withFileTypes: true })
      .filter((e) => e.isFile())
      .map((e) => path.join(e.parentPath, e.name));
  } catch {
    return [];
  }
  return abs
    .map((file) => ({
      name: path.relative(dir, file),
      url: `/runs/${path.relative(runsRoot, file)}`,
      isImage: IS_IMAGE.test(file),
      isStructure: IS_STRUCTURE.test(file),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
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
  const own = runRoot(artifacts, id);
  const runsRoot = own ? path.dirname(own) : null;

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">
            <Link href="/">← All runs</Link>
          </p>
          <h1 className="brand-title">Run {run.id}</h1>
          <p className="subtitle">{run.input_id}</p>
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
          <h2>Artifacts</h2>
          <p>
            {artifacts.length} registered.
          </p>
        </div>

        {artifacts.length === 0 ? (
          <div className="empty-state">
            <p>No artifacts registered for this run.</p>
          </div>
        ) : (
          artifacts.map((artifact) => {
            const files =
              artifact.format === "directory" && runsRoot
                ? browse(artifact.storage_uri, runsRoot)
                : [];
            const directLink =
              artifact.format !== "directory" && runsRoot
                ? `/runs/${path.relative(runsRoot, artifact.storage_uri)}`
                : null;
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
                {directLink ? (
                  <ul className="file-list">
                    <li>
                      <a href={directLink}>{path.basename(artifact.storage_uri)}</a>
                      {IS_STRUCTURE.test(artifact.storage_uri) ? (
                        <StructureViewer url={directLink} name={path.basename(artifact.storage_uri)} />
                      ) : null}
                    </li>
                  </ul>
                ) : files.length === 0 ? (
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
                        {file.isStructure ? (
                          <StructureViewer url={file.url} name={file.name} />
                        ) : null}
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
