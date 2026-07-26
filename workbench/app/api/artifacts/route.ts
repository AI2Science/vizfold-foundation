import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import Database from "better-sqlite3";

const REPO_ROOT = path.join(process.cwd(), "..", "..", "..");
const DB_PATH = path.join(REPO_ROOT, "data", "vizfold.db");

function parseLayer(filename: string): number {
  const match = filename.match(/layer[_-]?(\d+)/i);
  return match ? parseInt(match[1]) : -1;
}

function parseHead(filename: string): number {
  const match = filename.match(/head[_-]?(\d+)/i);
  return match ? parseInt(match[1]) : -1;
}

function parseResidue(filename: string): number | null {
  const match = filename.match(/res[_-]?(\d+)/i);
  return match ? parseInt(match[1]) : null;
}

function getAttentionType(filename: string): string {
  if (filename.includes("msa_row")) return "msa_row";
  if (filename.includes("triangle_start") || filename.includes("tri_start")) return "triangle_start";
  return "unknown";
}

function scanDirForPngs(dirPath: string, storageUri: string): object[] {
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) return [];

  const results: object[] = [];

  const scanSubdir = (subdir: string, subUri: string) => {
    if (!fs.existsSync(subdir)) return;
    const files = fs.readdirSync(subdir).filter((f: string) => f.endsWith(".png"));
    for (const file of files) {
      const attentionType = getAttentionType(file);
      const layer = parseLayer(file);
      const head = parseHead(file);
      const residue = parseResidue(file);
      const relPath = path.join(subUri, file);
      const publicPath = `/api/file?path=${encodeURIComponent(relPath)}`;
      results.push({
        attentionType,
        layer,
        head,
        residueIndex: residue,
        url: publicPath,
        label: `${attentionType === "msa_row" ? "MSA Row" : "Triangle Start"} · Layer ${layer} · Head ${head}`,
      });
    }
  };

  // Scan arc and arc_tri subdirectories
  scanSubdir(path.join(dirPath, "arc"), path.join(storageUri, "arc"));
  scanSubdir(path.join(dirPath, "arc_tri"), path.join(storageUri, "arc_tri"));

  // Also scan the directory itself for any PNGs at the top level
  const topFiles = fs.readdirSync(dirPath).filter((f: string) => f.endsWith(".png"));
  for (const file of topFiles) {
    const attentionType = getAttentionType(file);
    const layer = parseLayer(file);
    const head = parseHead(file);
    const residue = parseResidue(file);
    const relPath = path.join(storageUri, file);
    const publicPath = `/api/file?path=${encodeURIComponent(relPath)}`;
    results.push({
      attentionType,
      layer,
      head,
      residueIndex: residue,
      url: publicPath,
      label: `${attentionType === "msa_row" ? "MSA Row" : "Triangle Start"} · Layer ${layer} · Head ${head}`,
    });
  }

  // Scan predictions subfolder for PDB files
  const predDir = path.join(dirPath, "predictions");
  if (fs.existsSync(predDir)) {
    const pdbFiles = fs.readdirSync(predDir).filter((f: string) => f.endsWith(".pdb"));
    for (const file of pdbFiles) {
        const relPath = path.join(storageUri, "predictions", file);
        const publicPath = `/api/file?path=${encodeURIComponent(relPath)}`;
        results.push({
            attentionType: null,
            layer: null,
            head: null,
            residueIndex: null,
            url: publicPath,
            label: file,
            artifactType: "protein_structure",
            displayMode: "download",
        });
    }
}

  return results;
}

export async function GET() {
  try {
    if (!fs.existsSync(DB_PATH)) {
      return NextResponse.json(
        { error: "Executor database not found at " + DB_PATH },
        { status: 404 }
      );
    }

    const db = new Database(DB_PATH, { readonly: true });

    const runs = db.prepare(
      "SELECT * FROM runs ORDER BY submitted_at DESC LIMIT 1"
    ).all() as Array<{
      id: number;
      input_id: string;
      status: string;
      submitted_at: string;
    }>;

    if (runs.length === 0) {
      db.close();
      return NextResponse.json({ error: "No runs found in database" }, { status: 404 });
    }

    const run = runs[0];

    const artifacts = db.prepare(
      "SELECT * FROM artifacts WHERE run_id = ?"
    ).all(run.id) as Array<{
      id: number;
      run_id: number;
      artifact_type_id: number;
      format: string;
      storage_uri: string;
      metadata_json: string;
    }>;

    db.close();

    const allArtifacts: object[] = [];

    for (const artifact of artifacts) {
      const absPath = path.join(REPO_ROOT, artifact.storage_uri);
      const pngs = scanDirForPngs(absPath, artifact.storage_uri);
      allArtifacts.push(...pngs);
    }

    return NextResponse.json({
      runId: run.input_id,
      status: run.status,
      model: "OpenFold",
      protein: run.input_id.split("_")[0],
      artifacts: allArtifacts,
    });
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to read executor database", detail: String(error) },
      { status: 500 }
    );
  }
}
