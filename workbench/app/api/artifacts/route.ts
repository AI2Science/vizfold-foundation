import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const REPO_ROOT = path.join(process.cwd(), "..");
const DB_PATH = path.join(REPO_ROOT, "data", "vizfold.db");
const PUBLIC_ARC_DIR = path.join(process.cwd(), "public", "demo-run", "arc");
const PUBLIC_TRI_DIR = path.join(process.cwd(), "public", "demo-run", "arc_tri");

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

function buildArtifactFromFile(file: string, urlPrefix: string): object {
  return {
    attentionType: getAttentionType(file),
    layer: parseLayer(file),
    head: parseHead(file),
    residueIndex: parseResidue(file),
    url: `${urlPrefix}/${file}`,
    label: `${getAttentionType(file) === "msa_row" ? "MSA Row" : "Triangle Start"} · Layer ${parseLayer(file)} · Head ${parseHead(file)}`,
  };
}

function getFallbackArtifacts(): object[] {
  const results: object[] = [];

  if (fs.existsSync(PUBLIC_ARC_DIR)) {
    const files = fs.readdirSync(PUBLIC_ARC_DIR).filter((f: string) => f.endsWith(".png"));
    for (const file of files) {
      results.push(buildArtifactFromFile(file, "/demo-run/arc"));
    }
  }

  if (fs.existsSync(PUBLIC_TRI_DIR)) {
    const files = fs.readdirSync(PUBLIC_TRI_DIR).filter((f: string) => f.endsWith(".png"));
    for (const file of files) {
      results.push(buildArtifactFromFile(file, "/demo-run/arc_tri"));
    }
  }

  return results;
}

function scanDirForPngs(dirPath: string, storageUri: string): object[] {
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) return [];
  const results: object[] = [];

  const scanSubdir = (subdir: string, subUri: string) => {
    if (!fs.existsSync(subdir)) return;
    const files = fs.readdirSync(subdir).filter((f: string) => f.endsWith(".png"));
    for (const file of files) {
      const relPath = path.join(subUri, file);
      results.push({
        attentionType: getAttentionType(file),
        layer: parseLayer(file),
        head: parseHead(file),
        residueIndex: parseResidue(file),
        url: `/api/file?path=${encodeURIComponent(relPath)}`,
        label: `${getAttentionType(file) === "msa_row" ? "MSA Row" : "Triangle Start"} · Layer ${parseLayer(file)} · Head ${parseHead(file)}`,
      });
    }
  };

  scanSubdir(path.join(dirPath, "arc"), path.join(storageUri, "arc"));
  scanSubdir(path.join(dirPath, "arc_tri"), path.join(storageUri, "arc_tri"));

  const predDir = path.join(dirPath, "predictions");
  if (fs.existsSync(predDir)) {
    const pdbFiles = fs.readdirSync(predDir).filter((f: string) => f.endsWith(".pdb"));
    for (const file of pdbFiles) {
      const relPath = path.join(storageUri, "predictions", file);
      results.push({
        attentionType: null,
        layer: null,
        head: null,
        residueIndex: null,
        url: `/api/file?path=${encodeURIComponent(relPath)}`,
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
    // Try database first
    if (fs.existsSync(DB_PATH)) {
      let Database;
      try {
        Database = (await import("better-sqlite3")).default;
      } catch {
        // better-sqlite3 not available, fall through to static fallback
      }

      if (Database) {
        const db = new Database(DB_PATH, { readonly: true });
        const runs = db.prepare(
          "SELECT * FROM runs ORDER BY submitted_at DESC LIMIT 1"
        ).all() as Array<{ id: number; input_id: string; status: string }>;

        if (runs.length > 0) {
          const run = runs[0];
          const artifacts = db.prepare(
            "SELECT * FROM artifacts WHERE run_id = ?"
          ).all(run.id) as Array<{ storage_uri: string }>;
          db.close();

          const allArtifacts: object[] = [];
          for (const artifact of artifacts) {
            const absPath = path.join(REPO_ROOT, artifact.storage_uri);
            allArtifacts.push(...scanDirForPngs(absPath, artifact.storage_uri));
          }

          // If database has artifacts, return them
          if (allArtifacts.length > 0) {
            return NextResponse.json({
              runId: run.input_id,
              status: run.status,
              model: "OpenFold",
              protein: run.input_id.split("_")[0],
              artifacts: allArtifacts,
              source: "database",
            });
          }
        } else {
          const db2 = new Database(DB_PATH, { readonly: true });
          db2.close();
        }
      }
    }

    // Fallback to static public/demo-run folder
    const fallbackArtifacts = getFallbackArtifacts();
    return NextResponse.json({
      runId: "openfold-demo-run",
      status: "demo",
      model: "OpenFold",
      protein: "6KWC",
      artifacts: fallbackArtifacts,
      source: "static-fallback",
    });

  } catch (error) {
    // Always fall back to static artifacts on any error
    const fallbackArtifacts = getFallbackArtifacts();
    return NextResponse.json({
      runId: "openfold-demo-run",
      status: "demo",
      model: "OpenFold",
      protein: "6KWC",
      artifacts: fallbackArtifacts,
      source: "static-fallback",
    });
  }
}
