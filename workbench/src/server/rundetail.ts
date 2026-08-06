import { relative } from "node:path";

import { getRun, listArtifacts, runRootOf } from "./db.ts";
import { inventory, sequenceFor } from "./attention.ts";
import { listFiles } from "./runfiles.ts";
import { readActivations } from "./traces.ts";
import type { Artifact, FoldTarget, RunDetail, RunFile, RunRow } from "../shared/types.ts";

/** A batch writes one structure per FASTA tag, and the run records those tags joined with `+`.
 *  The tag ends at `_model_`: a bare prefix test would give 1G1J_1 every 1G1J_10 file too. */
const tagOf = (name: string) => name.split("_model_")[0] ?? "";

/** What the 3D viewer can actually open. A kind classifies; a format decides whether it renders. */
const EMBEDDABLE_STRUCTURE = new Set(["pdb", "cif", "ent"]);

/** The relaxed structure is the one to look at; the rest stay listed beside it. */
function preferred(structures: RunFile[]): RunFile | null {
  return (
    structures.find((file) => /(^|[^n])relaxed/i.test(file.name) && !/unrelaxed/i.test(file.name)) ??
    structures.find((file) => !/unrelaxed/i.test(file.name)) ??
    structures[0] ??
    null
  );
}

/** The run row and where it writes — everything a request for one file needs, and nothing that
 *  costs a walk of the run directory or a scan of its every artifact. */
export function openRun(id: number): { run: RunRow; root: string | null } | null {
  const run = getRun(id);
  if (!run) return null;
  // Artifacts register only once the run lands, so fall back to where the executor is writing —
  // structures show up per target while the fold is still going.
  return { run, root: runRootOf(id) };
}

/** The registered artifacts, pathed against the run root so the client can link them, and sized
 *  from the listing that already walked the directory rather than by asking again. */
function artifactsOf(id: number, root: string | null, files: RunFile[]): Artifact[] {
  const sizes = new Map(files.map((file) => [file.path, file.size]));
  return listArtifacts(id).flatMap((row) => {
    const raw = row as unknown as Artifact & { metadata_json?: string };
    // Directory rows are how a run is found, not results anyone asked for. The workspace is one;
    // a database seeded before per-file registration also holds `attention/`.
    if (raw.format === "directory") return [];
    const path = root ? relative(root, raw.storage_uri) : raw.storage_uri;
    let metadata: Record<string, unknown> = {};
    try {
      metadata = raw.metadata_json ? JSON.parse(raw.metadata_json) : {};
    } catch {
      // A row written by a version that recorded something else is still a row; it just carries
      // no coordinates, and everything downstream treats them as optional.
      metadata = {};
    }
    return [{ ...raw, path, size: sizes.get(path) ?? null, metadata }];
  });
}

export async function readRunDetail(id: number): Promise<RunDetail | null> {
  const open = openRun(id);
  if (!open) return null;
  const { run, root } = open;
  const { files, truncated } = root ? listFiles(root) : { files: [], truncated: false };
  const artifacts = artifactsOf(id, root, files);

  // The kind says what a file is; the format says whether a viewer can open it. ESMFold writes
  // its coordinates as a torch pickle — a protein structure, and not one 3Dmol can read.
  const registered = artifacts.filter(
    (artifact) =>
      artifact.type_slug === "protein_structure" && EMBEDDABLE_STRUCTURE.has(artifact.format),
  );
  // Union, not either-or: registration is a snapshot, so a structure written after the last one
  // is still on disk and still belongs on the page.
  const byPath = new Set(registered.map((artifact) => artifact.path));
  const structures = files.filter((file) => byPath.has(file.path) || file.kind === "structure");
  const targetOf = (file: RunFile) =>
    (registered.find((artifact) => artifact.path === file.path)?.metadata.target as
      | string
      | undefined) ?? tagOf(file.name);

  const tags = run.input_id.split("+");
  const targets: FoldTarget[] = tags.map((tag) => {
    // A lone target owns every file: ESMFold writes `structure/predicted.pdb`, without the tag.
    const own = tags.length === 1 ? structures : structures.filter((file) => targetOf(file) === tag);
    return {
      tag,
      sequence: sequenceFor(run.input_id, run.input_sequence, tag),
      structure: preferred(own),
      structures: own,
    };
  });

  return {
    run,
    artifacts,
    root,
    targets,
    attention: inventory(files),
    activations: root
      ? await readActivations(root, files)
      : { tensors: [], attentionStats: [], activationStats: [], meta: null, arrays: [] },
    files,
    filesTruncated: truncated,
  };
}
