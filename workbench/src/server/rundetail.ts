import { basename } from "node:path";

import { getRun, listArtifacts } from "./db.ts";
import { inventory, sequenceFor } from "./attention.ts";
import { listFiles, runRoot } from "./runfiles.ts";
import { readActivations } from "./traces.ts";
import type { Artifact, FoldTarget, RunDetail, RunFile, RunRow } from "../shared/types.ts";

/** A batch writes one structure per FASTA tag, and the run records those tags joined with `+`.
 *  The tag ends at `_model_`: a bare prefix test would give 1G1J_1 every 1G1J_10 file too. */
const tagOf = (name: string) => basename(name).split("_model_")[0] ?? "";

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
 *  costs a walk of the run directory. */
export function openRun(id: number): { run: RunRow; artifacts: Artifact[]; root: string | null } | null {
  const run = getRun(id);
  if (!run) return null;
  const artifacts = listArtifacts(id);
  // Artifacts register only once the run lands, so fall back to where the executor is writing —
  // structures show up per target while the fold is still going.
  return { run, artifacts, root: runRoot(artifacts, id) };
}

export async function readRunDetail(id: number): Promise<RunDetail | null> {
  const open = openRun(id);
  if (!open) return null;
  const { run, artifacts, root } = open;
  const { files, truncated } = root ? listFiles(root) : { files: [], truncated: false };
  const structures = files.filter((file) => file.kind === "structure");

  const tags = run.input_id.split("+");
  const targets: FoldTarget[] = tags.map((tag) => {
    // A lone target owns every file: ESMFold writes `structure/predicted.pdb`, without the tag.
    const own = tags.length === 1 ? structures : structures.filter((file) => tagOf(file.name) === tag);
    return {
      tag,
      sequence: sequenceFor(run.input_id, run.input_sequence, tags.length === 1 ? null : tag),
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
