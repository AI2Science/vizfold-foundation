import { existsSync, readdirSync, realpathSync, statSync } from "node:fs";
import { isAbsolute, join, normalize, relative, resolve, sep } from "node:path";

import { RUNS_DIR } from "./env.ts";
import type { Artifact, FileKind, RunFile } from "../shared/types.ts";

const KINDS: [RegExp, FileKind][] = [
  [/\.(pdb|cif|ent)$/i, "structure"],
  [/\.(png|jpe?g|gif|svg|webp)$/i, "image"],
  [/\.(pt|pth|npz|npy|pkl)$/i, "tensor"],
  [/\.(zip|tar|tgz|gz|bz2)$/i, "archive"],
  [/\.(txt|json|log|md|csv|tsv|ya?ml|fasta|fa|a3m|sto|hhr)$/i, "text"],
];

export const kindOf = (name: string): FileKind =>
  KINDS.find(([pattern]) => pattern.test(name))?.[1] ?? "other";

/** How many files one listing walks before it stops. A run that writes more than this is reported
 *  as truncated rather than quietly cut short. */
const FILE_LIMIT = 4000;

/** `<prefix>/runs/<id>`: what the artifact rows point at once the run lands, and where the
 *  executor is writing before then. */
export function runRoot(artifacts: Artifact[], id: number): string | null {
  const own = artifacts.find((artifact) => artifact.type_slug === "run_output_directory");
  if (own) return own.storage_uri;
  const marker = `${sep}runs${sep}${id}`;
  for (const artifact of artifacts) {
    const at = artifact.storage_uri.indexOf(marker);
    if (at >= 0) return artifact.storage_uri.slice(0, at + marker.length);
  }
  if (!RUNS_DIR) return null;
  const guess = join(RUNS_DIR, String(id));
  return existsSync(guess) ? guess : null;
}

/** Every file under `root`, relative-pathed, oldest name first. Depth-first and bounded: an
 *  activation dump is thousands of tensors, and the browser has to render the list. */
export function listFiles(root: string, limit = FILE_LIMIT): { files: RunFile[]; truncated: boolean } {
  const files: RunFile[] = [];
  let truncated = false;
  const walk = (dir: string) => {
    if (truncated) return;
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (truncated) return;
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
        continue;
      }
      if (!entry.isFile()) continue;
      if (files.length >= limit) {
        truncated = true;
        return;
      }
      const stats = statSync(full, { throwIfNoEntry: false });
      if (!stats) continue;
      const path = relative(root, full);
      files.push({
        path,
        name: entry.name,
        size: stats.size,
        modified: stats.mtime.toISOString(),
        kind: kindOf(entry.name),
      });
    }
  };
  walk(root);
  return { files, truncated };
}

/** Resolve a client-supplied relative path inside `root`, or null when it points outside it.
 *  Both the joined path and its realpath are checked: `..` and a symlink out both land outside. */
export function resolveInside(root: string, relativePath: string): string | null {
  if (!relativePath || isAbsolute(relativePath) || relativePath.includes("\0")) return null;
  const rootReal = realpath(root);
  const full = resolve(rootReal, normalize(relativePath));
  if (!contains(rootReal, full)) return null;
  const real = realpath(full);
  return contains(rootReal, real) ? full : null;
}

const contains = (root: string, path: string) => path === root || path.startsWith(root + sep);

/** A path that does not exist yet resolves to itself: containment still decides. */
function realpath(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    return path;
  }
}
