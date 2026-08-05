import { dirname, sep } from "node:path";

import type { AttentionHead, AttentionKind, AttentionMap, AttentionSource, RunFile } from "../shared/types.ts";

/** What the backends write, and what `examples/visualize_attention_data.py` reads:
 *  `msa_row_attn_layer<L>.txt` and `triangle_start_attn_layer<L>_residue_idx_<R|avg>.txt`,
 *  saved by `save_attention_topk` (OpenFold's evoformer) and by ESMFold's trace adapter. */
const NAME = /^(msa_row|triangle_start)_attn_layer(\d+)(?:_residue_idx_(\d+|avg))?\.(txt|npz)$/;

const KIND_OF: Record<string, AttentionKind> = {
  msa_row: "msa_row",
  triangle_start: "triangle_start",
};

/** OpenFold nests attention per target (`attention/<tag>/…`); ESMFold writes it flat. Whatever
 *  sits between the `attention` directory and the file is the target it belongs to. */
function tagOf(path: string): string | null {
  const parts = dirname(path).split(sep).filter((part) => part && part !== ".");
  const at = parts.lastIndexOf("attention");
  if (at < 0) return parts.at(-1) ?? null;
  const nested = parts.slice(at + 1);
  return nested.length ? nested.join("/") : null;
}

/**
 * What a path says about itself, or null when it is not an attention dump. The file name carries
 * the whole description, so a request for one names its own source and no listing is needed to
 * resolve it.
 */
export function describeSource(path: string): AttentionSource | null {
  const match = NAME.exec(path.split("/").at(-1) ?? "");
  if (!match || match[4] !== "txt") return null;
  const [, kind, layer, residue] = match;
  return {
    path,
    tag: tagOf(path),
    kind: KIND_OF[kind!]!,
    layer: Number(layer),
    residue: residue === undefined ? null : residue === "avg" ? "avg" : Number(residue),
    dense: null,
  };
}

/** The attention text files among a run's files, each paired with the dense array beside it. */
export function inventory(files: RunFile[]): AttentionSource[] {
  const dense = new Set(
    files.filter((file) => NAME.exec(file.name)?.[4] === "npz").map((file) => file.path),
  );
  const sources = files.flatMap((file) => {
    const source = describeSource(file.path);
    if (!source) return [];
    const beside = file.path.replace(/\.txt$/, ".npz");
    return [{ ...source, dense: dense.has(beside) ? beside : null }];
  });
  return sources.sort(
    (a, b) =>
      (a.tag ?? "").localeCompare(b.tag ?? "") ||
      a.kind.localeCompare(b.kind) ||
      a.layer - b.layer ||
      Number(a.residue ?? -1) - Number(b.residue ?? -1),
  );
}

/**
 * Parse one attention text file into per-head edge lists.
 *
 * The format the backends write:
 *
 *     Layer 47, Head 0
 *     12 39 0.410000
 *     8 14 0.320000
 *     Layer 47, Head 1
 *     …
 *
 * Edges come out strongest first, so `topK` is a slice rather than a second sort.
 */
export function parseAttention(text: string, topK: number | null): {
  heads: AttentionHead[];
  residues: number;
  edgesPerHead: number;
} {
  const heads = new Map<number, [number, number, number][]>();
  let current: [number, number, number][] | null = null;
  let residues = 0;
  let widest = 0;

  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    if (line.toLowerCase().startsWith("layer")) {
      // "Layer 47, Head 0" — the head index is the last field, whatever separates them.
      const head = Number(line.replace(/,/g, " ").trim().split(/\s+/).at(-1));
      if (!Number.isInteger(head)) continue;
      current = [];
      heads.set(head, current);
      continue;
    }
    if (!current) continue;
    const [i, j, weight] = line.split(/\s+/);
    const from = Number(i);
    const to = Number(j);
    const value = Number(weight);
    if (!Number.isFinite(from) || !Number.isFinite(to) || !Number.isFinite(value)) continue;
    current.push([from, to, value]);
    residues = Math.max(residues, from + 1, to + 1);
    widest = Math.max(widest, current.length);
  }

  return {
    residues,
    edgesPerHead: widest,
    heads: [...heads.entries()]
      .sort(([a], [b]) => a - b)
      .map(([head, edges]) => {
        // Written strongest-first, but a hand-made file need not be; sorting is cheap next to IO.
        const ordered = [...edges].sort((a, b) => b[2] - a[2]);
        const kept = topK === null ? ordered : ordered.slice(0, topK);
        const weights = kept.map(([, , weight]) => weight);
        return {
          head,
          edges: kept,
          min: weights.length ? Math.min(...weights) : 0,
          max: weights.length ? Math.max(...weights) : 0,
        } satisfies AttentionHead;
      }),
  };
}

/** The sequence a target was folded from, when the run row carries one per tag. A batch joins its
 *  targets with ":", in the same order as the "+"-joined ids. */
export function sequenceFor(inputId: string, inputSequence: string, tag: string | null): string | null {
  const tags = inputId.split("+");
  const sequences = inputSequence.split(":");
  if (tags.length !== sequences.length) return sequences.length === 1 ? (sequences[0] ?? null) : null;
  if (tag === null) return sequences.length === 1 ? (sequences[0] ?? null) : null;
  const at = tags.indexOf(tag);
  return at < 0 ? null : (sequences[at] ?? null);
}

/** Read and parse one source. The residue axis is labelled from the run's own sequence when the
 *  two agree on length — a mismatch means the file is not this target's, so the axis stays numeric. */
export async function readAttention(
  absolutePath: string,
  source: AttentionSource,
  sequence: string | null,
  topK: number | null,
): Promise<AttentionMap> {
  const text = await Bun.file(absolutePath).text();
  const { heads, residues, edgesPerHead } = parseAttention(text, topK);
  return {
    source,
    heads,
    residues,
    edgesPerHead,
    sequence: sequence && sequence.length >= residues ? sequence : null,
  };
}
