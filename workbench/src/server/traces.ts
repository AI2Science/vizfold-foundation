import { join } from "node:path";

import type {
  ActivationStats,
  Activations,
  AttentionStats,
  RunFile,
  TensorEntry,
} from "../shared/types.ts";

type IndexEntry = { path?: unknown; dtype?: unknown; shape?: unknown };

async function readJson(path: string): Promise<unknown | null> {
  const file = Bun.file(path);
  if (!(await file.exists())) return null;
  try {
    return await file.json();
  } catch {
    // A trace written while the run is still going can be half a file; it will be whole next poll.
    return null;
  }
}

const asObject = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const number = (value: unknown): number => (typeof value === "number" && Number.isFinite(value) ? value : 0);

function tensorsFrom(
  group: TensorEntry["group"],
  index: unknown,
  sizes: Map<string, number>,
): TensorEntry[] {
  const entries = asObject(index);
  if (!entries) return [];
  return Object.entries(entries).flatMap(([key, raw]) => {
    const entry = asObject(raw) as IndexEntry | null;
    if (!entry || typeof entry.path !== "string") return [];
    return [
      {
        group,
        key,
        path: entry.path,
        dtype: typeof entry.dtype === "string" ? entry.dtype : "",
        shape: Array.isArray(entry.shape) ? entry.shape.map(number) : [],
        size: sizes.get(entry.path) ?? null,
      } satisfies TensorEntry,
    ];
  });
}

function attentionStatsFrom(summary: unknown): AttentionStats[] {
  const block = asObject(asObject(summary)?.attention);
  if (!block) return [];
  return Object.entries(block).flatMap(([key, raw]) => {
    const stats = asObject(raw);
    return stats
      ? [
          {
            key,
            mean: number(stats.mean),
            std: number(stats.std),
            entropy_proxy: number(stats.entropy_proxy),
            sparsity_proxy: number(stats.sparsity_proxy),
          },
        ]
      : [];
  });
}

function activationStatsFrom(summary: unknown): ActivationStats[] {
  const block = asObject(asObject(summary)?.activations);
  if (!block) return [];
  return Object.entries(block).flatMap(([key, raw]) => {
    const stats = asObject(raw);
    return stats
      ? [{ key, norm_mean: number(stats.norm_mean), mean: number(stats.mean), std: number(stats.std) }]
      : [];
  });
}

/** Layer keys sort as text ("layer_10" before "layer_9"), so order on the number inside them. */
const layerOf = (key: string) => Number(key.match(/\d+/)?.[0] ?? Number.MAX_SAFE_INTEGER);
const byLayer = <T extends { key: string }>(a: T, b: T) => layerOf(a.key) - layerOf(b.key) || a.key.localeCompare(b.key);

/**
 * What a run stored of the model's internals: ESMFold's `meta.json`, `trace/index.json` and
 * `trace/summary.json` where they exist, plus every dense array on disk that no index claims —
 * OpenFold's `.npz` attention dumps and its `_output_dict.pkl`.
 */
export async function readActivations(root: string, files: RunFile[]): Promise<Activations> {
  const sizes = new Map(files.map((file) => [file.path, file.size]));
  const [index, summary, metaJson] = await Promise.all([
    readJson(join(root, "trace", "index.json")),
    readJson(join(root, "trace", "summary.json")),
    readJson(join(root, "meta.json")),
  ]);
  const meta = asObject(metaJson);

  const tensors = [
    ...tensorsFrom("attention", asObject(index)?.attention, sizes),
    ...tensorsFrom("activations", asObject(index)?.activations, sizes),
  ].sort((a, b) => a.group.localeCompare(b.group) || byLayer(a, b));

  const indexed = new Set(tensors.map((tensor) => tensor.path));
  const arrays = files.filter((file) => file.kind === "tensor" && !indexed.has(file.path));

  return {
    tensors,
    attentionStats: attentionStatsFrom(summary).sort(byLayer),
    activationStats: activationStatsFrom(summary).sort(byLayer),
    meta,
    arrays,
  };
}
