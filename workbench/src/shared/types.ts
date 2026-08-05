/** The wire contract between the Bun server and the React client. Every field is read from the
 *  executor: its SQLite database, its run output directories, or the `vizfold` binary itself. */

export type CliHealth = {
  bin: string;
  ok: boolean;
  /** Why `list proteins` failed, when it did — shown instead of an empty picker. */
  error: string | null;
};

export type Environment = {
  /** Backend slugs this dashboard serves. Empty means `vizfold serve` found none installed. */
  backends: string[];
  /** False when VIZFOLD_BACKENDS is unset (`bun dev` by hand): nothing is filtered. */
  backendsConfigured: boolean;
  prefix: string;
  runsDir: string;
  database: { path: string; present: boolean };
  cli: CliHealth;
};

export type Protein = {
  id: string;
  residues: number;
  description: string;
  sequence: string;
  /** `alignments/<id>` is there to reuse; false pays for the full MSA search. */
  alignments: boolean;
};

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

export type Artifact = {
  id: number;
  format: string;
  storage_uri: string;
  type_slug: string;
  type_label: string;
  viewer_kind: string;
  display_mode: string;
};

export type FileKind = "structure" | "image" | "text" | "tensor" | "archive" | "other";

export type RunFile = {
  /** Path relative to the run's output directory — also its id and its download URL suffix. */
  path: string;
  name: string;
  size: number;
  modified: string;
  kind: FileKind;
};

/** One folded target inside a run. A batch run carries several. */
export type FoldTarget = {
  tag: string;
  sequence: string | null;
  /** The structure to look at (relaxed when both landed), or null while it is still folding. */
  structure: RunFile | null;
  /** Everything else that target wrote, unrelaxed structures included. */
  structures: RunFile[];
};

export type AttentionKind = "msa_row" | "triangle_start";

/** One attention text file the run wrote, as the picker sees it. */
export type AttentionSource = {
  /** Path relative to the run output directory; stable across polls. */
  path: string;
  /** Which target wrote it, when the backend nests attention per target. */
  tag: string | null;
  kind: AttentionKind;
  layer: number;
  /** Triangle-start attention is saved per query residue; "avg" is the averaged file. */
  residue: number | "avg" | null;
  /** The dense array beside it, when the run was asked for full attention. */
  dense: string | null;
};

export type AttentionHead = {
  head: number;
  /** [residue i, residue j, weight], strongest first. */
  edges: [number, number, number][];
  min: number;
  max: number;
};

export type AttentionMap = {
  source: AttentionSource;
  heads: AttentionHead[];
  /** One past the highest residue index the file mentions. */
  residues: number;
  /** The target's sequence from the run row, when it lines up with the residue span. */
  sequence: string | null;
  /** How many edges the file holds per head before top-k trimming. */
  edgesPerHead: number;
};

export type TensorEntry = {
  /** Which half of `trace/index.json` listed it. */
  group: "attention" | "activations";
  key: string;
  path: string;
  dtype: string;
  shape: number[];
  /** Bytes on disk, or null when the indexed file is gone. */
  size: number | null;
};

export type AttentionStats = {
  key: string;
  mean: number;
  std: number;
  entropy_proxy: number;
  sparsity_proxy: number;
};

export type ActivationStats = {
  key: string;
  norm_mean: number;
  mean: number;
  std: number;
};

/** What the run stored of the model's internals, as it stored it. */
export type Activations = {
  /** ESMFold's `trace/index.json`. */
  tensors: TensorEntry[];
  /** ESMFold's `trace/summary.json`, per layer. */
  attentionStats: AttentionStats[];
  activationStats: ActivationStats[];
  /** ESMFold's `meta.json` — model, device, dtype, layer and head counts. */
  meta: Record<string, unknown> | null;
  /** Dense arrays on disk that no index mentions: OpenFold's `.npz` and `_output_dict.pkl`. */
  arrays: RunFile[];
};

export type RunDetail = {
  run: RunRow;
  artifacts: Artifact[];
  /** The run's output directory, or null when nothing has been written yet. */
  root: string | null;
  targets: FoldTarget[];
  attention: AttentionSource[];
  activations: Activations;
  files: RunFile[];
  /** True when the run wrote more files than one listing walks; the file list is a prefix. */
  filesTruncated: boolean;
};

export type FoldRequest = {
  ids: string[];
  attn: boolean;
  backend?: string;
};
