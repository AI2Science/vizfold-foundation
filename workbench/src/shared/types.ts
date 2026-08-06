/** The wire contract between the Bun server and the React client. Every field is read from the
 *  executor: its SQLite database, its run output directories, or the `vizfold` binary itself. */

export type Environment = {
  /** Backend slugs this dashboard serves. Empty means `vizfold serve` found none installed. */
  backends: string[];
  prefix: string;
  database: { path: string; present: boolean };
  cli: {
    bin: string;
    ok: boolean;
    /** Why `list proteins` failed, when it did — shown instead of an empty picker. */
    error: string | null;
  };
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

/** One produced result, and what its kind says about showing it. The executor classifies every
 *  file a run wrote; the dashboard reads the classification rather than guessing from a name. */
export type Artifact = {
  id: number;
  /** Path on disk, and the run-relative path the file routes resolve. */
  storage_uri: string;
  path: string;
  format: string;
  size: number | null;
  type_slug: string;
  type_label: string;
  /** `embedded` in a viewer, `download` as a link, `internal` for something read but not shown. */
  display_mode: string;
  /** Which viewer the kind asks for: structure_viewer, arc_diagram, stat_panel, key_values … */
  viewer_kind: string;
  /** What the executor read off the file name: target, layer, attention type, stage, relaxed … */
  metadata: Record<string, unknown>;
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
  /** Every produced result the executor classified. The tabs are built from the kinds present. */
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
