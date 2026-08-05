import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { kindOf, listFiles, resolveInside, runRoot } from "./runfiles.ts";
import type { Artifact } from "../shared/types.ts";

const artifact = (type_slug: string, storage_uri: string): Artifact => ({
  id: 1,
  format: "directory",
  storage_uri,
  type_slug,
  type_label: type_slug,
  viewer_kind: "directory_link",
  display_mode: "download",
});

function sandbox(): string {
  const root = mkdtempSync(join(tmpdir(), "vizfold-workbench-"));
  mkdirSync(join(root, "run", "attention", "1UBQ_1"), { recursive: true });
  writeFileSync(join(root, "run", "1UBQ_1_model_1_ptm_relaxed.pdb"), "ATOM\n");
  writeFileSync(join(root, "run", "attention", "1UBQ_1", "msa_row_attn_layer0.txt"), "Layer 0, Head 0\n");
  writeFileSync(join(root, "outside.pdb"), "ATOM\n");
  return root;
}

describe("listFiles", () => {
  test("walks into subdirectories and paths everything off the run root", () => {
    const root = join(sandbox(), "run");
    const { files, truncated } = listFiles(root);
    expect(files.map((file) => file.path).sort()).toEqual([
      "1UBQ_1_model_1_ptm_relaxed.pdb",
      "attention/1UBQ_1/msa_row_attn_layer0.txt",
    ]);
    expect(truncated).toBe(false);
  });

  test("a run larger than the limit is reported as truncated, never silently cut", () => {
    const root = join(sandbox(), "run");
    const { files, truncated } = listFiles(root, 1);
    expect(files).toHaveLength(1);
    expect(truncated).toBe(true);
  });
});

describe("resolveInside", () => {
  test("resolves a path within the run", () => {
    const root = join(sandbox(), "run");
    expect(resolveInside(root, "attention/1UBQ_1/msa_row_attn_layer0.txt")).toBe(
      join(root, "attention/1UBQ_1/msa_row_attn_layer0.txt"),
    );
  });

  test("refuses to climb out of the run", () => {
    const root = join(sandbox(), "run");
    expect(resolveInside(root, "../outside.pdb")).toBeNull();
    expect(resolveInside(root, "/etc/passwd")).toBeNull();
    expect(resolveInside(root, "")).toBeNull();
  });

  test("refuses a symlink that points out of the run", () => {
    const base = sandbox();
    const root = join(base, "run");
    symlinkSync(join(base, "outside.pdb"), join(root, "escape.pdb"));
    expect(resolveInside(root, "escape.pdb")).toBeNull();
  });
});

describe("runRoot", () => {
  test("prefers the registered run output directory", () => {
    expect(runRoot([artifact("run_output_directory", "/prefix/runs/7")], 7)).toBe("/prefix/runs/7");
  });

  test("falls back to the run directory inside another artifact's path", () => {
    expect(runRoot([artifact("attention_output_directory", "/prefix/runs/7/attention")], 7)).toBe(
      "/prefix/runs/7",
    );
  });
});

test("kindOf classifies what the backends write", () => {
  expect(kindOf("predicted.pdb")).toBe("structure");
  expect(kindOf("layer_000.pt")).toBe("tensor");
  expect(kindOf("msa_row_attn_layer0.txt")).toBe("text");
  expect(kindOf("panel.png")).toBe("image");
  expect(kindOf("run.sock")).toBe("other");
});
