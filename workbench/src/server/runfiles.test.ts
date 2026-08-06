import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { kindOf, listFiles, resolveInside } from "./runfiles.ts";

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
  test("resolves a path within the run to the file it names", () => {
    const root = join(sandbox(), "run");
    const wanted = "attention/1UBQ_1/msa_row_attn_layer0.txt";
    const resolved = resolveInside(root, wanted);
    // Containment is decided on the run root's real path, so a symlinked root resolves through it:
    // on macOS the temp directory is exactly that, `/var/…` standing for `/private/var/…`.
    expect(resolved).toBe(realpathSync(join(root, wanted)));
    expect(readFileSync(resolved!, "utf8")).toBe("Layer 0, Head 0\n");
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


test("kindOf classifies what the backends write", () => {
  expect(kindOf("predicted.pdb")).toBe("structure");
  expect(kindOf("layer_000.pt")).toBe("tensor");
  expect(kindOf("msa_row_attn_layer0.txt")).toBe("text");
  expect(kindOf("panel.png")).toBe("image");
  expect(kindOf("run.sock")).toBe("other");
});
