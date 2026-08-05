import { describe, expect, test } from "bun:test";

import { inventory, parseAttention, sequenceFor } from "./attention.ts";
import type { RunFile } from "../shared/types.ts";

const file = (path: string): RunFile => ({
  path,
  name: path.split("/").at(-1)!,
  size: 1,
  modified: "2026-01-01T00:00:00.000Z",
  kind: "text",
});

// The format `save_attention_topk` writes in openfold/model/evoformer.py.
const DUMP = `Layer 47, Head 0
12 39 0.410000
8 14 0.320000
Layer 47, Head 1
4 5 0.400000
6 7 0.200000
`;

describe("parseAttention", () => {
  test("groups edges by head, strongest first", () => {
    const { heads, residues, edgesPerHead } = parseAttention(DUMP, null);
    expect(heads).toEqual([
      { head: 0, edges: [[12, 39, 0.41], [8, 14, 0.32]], min: 0.32, max: 0.41 },
      { head: 1, edges: [[4, 5, 0.4], [6, 7, 0.2]], min: 0.2, max: 0.4 },
    ]);
    expect(residues).toBe(40);
    expect(edgesPerHead).toBe(2);
  });

  test("top-k keeps the strongest edges of each head", () => {
    const { heads } = parseAttention(DUMP, 1);
    expect(heads.map((head) => head.edges)).toEqual([[[12, 39, 0.41]], [[4, 5, 0.4]]]);
  });

  test("an unordered file still comes out strongest first", () => {
    const { heads } = parseAttention("Layer 3, Head 0\n1 2 0.1\n3 4 0.9\n", 1);
    expect(heads[0]?.edges).toEqual([[3, 4, 0.9]]);
  });

  test("a half-written file parses to what is there", () => {
    const { heads, residues } = parseAttention("Layer 0, Head 0\n1 2 0.5\n3 4 0", 8);
    expect(heads[0]?.edges).toEqual([[1, 2, 0.5], [3, 4, 0]]);
    expect(residues).toBe(5);
  });

  test("edges before any header are dropped, not thrown", () => {
    expect(parseAttention("5 6 0.2\nLayer 1, Head 0\n1 1 0.3\n", null).heads).toEqual([
      { head: 0, edges: [[1, 1, 0.3]], min: 0.3, max: 0.3 },
    ]);
  });
});

describe("inventory", () => {
  test("reads target, kind, layer and query residue off the paths both backends write", () => {
    const found = inventory([
      file("attention/1UBQ_1/msa_row_attn_layer47.txt"),
      file("attention/1UBQ_1/msa_row_attn_layer47.npz"),
      file("attention/1UBQ_1/triangle_start_attn_layer47_residue_idx_18.txt"),
      file("attention/msa_row_attn_layer0.txt"),
      file("1UBQ_1_model_1_ptm_relaxed.pdb"),
    ]);

    expect(found).toEqual([
      {
        path: "attention/msa_row_attn_layer0.txt",
        tag: null,
        kind: "msa_row",
        layer: 0,
        residue: null,
        dense: null,
      },
      {
        path: "attention/1UBQ_1/msa_row_attn_layer47.txt",
        tag: "1UBQ_1",
        kind: "msa_row",
        layer: 47,
        residue: null,
        dense: "attention/1UBQ_1/msa_row_attn_layer47.npz",
      },
      {
        path: "attention/1UBQ_1/triangle_start_attn_layer47_residue_idx_18.txt",
        tag: "1UBQ_1",
        kind: "triangle_start",
        layer: 47,
        residue: 18,
        dense: null,
      },
    ]);
  });

  test("the averaged triangle-start file is kept as its own source", () => {
    const [source] = inventory([file("attention/triangle_start_attn_layer12_residue_idx_avg.txt")]);
    expect(source?.residue).toBe("avg");
  });
});

describe("sequenceFor", () => {
  test("a batch pairs its `+`-joined ids with its `:`-joined sequences", () => {
    expect(sequenceFor("1UBQ_1+6KWC_1", "MQIF:GSTI", "6KWC_1")).toBe("GSTI");
    expect(sequenceFor("1UBQ_1+6KWC_1", "MQIF:GSTI", "1UBQ_1")).toBe("MQIF");
  });

  test("a lone target owns the run's sequence, tagged or not", () => {
    expect(sequenceFor("1UBQ_1", "MQIF", null)).toBe("MQIF");
    expect(sequenceFor("1UBQ_1", "MQIF", "1UBQ_1")).toBe("MQIF");
  });

  test("an untagged file in a batch cannot be attributed", () => {
    expect(sequenceFor("1UBQ_1+6KWC_1", "MQIF:GSTI", null)).toBeNull();
    expect(sequenceFor("1UBQ_1+6KWC_1", "MQIF:GSTI", "2OMF_1")).toBeNull();
  });
});
