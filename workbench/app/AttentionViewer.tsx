"use client";

import { useState } from "react";
import { Segmented } from "@/app/StructureViewer";

// The names the arc pipeline writes (examples/visualize_attention_arc_diagram_demo_utils.py):
// `msa_row_head_<h>_layer_<l>_<protein>_arc.png`, `tri_start_res_<r>_head_<h>_layer_<l>_...`.
type Kind = "msa_row" | "tri_start";

const KINDS: [Kind, string][] = [
  ["msa_row", "MSA row"],
  ["tri_start", "Triangle start"],
];

const field = (name: string, key: string) =>
  Number(name.match(new RegExp(`${key}_(\\d+)`))?.[1] ?? NaN);

const kindOf = (name: string): Kind => (name.includes("msa_row") ? "msa_row" : "tri_start");

/** Whichever of layer/residue/head the name carries; the protein is already on the page. */
const label = (name: string) =>
  ([["layer", "layer"], ["residue", "res"], ["head", "head"]] as const)
    .filter(([, key]) => !Number.isNaN(field(name, key)))
    .map(([word, key]) => `${word} ${field(name, key)}`)
    .join(" · ");

/** Heads at a layer are read side by side: every one stays on screen as a thumbnail and the
 *  picked one gets the full-width view. */
export default function AttentionViewer({ images }: { images: { name: string; url: string }[] }) {
  // The picked name carries the attention type too, so there is no second selection to keep in
  // sync — and a name that is no longer on offer falls back instead of showing an empty view.
  const [picked, setPicked] = useState("");

  const sorted = [...images].sort(
    (a, b) =>
      field(a.name, "layer") - field(b.name, "layer") ||
      field(a.name, "head") - field(b.name, "head"),
  );
  const first = (kind: Kind) => sorted.find((image) => kindOf(image.name) === kind);
  const kinds = KINDS.filter(([kind]) => first(kind));
  const kind = sorted.some((image) => image.name === picked)
    ? kindOf(picked)
    : (kinds[0]?.[0] ?? "msa_row");

  const shown = sorted.filter((image) => kindOf(image.name) === kind);
  const current = shown.find((image) => image.name === picked) ?? shown[0];
  if (!current) return null;

  return (
    <>
      {kinds.length > 1 ? (
        <Segmented
          label="Attention"
          value={kind}
          onChange={(next) => setPicked(first(next)?.name ?? "")}
          options={kinds}
        />
      ) : null}
      <p className="field-note">{label(current.name)}</p>
      {/* Full resolution on click: the arcs are dense enough that the fitted view crowds them. */}
      <a href={current.url} target="_blank" rel="noreferrer">
        <img src={current.url} alt={current.name} className="arc-main" />
      </a>
      <div className="arc-gallery">
        {shown.map((image) => (
          <button
            key={image.name}
            type="button"
            className={image.name === current.name ? "arc-thumb arc-on" : "arc-thumb"}
            onClick={() => setPicked(image.name)}
          >
            <img src={image.url} alt={image.name} />
            <span>head {field(image.name, "head")}</span>
          </button>
        ))}
      </div>
    </>
  );
}
