"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import type { Example } from "@/lib/vizfold";

export default function FoldCard({ examples }: { examples: Example[] }) {
  const router = useRouter();
  const [inputId, setInputId] = useState(examples[0]?.id ?? "");
  const [attn, setAttn] = useState(true);
  const [folding, setFolding] = useState(false);
  const [error, setError] = useState("");

  if (examples.length === 0) {
    return (
      <section className="panel">
        <div className="panel-header">
          <h2>Fold a protein</h2>
        </div>
        <p className="field-note">
          No examples found. They come from <code>examples/monomer</code> in the VizFold checkout —
          run <code>vizfold install openfold</code>, then reload.
        </p>
      </section>
    );
  }

  async function fold(event: React.FormEvent) {
    event.preventDefault();
    setFolding(true);
    setError("");
    try {
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ inputId, attn }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? "Could not start the fold.");
      router.push(`/runs/${body.runId}`);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem));
      setFolding(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Fold a protein</h2>
        <p>Bundled examples fold in about a minute — their alignments are precomputed.</p>
      </div>

      <form className="run-form" onSubmit={fold}>
        <label className="field">
          <span>Protein</span>
          <select
            value={inputId}
            onChange={(event) => setInputId(event.target.value)}
            disabled={folding}
          >
            {examples.map((example) => (
              <option key={example.id} value={example.id}>
                {example.id}
                {example.description ? ` — ${example.description}` : ""} ({example.residues}{" "}
                residues)
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="check">
            <input
              type="checkbox"
              checked={attn}
              onChange={(event) => setAttn(event.target.checked)}
              disabled={folding}
            />
            Dump attention maps
          </span>
          <p className="field-note">
            Writes one trace per layer and head, which the run page renders alongside the structure.
          </p>
        </label>

        {error ? <p className="field-note">{error}</p> : null}

        <button className="primary-button" type="submit" disabled={folding}>
          {folding ? "Starting…" : "Fold"}
        </button>
      </form>
    </section>
  );
}
