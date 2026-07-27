"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import type { Protein } from "@/lib/vizfold";

export default function FoldCard({
  proteins,
  backends,
}: {
  proteins: Protein[];
  backends: string[];
}) {
  const router = useRouter();
  const [selected, setSelected] = useState<string[]>([]);
  const [backend, setBackend] = useState(backends[0] ?? "");
  const [attn, setAttn] = useState(true);
  const [folding, setFolding] = useState(false);
  const [error, setError] = useState("");

  // ESMFold folds one target per run; OpenFold takes the whole selection in one execution.
  const single = backend === "esmfold";

  if (proteins.length === 0) {
    return (
      <section className="panel">
        <div className="panel-header">
          <h2>Fold proteins</h2>
        </div>
        <p className="field-note">
          No proteins found. They come from <code>examples/monomer</code> in the VizFold checkout —
          run <code>vizfold install repo</code>, then reload.
        </p>
      </section>
    );
  }

  function toggle(id: string) {
    setSelected((current) =>
      single
        ? [id]
        : current.includes(id)
          ? current.filter((one) => one !== id)
          : [...current, id],
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
        body: JSON.stringify({ ids: selected, attn, backend }),
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
        <h2>Fold proteins</h2>
        <p>
          {single
            ? "ESMFold folds one at a time, so pick one. Alignments Y"
            : "Pick one or more — they fold as a single run, with the model loaded once. Alignments Y"}
          means they are precomputed and the fold takes about a minute; N pays for the full MSA
          search.
        </p>
      </div>

      <form className="run-form" onSubmit={fold}>
        <div className="field">
          <span>Proteins</span>
          <ul className="picker">
            {proteins.map((protein) => (
              <li key={protein.id}>
                <label>
                  <input
                    type="checkbox"
                    checked={selected.includes(protein.id)}
                    onChange={() => toggle(protein.id)}
                    disabled={folding}
                  />
                  <strong>{protein.id}</strong>
                  <span className="tag">{protein.residues} residues</span>
                  <span className={protein.alignments ? "tag tag-on" : "tag"}>
                    alignments {protein.alignments ? "Y" : "N"}
                  </span>
                  <span className="note">{protein.description}</span>
                </label>
              </li>
            ))}
          </ul>
        </div>

        {backends.length > 1 ? (
          <label className="field">
            <span>Model</span>
            <select
              value={backend}
              onChange={(event) => {
                setBackend(event.target.value);
                if (event.target.value === "esmfold") setSelected((current) => current.slice(0, 1));
              }}
              disabled={folding}
            >
              {backends.map((slug) => (
                <option key={slug} value={slug}>
                  {slug}
                </option>
              ))}
            </select>
          </label>
        ) : null}

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

        <button className="primary-button" type="submit" disabled={folding || selected.length === 0}>
          {folding ? "Starting…" : selected.length > 1 ? `Fold ${selected.length} proteins` : "Fold"}
        </button>
      </form>
    </section>
  );
}
