import { Toast } from "@base-ui-components/react/toast";
import { useState } from "react";

import { startFold } from "../api.ts";
import { useNavigate } from "../router.tsx";
import { Banner, Empty, Picker, Search, Tick, Toggler } from "./ui.tsx";
import type { Environment, Protein } from "../../shared/types.ts";

export default function FoldForm({
  proteins,
  environment,
  onStarted,
}: {
  proteins: Protein[];
  environment: Environment;
  onStarted: () => void;
}) {
  const navigate = useNavigate();
  const toast = Toast.useToastManager();
  const backends = environment.backends;
  const [backend, setBackend] = useState(backends[0] ?? "");
  const [selected, setSelected] = useState<string[]>([]);
  const [attn, setAttn] = useState(true);
  const [query, setQuery] = useState("");
  const [folding, setFolding] = useState(false);

  // ESMFold folds one target per run; OpenFold takes the whole selection in one execution.
  const single = backend === "esmfold";

  const shown = proteins.filter(
    (protein) =>
      query === "" ||
      `${protein.id} ${protein.description}`.toLowerCase().includes(query.toLowerCase()),
  );

  if (!environment.cli.ok) {
    return (
      <Banner tone="critical" title="The vizfold CLI did not answer">
        <p className="note">{environment.cli.error}</p>
        <p className="note">
          Past runs still read from the database. Folding needs the binary at{" "}
          <code>{environment.cli.bin}</code>.
        </p>
      </Banner>
    );
  }

  if (backends.length === 0) {
    return (
      <Banner tone="warning" title="No backend is being served">
        Install one — <code>vizfold install openfold</code> or <code>vizfold install esmfold</code> —
        then restart <code>vizfold serve</code>.
      </Banner>
    );
  }

  if (proteins.length === 0) {
    return (
      <Empty title="No proteins to fold">
        <p className="note">
          They come from <code>examples/monomer</code> in the VizFold checkout — run{" "}
          <code>vizfold install repo</code>, then reload.
        </p>
      </Empty>
    );
  }

  const toggle = (id: string) =>
    setSelected((current) =>
      single
        ? [id]
        : current.includes(id)
          ? current.filter((one) => one !== id)
          : [...current, id],
    );

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setFolding(true);
    try {
      const runId = await startFold({ ids: selected, attn, backend });
      toast.add({
        title: `Run ${runId} started`,
        description: `${selected.join(", ")} on ${backend || "the default backend"}.`,
      });
      onStarted();
      // The toast provider sits above the router, so the toast outlives this form.
      navigate(`/runs/${runId}`);
    } catch (problem) {
      toast.add({
        type: "critical",
        title: "Could not start the fold",
        description: problem instanceof Error ? problem.message : String(problem),
        timeout: 12_000,
      });
      setFolding(false);
    }
  };

  const residues = proteins
    .filter((protein) => selected.includes(protein.id))
    .reduce((total, protein) => total + protein.residues, 0);

  return (
    <form className="stack" onSubmit={submit}>
      <div className="row">
        {backends.length > 1 ? (
          <Picker
            label="Model"
            value={backend}
            options={backends.map((slug) => ({ value: slug, label: slug }))}
            onChange={(next) => {
              setBackend(next);
              if (next === "esmfold") setSelected((current) => current.slice(0, 1));
            }}
            disabled={folding}
          />
        ) : null}
        <Search
          label="Search"
          value={query}
          onChange={setQuery}
          placeholder="id or description…"
          grow
        />
      </div>

      <p className="note">
        {single
          ? "ESMFold folds one target at a time, so pick one."
          : "Pick as many as you need — they fold as a single run, with the model loaded once."}{" "}
        <strong>alignments Y</strong> means the MSA is precomputed and the fold takes about a minute;
        N pays for the full search.
      </p>

      <div className="picker">
        {shown.map((protein) => {
          const on = selected.includes(protein.id);
          return (
            <label key={protein.id} className="protein" data-selected={on}>
              <Tick checked={on} onChange={() => toggle(protein.id)} disabled={folding} />
              <span className="protein-body">
                <span className="protein-id">{protein.id}</span>
                <span className="protein-meta">
                  <span className="tag">{protein.residues} residues</span>
                  <span className="tag" data-on={protein.alignments}>
                    alignments {protein.alignments ? "Y" : "N"}
                  </span>
                </span>
                {protein.description ? (
                  <span className="protein-desc">{protein.description}</span>
                ) : null}
              </span>
            </label>
          );
        })}
      </div>
      {shown.length === 0 ? <p className="note">Nothing matches “{query}”.</p> : null}

      <Toggler
        checked={attn}
        onChange={setAttn}
        disabled={folding}
        label="Dump attention maps"
        hint="One trace per layer and head, which the run page draws arc diagrams from."
      />

      <div className="row">
        <button className="button" data-variant="primary" type="submit" disabled={folding || selected.length === 0}>
          {folding ? <span className="spinner" /> : null}
          {folding
            ? "Starting…"
            : selected.length > 1
              ? `Fold ${selected.length} proteins`
              : "Fold"}
        </button>
        {selected.length > 0 ? (
          <span className="note">
            {selected.join(", ")} · {residues} residues
          </span>
        ) : null}
        {selected.length > 0 ? (
          <button
            type="button"
            className="button"
            data-variant="ghost"
            onClick={() => setSelected([])}
            disabled={folding}
          >
            Clear
          </button>
        ) : null}
      </div>
    </form>
  );
}
