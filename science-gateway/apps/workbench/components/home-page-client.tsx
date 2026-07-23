"use client";

import { useState, type SubmitEventHandler } from "react";

export type ModelOption = {
  slug: string;
  label: string;
  summary: string;
  capabilities: string[];
};

export type ExecutionTargetOption = {
  slug: string;
  label: string;
  summary: string;
  kind: string;
  status: "available" | "planned" | "disabled";
};

type MockRunResult = {
  jobName: string;
  modelLabel: string;
  executionTargetLabel: string;
  sequenceLength: number;
  status: "queued" | "mock-complete";
  note: string;
};

type HomePageClientProps = {
  modelOptions: ModelOption[];
  executionTargets: ExecutionTargetOption[];
};

const DEFAULT_SEQUENCE = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDL";

export function HomePageClient({
  modelOptions,
  executionTargets,
}: HomePageClientProps)
 {
  const [selectedModel, setSelectedModel] = useState(modelOptions[0]?.slug ?? "");
  const [selectedExecutionTargetSlug, setSelectedExecutionTargetSlug] = useState(
  executionTargets[0]?.slug ?? ""
  );
  const selectedExecutionTarget = executionTargets.find(
    (target) => target.slug === selectedExecutionTargetSlug
  );
  const [sequence, setSequence] = useState(DEFAULT_SEQUENCE);
  const [jobName, setJobName] = useState("");
  const [result, setResult] = useState<MockRunResult | null>(null);

  const modelDetails = modelOptions.find((modelOption) => modelOption.slug === selectedModel);
  const executionTargetDetails = executionTargets.find(
  (target) => target.slug === selectedExecutionTargetSlug
  );

  const handleSubmit: SubmitEventHandler<HTMLFormElement> = (event) => {
    event.preventDefault();

    if (!modelDetails || !executionTargetDetails) {
      return;
    }

    const normalizedSequence = sequence.replace(/\s+/g, "").toUpperCase();

    setResult({
      jobName: jobName.trim() || "Untitled VizFold run",
      modelLabel: modelDetails.label,
      executionTargetLabel: executionTargetDetails.label,
      sequenceLength: normalizedSequence.length,
      status: "mock-complete",
      note: "Mock structure result generated locally for the UI prototype. API execution will plug in here later.",
    });
  };

  if (!modelDetails || !executionTargetDetails) {
    return null;
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">VizFold v2</p>
          <h1 className="brand-title">VizFold</h1>
          <p className="subtitle">
            A prototype multi-model protein structure visualization workbench for comparing
            folding backends and organizing future runs. This interface is work-in-progress
            and currently uses mock frontend data only.
          </p>
        </div>
      </section>

      <div className="workspace-grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Run prototype job</h2>
            <p>Work-in-progress UI only for now. No persistence or executor connection yet.</p>
          </div>

          <form className="run-form" onSubmit={handleSubmit}>
            <label className="field">
              <span>Model</span>
              <select
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
              >
                {modelOptions.map((modelOption) => (
                  <option key={modelOption.slug} value={modelOption.slug}>
                    {modelOption.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Execution target</span>
              <select
                value={selectedExecutionTargetSlug}
                onChange={(event) =>
                  setSelectedExecutionTargetSlug(event.target.value)
                }
              >
                {executionTargets.map((target) => (
                  <option
                    key={target.slug}
                    value={target.slug}
                    disabled={target.status !== "available"}
                  >
                    {target.label}
                    {target.status !== "available" ? " — planned" : ""}
                  </option>
                ))}
              </select>
              <p className="field-note">{executionTargetDetails.summary}</p>
            </label>

            <label className="field">
              <span>Job name</span>
              <input
                type="text"
                placeholder="Optional run label"
                value={jobName}
                onChange={(event) => setJobName(event.target.value)}
              />
            </label>

            <label className="field">
              <span>Protein sequence</span>
              <textarea
                rows={8}
                value={sequence}
                onChange={(event) => setSequence(event.target.value)}
                placeholder="Paste an amino acid sequence"
              />
            </label>

            <button className="primary-button" type="submit">
              Run
            </button>
          </form>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Model capabilities</h2>
            <p>{modelDetails.summary}</p>
          </div>

          <ul className="capability-list">
            {modelDetails.capabilities.map((capability) => (
              <li key={capability}>{capability}</li>
            ))}
          </ul>
        </section>

        <section className="panel result-panel">
          <div className="panel-header">
            <h2>Output</h2>
            <p>Mock result panel for early architecture and layout testing.</p>
          </div>

          {result ? (
            <div className="result-card">
              <div className="result-row">
                <span>Job</span>
                <strong>{result.jobName}</strong>
              </div>
              <div className="result-row">
                <span>Model</span>
                <strong>{result.modelLabel}</strong>
              </div>
              <div className="result-row">
                <span>Execution target</span>
                <strong>{result.executionTargetLabel}</strong>
              </div>
              <div className="result-row">
                <span>Sequence length</span>
                <strong>{result.sequenceLength} aa</strong>
              </div>
              <div className="result-row">
                <span>Status</span>
                <strong>{result.status}</strong>
              </div>
              <p className="result-note">{result.note}</p>
            </div>
          ) : (
            <div className="empty-state">
              <p>No run submitted yet.</p>
              <p>Your first prototype result will appear here after pressing Run.</p>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
