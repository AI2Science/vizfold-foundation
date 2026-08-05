import { Tabs } from "@base-ui-components/react/tabs";
import { useState } from "react";
import type { ReactNode } from "react";

import { fetchRun, fileUrl, isActive, useAsync } from "../api.ts";
import ActivationsPanel from "../components/ActivationsPanel.tsx";
import AttentionPanel from "../components/AttentionPanel.tsx";
import FilesPanel from "../components/FilesPanel.tsx";
import StructureViewer from "../components/StructureViewer.tsx";
import { Banner, Empty, Field, Picker, Status, bytes, when } from "../components/ui.tsx";
import { Link } from "../router.tsx";
import type { RunDetail } from "../../shared/types.ts";

function Structures({ detail }: { detail: RunDetail }) {
  const withStructure = detail.targets.filter((target) => target.structure);
  const [tag, setTag] = useState("");
  const target = withStructure.find((one) => one.tag === tag) ?? withStructure[0];

  if (!target?.structure) return null;

  return (
    <div>
      {withStructure.length > 1 ? (
        <div className="control-row">
          <Picker
            label="Target"
            value={target.tag}
            options={withStructure.map((one) => ({ value: one.tag, label: one.tag }))}
            onChange={setTag}
          />
          <Field label="Residues">{target.sequence?.length ?? "—"}</Field>
        </div>
      ) : null}

      <div className="panel-body stack">
        <div className="row">
          <strong>{target.tag}</strong>
          <span className="path">{target.structure.path}</span>
          <div className="spacer" />
          <a className="button" href={fileUrl(detail.run.id, target.structure.path)} download>
            Download
          </a>
        </div>

        <StructureViewer
          key={target.structure.path}
          url={fileUrl(detail.run.id, target.structure.path)}
          name={target.structure.name}
        />

        {target.structures.length > 1 ? (
          <p className="note">
            Also written:{" "}
            {target.structures
              .filter((file) => file.path !== target.structure?.path)
              .map((file, index) => (
                <span key={file.path}>
                  {index > 0 ? ", " : ""}
                  <a href={fileUrl(detail.run.id, file.path)} download>
                    {file.name}
                  </a>{" "}
                  <span className="muted">({bytes(file.size)})</span>
                </span>
              ))}
          </p>
        ) : null}
      </div>
    </div>
  );
}

type Tab = { value: string; label: string; count?: number; render: () => ReactNode };

export default function RunPage({ id }: { id: number }) {
  const { data: detail, error, loading } = useAsync(
    (signal) => fetchRun(id, signal),
    [id],
    // The executor writes the run row and its outputs as it goes; a run in flight is re-read.
    3000,
  );
  const [tab, setTab] = useState<string | null>(null);

  if (loading && !detail) {
    return (
      <section className="panel">
        <div className="panel-body">
          <div className="skeleton" style={{ height: 160 }} />
        </div>
      </section>
    );
  }

  if (error || !detail) {
    return (
      <Banner tone="critical" title={`Run ${id}`}>
        {error ?? "Not found."}
      </Banner>
    );
  }

  const { run } = detail;
  const running = isActive(run.status);
  const structures = detail.targets.filter((target) => target.structure);
  const hasActivations =
    detail.activations.tensors.length > 0 ||
    detail.activations.arrays.length > 0 ||
    detail.activations.activationStats.length > 0 ||
    detail.activations.attentionStats.length > 0 ||
    detail.activations.meta !== null;

  const tabs: Tab[] = [];
  if (structures.length) {
    tabs.push({
      value: "structure",
      label: "Structure",
      count: structures.length,
      render: () => <Structures detail={detail} />,
    });
  }
  if (detail.attention.length) {
    tabs.push({
      value: "attention",
      label: "Attention",
      count: detail.attention.length,
      render: () => <AttentionPanel detail={detail} />,
    });
  }
  if (hasActivations) {
    tabs.push({
      value: "activations",
      label: "Activations",
      count: detail.activations.tensors.length + detail.activations.arrays.length || undefined,
      render: () => <ActivationsPanel detail={detail} />,
    });
  }
  if (detail.files.length) {
    tabs.push({
      value: "files",
      label: "Files",
      count: detail.files.length,
      render: () => <FilesPanel detail={detail} />,
    });
  }

  const selected = tabs.find((one) => one.value === tab)?.value ?? tabs[0]?.value ?? "";

  return (
    <>
      <section className="panel hero">
        <div>
          <p className="note">
            <Link href="/runs">← All runs</Link>
          </p>
          <h1 style={{ marginTop: 6 }}>Run {run.id}</h1>
          <p className="note" style={{ marginTop: 6 }}>
            {run.input_id.split("+").join(", ")} · {run.model_slug} on {run.target_slug}
          </p>
        </div>
        <dl className="kv">
          <div>
            <dt>Status</dt>
            <dd>
              <Status status={run.status} />
            </dd>
          </div>
          <div>
            <dt>Submitted</dt>
            <dd>{when(run.submitted_at)}</dd>
          </div>
          <div>
            <dt>Started</dt>
            <dd>{when(run.started_at)}</dd>
          </div>
          <div>
            <dt>Completed</dt>
            <dd>{when(run.completed_at)}</dd>
          </div>
        </dl>
      </section>

      {run.error_message ? (
        <Banner tone="critical" title="The executor reported an error">
          {run.error_message}
        </Banner>
      ) : null}

      {tabs.length === 0 ? (
        <section className="panel">
          <Empty title={running ? "Folding — nothing written yet" : "This run wrote no output"}>
            <p className="note">
              {running
                ? "The page re-reads the run every few seconds; structures, attention and activations appear here as the executor writes them."
                : detail.root
                  ? `Nothing was found under ${detail.root}.`
                  : "The executor has not created this run's output directory."}
            </p>
          </Empty>
        </section>
      ) : (
        <section className="panel">
          <Tabs.Root value={selected} onValueChange={(next) => setTab(String(next))}>
            <Tabs.List className="tabs-list">
              {tabs.map((one) => (
                <Tabs.Tab key={one.value} value={one.value} className="tab">
                  {one.label}
                  {one.count === undefined ? null : <span className="tab-badge">{one.count}</span>}
                </Tabs.Tab>
              ))}
              <Tabs.Indicator className="tabs-indicator" />
            </Tabs.List>
            {tabs.map((one) => (
              <Tabs.Panel key={one.value} value={one.value} className="tab-panel flush">
                {one.render()}
              </Tabs.Panel>
            ))}
          </Tabs.Root>
        </section>
      )}

      {running ? (
        <p className="note row">
          <span className="spinner" /> Re-reading the run while it is in flight.
        </p>
      ) : null}
    </>
  );
}
