import { useState } from "react";

import Bars from "./Bars.tsx";
import FileTable from "./FileTable.tsx";
import { Picker, Segmented, bytes } from "./ui.tsx";
import { fileUrl } from "../api.ts";
import type { RunDetail } from "../../shared/types.ts";

type Measure = "norm_mean" | "mean" | "std";
type AttentionMeasure = "entropy_proxy" | "mean" | "std" | "sparsity_proxy";

const MEASURE_LABEL: Record<Measure, string> = {
  norm_mean: "RMS norm",
  mean: "mean",
  std: "std",
};

const ATTENTION_LABEL: Record<AttentionMeasure, string> = {
  entropy_proxy: "entropy",
  mean: "mean",
  std: "std",
  sparsity_proxy: "sparsity",
};

const META_ROWS: [string, string][] = [
  ["model_name", "Model"],
  ["device", "Device"],
  ["dtype", "Compute dtype"],
  ["tensor_format", "Stored as"],
  ["sequence_length", "Residues"],
  ["layer_count", "Layers"],
  ["head_count", "Heads"],
  ["trace_mode", "Traced"],
  ["top_k", "Edges per head"],
  ["date_time", "Written"],
];

/** A `{value: label}` record read as the option list a Segmented takes. */
const optionsOf = <T extends string>(labels: Record<T, string>) =>
  (Object.keys(labels) as T[]).map((value) => ({ value, label: labels[value] }));

export default function ActivationsPanel({ detail }: { detail: RunDetail }) {
  const { activations, run } = detail;
  const [measure, setMeasure] = useState<Measure>("norm_mean");
  const [attentionMeasure, setAttentionMeasure] = useState<AttentionMeasure>("entropy_proxy");
  const [group, setGroup] = useState<"activations" | "attention">("activations");

  const tensors =
    group === "activations"
      ? activations.tensors.filter((tensor) => tensor.group === "activations")
      : activations.tensors.filter((tensor) => tensor.group === "attention");

  return (
    <div className="panel-body stack">
      {activations.meta ? (
        <dl className="kv">
          {META_ROWS.filter(([key]) => activations.meta?.[key] !== undefined).map(([key, label]) => (
            <div key={key}>
              <dt>{label}</dt>
              <dd>{String(activations.meta?.[key])}</dd>
            </div>
          ))}
        </dl>
      ) : null}

      {activations.activationStats.length > 0 ? (
        <section>
          <div className="row" style={{ marginBottom: 10 }}>
            <h3>Activation magnitude by layer</h3>
            <div className="spacer" />
            <Segmented
              value={measure}
              onChange={setMeasure}
              options={optionsOf(MEASURE_LABEL)}
            />
          </div>
          <Bars
            unitLabel={MEASURE_LABEL[measure]}
            bars={activations.activationStats.map((stat) => ({
              label: stat.key,
              value: stat[measure],
              detail: `RMS ${stat.norm_mean.toPrecision(3)} · mean ${stat.mean.toPrecision(3)} · std ${stat.std.toPrecision(3)}`,
            }))}
          />
          <p className="note muted">
            From the run's own <code>trace/summary.json</code>, over the tensors it stored.
          </p>
        </section>
      ) : null}

      {activations.attentionStats.length > 0 ? (
        <section>
          <div className="row" style={{ marginBottom: 10 }}>
            <h3>Attention statistics by layer</h3>
            <div className="spacer" />
            <Segmented
              value={attentionMeasure}
              onChange={setAttentionMeasure}
              options={optionsOf(ATTENTION_LABEL)}
            />
          </div>
          <Bars
            unitLabel={ATTENTION_LABEL[attentionMeasure]}
            bars={activations.attentionStats.map((stat) => ({
              label: stat.key,
              value: stat[attentionMeasure],
              detail: `entropy ${stat.entropy_proxy.toPrecision(3)} · sparsity ${stat.sparsity_proxy.toPrecision(3)}`,
            }))}
          />
        </section>
      ) : null}

      {activations.tensors.length > 0 ? (
        <section>
          <div className="row" style={{ marginBottom: 10 }}>
            <h3>Stored tensors</h3>
            <div className="spacer" />
            <Picker
              label="Group"
              value={group}
              onChange={setGroup}
              options={[
                { value: "activations" as const, label: "Activations" },
                { value: "attention" as const, label: "Attention" },
              ]}
            />
          </div>
          <div className="table-wrap">
            <table className="data responsive">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Shape</th>
                  <th>Dtype</th>
                  <th className="num">Size</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {tensors.map((tensor) => (
                  <tr key={tensor.path}>
                    <td data-label="Layer">{tensor.key}</td>
                    <td data-label="Shape" className="mono">
                      [{tensor.shape.join(", ")}]
                    </td>
                    <td data-label="Dtype" className="mono">
                      {tensor.dtype}
                    </td>
                    <td data-label="Size" className="num">
                      {tensor.size === null ? "—" : bytes(tensor.size)}
                    </td>
                    <td data-label="File">
                      <a href={fileUrl(run.id, tensor.path)} download>
                        download
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {activations.arrays.length > 0 ? (
        <section>
          <h3 style={{ marginBottom: 10 }}>Dense arrays</h3>
          <FileTable runId={run.id} files={activations.arrays} kinds={detail.artifacts} />
        </section>
      ) : null}
    </div>
  );
}
