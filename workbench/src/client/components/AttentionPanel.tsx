import { useCallback, useMemo, useState } from "react";

import { fetchAttention, fileUrl, useAsync } from "../api.ts";
import ArcDiagram from "./ArcDiagram.tsx";
import { Amount, Banner, Empty, Picker, Segmented, bytes } from "./ui.tsx";
import type { AttentionSource, RunDetail } from "../../shared/types.ts";

const KIND_LABEL: Record<AttentionSource["kind"], string> = {
  msa_row: "MSA row",
  triangle_start: "Triangle start",
};

const TOP_K = [10, 25, 50, 100, 200, 500, 1000];

const describe = (source: AttentionSource) =>
  `${KIND_LABEL[source.kind]} · layer ${source.layer}${
    source.residue === null ? "" : ` · residue ${source.residue}`
  }`;

export default function AttentionPanel({ detail }: { detail: RunDetail }) {
  const sources = detail.attention;
  const [view, setView] = useState<"arcs" | "table">("arcs");
  const [topKIndex, setTopKIndex] = useState(2);
  const [picked, setPicked] = useState<string>("");
  const [head, setHead] = useState(0);

  const tags = useMemo(
    () => [...new Set(sources.map((source) => source.tag ?? ""))],
    [sources],
  );
  const kinds = useMemo(
    () => [...new Set(sources.map((source) => source.kind))],
    [sources],
  );

  // The picked path carries target, kind, layer and residue at once, so there is no second
  // selection to keep in sync — and a path that is no longer written falls back to the first.
  const current = sources.find((source) => source.path === picked) ?? sources[0];
  const tag = current?.tag ?? "";
  const kind = current?.kind ?? kinds[0];

  const inKind = sources.filter(
    (source) => (source.tag ?? "") === tag && source.kind === kind,
  );
  const layers = [...new Set(inKind.map((source) => source.layer))].sort((a, b) => a - b);
  const residues = [
    ...new Set(
      inKind
        .filter((source) => source.layer === current?.layer)
        .map((source) => String(source.residue ?? "")),
    ),
  ].filter(Boolean);

  const jump = useCallback(
    (next: AttentionSource | undefined) => {
      if (next) setPicked(next.path);
    },
    [],
  );

  const topK = TOP_K[topKIndex] ?? 50;
  const runId = detail.run.id;
  const path = current?.path ?? "";
  const { data, error, loading } = useAsync(
    (signal) =>
      path
        ? fetchAttention(runId, path, topK, signal)
        : Promise.reject(new Error("No attention file selected.")),
    [runId, path, topK],
    null,
  );

  if (sources.length === 0) {
    return (
      <Empty title="No attention dumped for this run">
        <p className="note">
          Attention writes as the fold runs, into <code>attention/</code> under the run directory.
          Fold with “dump attention maps” on to get it.
        </p>
      </Empty>
    );
  }

  const heads = data?.heads ?? [];
  const shown = heads.find((one) => one.head === head) ?? heads[0];

  return (
    <div>
      <div className="control-row">
        {tags.length > 1 ? (
          <Picker
            label="Target"
            value={tag}
            options={tags.map((one) => ({ value: one, label: one || "run" }))}
            onChange={(next) =>
              jump(sources.find((source) => (source.tag ?? "") === next && source.kind === kind) ??
                sources.find((source) => (source.tag ?? "") === next))
            }
          />
        ) : null}

        {kinds.length > 1 ? (
          <Picker
            label="Attention"
            value={kind ?? "msa_row"}
            options={kinds.map((one) => ({ value: one, label: KIND_LABEL[one] }))}
            onChange={(next) =>
              jump(sources.find((source) => (source.tag ?? "") === tag && source.kind === next))
            }
          />
        ) : null}

        <Picker
          label="Layer"
          value={current?.layer ?? 0}
          options={layers.map((layer) => ({ value: layer, label: `layer ${layer}` }))}
          onChange={(next) =>
            jump(inKind.find((source) => source.layer === next))
          }
        />

        {residues.length > 0 ? (
          <Picker
            label="Query residue"
            value={String(current?.residue ?? "")}
            options={residues.map((residue) => ({ value: residue, label: residue }))}
            onChange={(next) =>
              jump(
                inKind.find(
                  (source) => source.layer === current?.layer && String(source.residue) === next,
                ),
              )
            }
          />
        ) : null}

        {heads.length > 1 ? (
          <Picker
            label="Head"
            value={shown?.head ?? 0}
            options={heads.map((one) => ({ value: one.head, label: `head ${one.head}` }))}
            onChange={setHead}
          />
        ) : null}

        <Amount
          label="Edges per head"
          value={topKIndex}
          min={0}
          max={TOP_K.length - 1}
          onChange={setTopKIndex}
          format={(index) => String(TOP_K[index] ?? "")}
        />

        <Segmented
          label="View"
          value={view}
          options={[
            { value: "arcs", label: "Arcs" },
            { value: "table", label: "Table" },
          ]}
          onChange={setView}
        />
      </div>

      <div className="panel-body stack">
        <div className="row">
          <strong>{current ? describe(current) : ""}</strong>
          {data ? (
            <span className="muted">
              {data.residues} residues · {data.heads.length} heads · showing the strongest{" "}
              {shown?.edges.length ?? 0} of {data.edgesPerHead} edges this head recorded
            </span>
          ) : null}
        </div>

        {error ? <Banner tone="critical" title="Could not read that attention file">{error}</Banner> : null}
        {loading && !data ? <div className="skeleton" style={{ height: 220 }} /> : null}

        {data && shown ? (
          view === "arcs" ? (
            <ArcDiagram map={data} head={shown} />
          ) : (
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th className="num">#</th>
                    <th>From</th>
                    <th>To</th>
                    <th className="num">Separation</th>
                    <th className="num">Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.edges.map(([from, to, weight], index) => (
                    <tr key={`${from}-${to}-${index}`}>
                      <td className="num muted">{index + 1}</td>
                      <td>
                        {from}
                        {data.sequence?.[from] ? ` ${data.sequence[from]}` : ""}
                      </td>
                      <td>
                        {to}
                        {data.sequence?.[to] ? ` ${data.sequence[to]}` : ""}
                      </td>
                      <td className="num">{Math.abs(to - from)}</td>
                      <td className="num">{weight.toFixed(6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        ) : null}

        <p className="note">
          Read from <code className="path">{current?.path}</code>
          {current?.dense ? (
            <>
              {" · "}
              <a href={fileUrl(runId, current.dense)} download>
                dense array ({bytes(
                  detail.files.find((file) => file.path === current.dense)?.size ?? 0,
                )})
              </a>
            </>
          ) : null}
        </p>
      </div>
    </div>
  );
}
