import { useMemo, useState } from "react";

import { fileUrl } from "../api.ts";
import { Banner, Empty, Segmented, bytes, when } from "./ui.tsx";
import type { FileKind, RunDetail } from "../../shared/types.ts";

const KINDS: { value: FileKind | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "structure", label: "Structures" },
  { value: "text", label: "Text" },
  { value: "tensor", label: "Tensors" },
  { value: "image", label: "Images" },
  { value: "other", label: "Other" },
];

export default function FilesPanel({ detail }: { detail: RunDetail }) {
  const [kind, setKind] = useState<FileKind | "all">("all");
  const [query, setQuery] = useState("");

  const present = useMemo(
    () => new Set(detail.files.map((file) => file.kind)),
    [detail.files],
  );
  const files = detail.files.filter(
    (file) =>
      (kind === "all" || file.kind === kind) &&
      (query === "" || file.path.toLowerCase().includes(query.toLowerCase())),
  );

  if (detail.files.length === 0) {
    return (
      <Empty title="Nothing written yet">
        <p className="note">
          {detail.root
            ? `The run directory ${detail.root} is empty.`
            : "The executor has not created this run's output directory."}
        </p>
      </Empty>
    );
  }

  return (
    <div>
      <div className="control-row">
        <Segmented
          label="Kind"
          value={kind}
          onChange={setKind}
          options={KINDS.filter((option) => option.value === "all" || present.has(option.value))}
        />
        <div className="control">
          <span className="control-label">Filter</span>
          <input
            className="select-trigger"
            style={{ minWidth: 200 }}
            value={query}
            placeholder="path contains…"
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <div className="control">
          <span className="control-label">Run directory</span>
          <span className="path">{detail.root ?? "—"}</span>
        </div>
      </div>

      <div className="panel-body stack">
        {detail.filesTruncated ? (
          <Banner tone="warning" title="Listing stopped early">
            This run wrote more files than one listing walks, so the list below is the first part of
            it. Everything is still on disk under the run directory.
          </Banner>
        ) : null}

        <div className="table-wrap">
          <table className="data responsive">
            <thead>
              <tr>
                <th>Path</th>
                <th className="num">Size</th>
                <th>Modified</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {files.map((file) => (
                <tr key={file.path}>
                  <td data-label="Path" className="path">
                    {file.path}
                  </td>
                  <td data-label="Size" className="num">
                    {bytes(file.size)}
                  </td>
                  <td data-label="Modified">{when(file.modified)}</td>
                  <td data-label="Open">
                    <a href={fileUrl(detail.run.id, file.path)} target="_blank" rel="noreferrer">
                      open
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {files.length === 0 ? <p className="note">Nothing matches that filter.</p> : null}
      </div>
    </div>
  );
}
