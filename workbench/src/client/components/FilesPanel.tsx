import { useState } from "react";

import FileTable from "./FileTable.tsx";
import { Banner, Field, Search, Segmented } from "./ui.tsx";
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

  const present = new Set(detail.files.map((file) => file.kind));
  const files = detail.files.filter(
    (file) =>
      (kind === "all" || file.kind === kind) &&
      (query === "" || file.path.toLowerCase().includes(query.toLowerCase())),
  );

  return (
    <div>
      <div className="control-row">
        <Segmented
          label="Kind"
          value={kind}
          onChange={setKind}
          options={KINDS.filter((option) => option.value === "all" || present.has(option.value))}
        />
        <Search value={query} onChange={setQuery} placeholder="path contains…" />
        <Field label="Run directory">
          <span className="path">{detail.root ?? "—"}</span>
        </Field>
      </div>

      <div className="panel-body stack">
        {detail.filesTruncated ? (
          <Banner tone="warning" title="Listing stopped early">
            This run wrote more files than one listing walks, so the list below is the first part of
            it. Everything is still on disk under the run directory.
          </Banner>
        ) : null}

        <FileTable runId={detail.run.id} files={files} kinds={detail.artifacts} />
        {files.length === 0 ? <p className="note">Nothing matches that filter.</p> : null}
      </div>
    </div>
  );
}
