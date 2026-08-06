import { fileUrl } from "../api.ts";
import { bytes, when } from "./ui.tsx";
import type { Artifact, RunFile } from "../../shared/types.ts";

/** Every list of a run's files is the same table: what it is, how big, when it landed, and a link
 *  to it. `kinds` is the executor's classification, which is what "what it is" means once a run
 *  has landed — before that the table still lists what is on disk. */
export default function FileTable({
  runId,
  files,
  kinds = [],
}: {
  runId: number;
  files: RunFile[];
  kinds?: Artifact[];
}) {
  const labelOf = new Map(kinds.map((artifact) => [artifact.path, artifact.type_label]));
  return (
    <div className="table-wrap">
      <table className="data responsive">
        <thead>
          <tr>
            <th>Path</th>
            <th>Kind</th>
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
              <td data-label="Kind">{labelOf.get(file.path) ?? "—"}</td>
              <td data-label="Size" className="num">
                {bytes(file.size)}
              </td>
              <td data-label="Modified">{when(file.modified)}</td>
              <td data-label="Open">
                <a href={fileUrl(runId, file.path)} target="_blank" rel="noreferrer">
                  open
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
