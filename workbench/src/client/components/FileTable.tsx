import { fileUrl } from "../api.ts";
import { bytes, when } from "./ui.tsx";
import type { RunFile } from "../../shared/types.ts";

/** Every list of a run's files is the same table: what it is, how big, when it landed, and a link
 *  to it. Used for the files tab and for the dense arrays an activation dump leaves behind. */
export default function FileTable({ runId, files }: { runId: number; files: RunFile[] }) {
  return (
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
