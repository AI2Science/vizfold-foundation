import { Link } from "../router.tsx";
import { Empty, Status, when } from "./ui.tsx";
import type { RunRow } from "../../shared/types.ts";

export default function RunsTable({ runs, limit }: { runs: RunRow[]; limit?: number }) {
  if (runs.length === 0) {
    return (
      <Empty title="No runs yet">
        <p className="note">Pick one or more proteins and hit Fold; the run shows up here.</p>
      </Empty>
    );
  }

  const shown = limit ? runs.slice(0, limit) : runs;

  return (
    <div className="table-wrap">
      <table className="data responsive">
        <thead>
          <tr>
            <th className="num">Run</th>
            <th>Proteins</th>
            <th>Model</th>
            <th>Target</th>
            <th>Status</th>
            <th>Submitted</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((run) => (
            <tr key={run.id}>
              <td className="num" data-label="Run">
                <Link href={`/runs/${run.id}`} className="run-link">
                  {run.id}
                </Link>
              </td>
              <td data-label="Proteins">
                <Link href={`/runs/${run.id}`}>{run.input_id.split("+").join(", ")}</Link>
              </td>
              <td data-label="Model">{run.model_slug}</td>
              <td data-label="Target">{run.target_slug}</td>
              <td data-label="Status">
                <Status status={run.status} />
              </td>
              <td data-label="Submitted">{when(run.submitted_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
