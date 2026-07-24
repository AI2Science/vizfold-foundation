import Link from "next/link";
import { listRuns } from "@/lib/db";

// Read the executor db per request; never prerender a stale run list at build time.
export const dynamic = "force-dynamic";

export default function HomePage() {
  const runs = listRuns();

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">VizFold v2</p>
          <h1 className="brand-title">VizFold</h1>
          <p className="subtitle">
            Runs recorded by the VizFold executor. Queue new runs with{" "}
            <code>vizfold queue-run</code> and execute them with{" "}
            <code>vizfold execute-run</code>.
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Runs</h2>
          <p>
            {runs.length} run{runs.length === 1 ? "" : "s"} on record.
          </p>
        </div>

        {runs.length === 0 ? (
          <div className="empty-state">
            <p>No runs yet.</p>
            <p>
              Queue one with <code>vizfold queue-run openfold …</code>, then{" "}
              <code>vizfold execute-run &lt;id&gt;</code>.
            </p>
          </div>
        ) : (
          <table className="runs-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Input</th>
                <th>Model</th>
                <th>Target</th>
                <th>Status</th>
                <th>Submitted</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id}>
                  <td>
                    <Link href={`/runs/${run.id}`}>{run.id}</Link>
                  </td>
                  <td>
                    <Link href={`/runs/${run.id}`}>{run.input_id}</Link>
                  </td>
                  <td>{run.model_slug}</td>
                  <td>{run.target_slug}</td>
                  <td>
                    <span className={`status status-${run.status}`}>{run.status}</span>
                  </td>
                  <td>{run.submitted_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
