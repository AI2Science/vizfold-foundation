import Link from "next/link";
import { listRuns } from "@/lib/db";
import { FOLDABLE, listExamples } from "@/lib/vizfold";
import FoldCard from "@/app/FoldCard";
import Poller from "@/app/Poller";

// Read the executor db per request; never prerender a stale run list at build time.
export const dynamic = "force-dynamic";

export default async function HomePage() {
  const runs = listRuns();
  // The dashboard is still useful for browsing past runs when the CLI is unreachable, so a failed
  // lookup degrades to an empty picker (which explains itself) rather than a 500.
  const examples = await listExamples().catch(() => []);

  return (
    <main className="page-shell">
      <Poller statuses={runs.map((run) => run.status)} />
      <section className="hero-card">
        <div className="hero-copy">
          <h1 className="brand-title">VizFold</h1>
          <p className="subtitle">
            Fold a protein and inspect what the model computed — the predicted structure and the
            attention behind it.
          </p>
        </div>
      </section>

      <FoldCard examples={examples} backends={FOLDABLE} />

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
            <p>Pick a protein above and hit Fold.</p>
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
