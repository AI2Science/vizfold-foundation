import Link from "next/link";
import { listRuns } from "@/lib/db";
import { FOLDABLE, listProteins } from "@/lib/vizfold";
import FoldCard from "@/app/FoldCard";
import Poller from "@/app/Poller";

// Read the executor db per request; never prerender a stale run list.
export const dynamic = "force-dynamic";

export default async function HomePage() {
  const runs = listRuns();
  // Past runs stay browsable when the CLI is unreachable: an empty picker explains itself, a 500
  // does not.
  const proteins = await listProteins().catch(() => []);
  // A batch run records its tags joined with `+`, so one row can carry several proteins.
  const folded = new Set(
    runs.filter((run) => run.status === "completed").flatMap((run) => run.input_id.split("+")),
  );

  return (
    <main className="page-shell">
      <Poller statuses={runs.map((run) => run.status)} />
      <section className="hero-card">
        <div className="hero-copy">
          <h1 className="brand-title">VizFold</h1>
          <p className="subtitle">
            Fold a protein and inspect what the model computed — the predicted structure and the
            attention behind it. No run yet? See the <Link href="/demo">attention demo</Link>.
          </p>
        </div>
        <dl className="hero-stats">
          <div>
            <dt>Serving</dt>
            <dd>{FOLDABLE.join(", ") || "no backend"}</dd>
          </div>
          <div>
            <dt>Folded</dt>
            <dd>{folded.size}</dd>
          </div>
          <div>
            <dt>Available to fold</dt>
            <dd>{proteins.length}</dd>
          </div>
        </dl>
      </section>

      <FoldCard proteins={proteins} backends={FOLDABLE} />

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
            <p>Pick one or more proteins above and hit Fold.</p>
          </div>
        ) : (
          <table className="runs-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Proteins</th>
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
                    <Link href={`/runs/${run.id}`}>{run.input_id.split("+").join(", ")}</Link>
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
