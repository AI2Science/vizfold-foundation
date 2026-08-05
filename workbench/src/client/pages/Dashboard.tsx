import { fetchProteins, fetchRuns, isActive, useAsync } from "../api.ts";
import FoldForm from "../components/FoldForm.tsx";
import RunsTable from "../components/RunsTable.tsx";
import { Banner, Panel } from "../components/ui.tsx";
import { Link } from "../router.tsx";
import type { Environment } from "../../shared/types.ts";

export default function Dashboard({
  environment,
  reloadEnvironment,
}: {
  environment: Environment | null;
  reloadEnvironment: () => void;
}) {
  const runs = useAsync((signal) => fetchRuns(signal), [], 4000);
  const proteins = useAsync((signal) => fetchProteins(signal).catch(() => []), [], null);

  const rows = runs.data ?? [];
  const active = rows.filter((run) => isActive(run.status));
  const folded = new Set(
    rows
      .filter((run) => run.status === "completed")
      .flatMap((run) => run.input_id.split("+")),
  );

  return (
    <>
      <section className="panel hero">
        <div>
          <h1>Fold, then read the model</h1>
          <p className="note" style={{ marginTop: 8 }}>
            Every fold keeps what the model computed on the way: the predicted structure, the
            attention behind it, and the activations it stored. Pick proteins below; the run page
            draws its arc diagrams straight from that run's own attention dump.
          </p>
        </div>
        <dl className="stat-row">
          <div className="stat">
            <dt>Runs</dt>
            <dd>{rows.length}</dd>
          </div>
          <div className="stat">
            <dt>In flight</dt>
            <dd>{active.length}</dd>
          </div>
          <div className="stat">
            <dt>Folded</dt>
            <dd>{folded.size}</dd>
          </div>
          <div className="stat">
            <dt>Serving</dt>
            <dd className="small">{environment?.backends.join(", ") || "—"}</dd>
          </div>
        </dl>
      </section>

      {environment && !environment.database.present ? (
        <Banner tone="warning" title="No run database yet">
          The executor creates <code>{environment.database.path || "vizfold.db"}</code> on its first
          run. Until then there is nothing to list.
        </Banner>
      ) : null}

      {runs.error ? (
        <Banner tone="critical" title="Could not read the runs">
          {runs.error}
        </Banner>
      ) : null}

      <Panel
        title="Fold proteins"
        subtitle={
          proteins.data
            ? `${proteins.data.length} bundled ${proteins.data.length === 1 ? "protein" : "proteins"} available.`
            : undefined
        }
      >
        {environment ? (
          <FoldForm
            proteins={proteins.data ?? []}
            environment={environment}
            onStarted={() => {
              runs.reload();
              reloadEnvironment();
            }}
          />
        ) : (
          <div className="skeleton" style={{ height: 120 }} />
        )}
      </Panel>

      <Panel
        title="Recent runs"
        subtitle={rows.length ? `${rows.length} on record.` : undefined}
        actions={rows.length > 8 ? <Link href="/runs" className="button" data-variant="ghost">View all</Link> : undefined}
        flush
      >
        <RunsTable runs={rows} limit={8} />
      </Panel>
    </>
  );
}
