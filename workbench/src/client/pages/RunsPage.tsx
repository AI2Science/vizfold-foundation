import { fetchRuns, isActive, useAsync } from "../api.ts";
import RunsTable from "../components/RunsTable.tsx";
import { Banner, Panel } from "../components/ui.tsx";

export default function RunsPage() {
  const runs = useAsync((signal) => fetchRuns(signal), [], 4000);
  const rows = runs.data ?? [];
  const active = rows.filter((run) => isActive(run.status)).length;

  return (
    <>
      {runs.error ? (
        <Banner tone="critical" title="Could not read the runs">
          {runs.error}
        </Banner>
      ) : null}
      <Panel
        title="Runs"
        subtitle={
          rows.length
            ? `${rows.length} on record${active ? `, ${active} in flight` : ""}.`
            : undefined
        }
        flush
      >
        {runs.loading && !runs.data ? (
          <div style={{ padding: 18 }}>
            <div className="skeleton" style={{ height: 120 }} />
          </div>
        ) : (
          <RunsTable runs={rows} />
        )}
      </Panel>
    </>
  );
}
