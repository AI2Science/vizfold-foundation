import index from "../client/index.html";
import { readAttention, sequenceFor } from "./attention.ts";
import { databasePresent, listRuns } from "./db.ts";
import { BACKENDS, BIN, DB_PATH, FOLDABLE, PORT, PREFIX, RUNS_DIR } from "./env.ts";
import { readRunDetail } from "./rundetail.ts";
import { resolveInside } from "./runfiles.ts";
import { foldInBackground, listProteins, queueRun } from "./vizfold.ts";
import type { Environment, Protein } from "../shared/types.ts";

const json = (body: unknown, status = 200) => Response.json(body, { status });
const fail = (message: string, status: number) => json({ error: message }, status);

/** `list proteins` shells out; the fold form and the environment card both want it, and the client
 *  polls while a run is going. One answer per window, so polling never fans out into processes. */
const PROTEIN_TTL_MS = 5_000;
let cached: { at: number; proteins: Protein[]; error: string | null } | null = null;

async function proteins(): Promise<{ proteins: Protein[]; error: string | null }> {
  if (cached && Date.now() - cached.at < PROTEIN_TTL_MS) return cached;
  try {
    cached = { at: Date.now(), proteins: await listProteins(), error: null };
  } catch (problem) {
    cached = {
      at: Date.now(),
      proteins: [],
      error: problem instanceof Error ? problem.message : String(problem),
    };
  }
  return cached;
}

async function environment(): Promise<Environment> {
  const { error } = await proteins();
  return {
    backends: FOLDABLE,
    backendsConfigured: BACKENDS !== null,
    prefix: PREFIX,
    runsDir: RUNS_DIR,
    database: { path: DB_PATH, present: databasePresent() },
    cli: { bin: BIN, ok: error === null, error },
  };
}

const runIdOf = (raw: string | undefined) => {
  const id = Number(raw);
  return Number.isInteger(id) && id > 0 ? id : null;
};

async function startFold(request: Request): Promise<Response> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return fail("Expected a JSON body.", 400);
  }
  const { ids, attn = true, backend = "" } = (body ?? {}) as {
    ids?: unknown;
    attn?: unknown;
    backend?: unknown;
  };

  const { proteins: known, error } = await proteins();
  if (error) return fail(error, 502);

  // Trust boundary: only ids the CLI itself listed are ever handed back to it.
  const listed = new Set(known.map((protein) => protein.id));
  const wanted: unknown[] = Array.isArray(ids) ? ids : [];
  const picked = wanted.filter((id): id is string => typeof id === "string" && listed.has(id));
  if (picked.length === 0 || picked.length !== wanted.length) {
    return fail("Pick one or more listed proteins.", 400);
  }
  // Serving nothing means nothing can fold; an unnamed backend would otherwise fall through to
  // the CLI's default, which is exactly the backend `serve` found no installation of.
  if (BACKENDS?.length === 0) return fail("No backend is being served.", 400);
  // Same: only a backend this dashboard serves reaches the CLI's argv.
  if (typeof backend !== "string") return fail("Backend must be a string.", 400);
  if (backend && BACKENDS && !BACKENDS.includes(backend)) {
    return fail(`Backend "${backend}" is not being served.`, 400);
  }

  try {
    const runId = await queueRun(picked, Boolean(attn), backend);
    foldInBackground(runId);
    return json({ runId }, 201);
  } catch (problem) {
    // Only "could not create the run" — a later fold failure lands on the run row instead.
    return fail(problem instanceof Error ? problem.message : String(problem), 500);
  }
}

const server = Bun.serve({
  port: PORT,
  development: process.env.NODE_ENV !== "production",
  routes: {
    "/api/environment": { GET: async () => json(await environment()) },

    "/api/proteins": {
      GET: async () => {
        const { proteins: listed, error } = await proteins();
        return error ? fail(error, 502) : json(listed);
      },
    },

    "/api/runs": {
      GET: () => json(listRuns()),
      POST: startFold,
    },

    "/api/runs/:id": {
      GET: async (request) => {
        const id = runIdOf(request.params.id);
        if (id === null) return fail("Not a run id.", 400);
        const detail = await readRunDetail(id);
        return detail ? json(detail) : fail(`No run ${id}.`, 404);
      },
    },

    // The arc diagrams are drawn in the browser from these edges — the run's own attention dump,
    // parsed on demand, never a rendered image checked in beside the code.
    "/api/runs/:id/attention": {
      GET: async (request) => {
        const id = runIdOf(request.params.id);
        if (id === null) return fail("Not a run id.", 400);
        const url = new URL(request.url);
        const wanted = url.searchParams.get("path") ?? "";
        const rawTopK = url.searchParams.get("topK");
        const topK = rawTopK === null || rawTopK === "all" ? null : Number(rawTopK);
        if (topK !== null && (!Number.isInteger(topK) || topK < 1)) {
          return fail("topK must be a positive integer or 'all'.", 400);
        }

        const detail = await readRunDetail(id);
        if (!detail || !detail.root) return fail(`No run ${id}.`, 404);
        const source = detail.attention.find((candidate) => candidate.path === wanted);
        if (!source) return fail("That run wrote no such attention file.", 404);
        const full = resolveInside(detail.root, source.path);
        if (!full) return fail("That path is outside the run.", 400);

        const sequence = sequenceFor(detail.run.input_id, detail.run.input_sequence, source.tag);
        return json(await readAttention(full, source, sequence, topK));
      },
    },

    // Structures for the 3D viewer, images, and every download link on the files tab.
    "/api/runs/:id/file": {
      GET: async (request) => {
        const id = runIdOf(request.params.id);
        if (id === null) return fail("Not a run id.", 400);
        const detail = await readRunDetail(id);
        if (!detail || !detail.root) return fail(`No run ${id}.`, 404);
        const wanted = new URL(request.url).searchParams.get("path") ?? "";
        const full = resolveInside(detail.root, wanted);
        if (!full) return fail("That path is outside the run.", 400);
        const file = Bun.file(full);
        if (!(await file.exists())) return fail("No such file in this run.", 404);
        return new Response(file);
      },
    },

    "/*": index,
  },
});

const where = PREFIX ? PREFIX : "no OPENFOLD_PREFIX — start it with `vizfold serve`";
console.log(`VizFold workbench on http://localhost:${server.port} (${where})`);
console.log(`  runs      ${RUNS_DIR || "—"}`);
console.log(`  database  ${DB_PATH || "—"}${databasePresent() ? "" : " (not created yet)"}`);
console.log(`  backends  ${FOLDABLE.join(", ") || "none served"}`);
