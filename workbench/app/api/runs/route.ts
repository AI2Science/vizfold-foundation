import { NextResponse } from "next/server";
import { BACKENDS, foldInBackground, listProteins, queueRun } from "@/lib/vizfold";

export async function POST(request: Request) {
  const { ids, attn = true, backend = "" } = await request.json();

  // Trust boundary: only ids the CLI itself listed are ever handed back to it.
  const known = new Set((await listProteins()).map((protein) => protein.id));
  const wanted: unknown[] = Array.isArray(ids) ? ids : [];
  const picked = wanted.filter((id): id is string => typeof id === "string" && known.has(id));
  if (picked.length === 0 || picked.length !== wanted.length) {
    return NextResponse.json(
      { error: "Pick one or more listed proteins." },
      { status: 400 },
    );
  }
  // Serving nothing means nothing can fold; an unnamed backend would otherwise fall through to
  // the CLI's default, which is exactly the backend `serve` found no installation of.
  if (BACKENDS?.length === 0) {
    return NextResponse.json(
      { error: "No backend is being served." },
      { status: 400 },
    );
  }
  // Same: only a backend this dashboard serves reaches the CLI's argv.
  if (backend && BACKENDS && !BACKENDS.includes(backend)) {
    return NextResponse.json(
      { error: `Backend "${backend}" is not being served.` },
      { status: 400 },
    );
  }

  try {
    const runId = await queueRun(picked, Boolean(attn), backend);
    foldInBackground(runId);
    return NextResponse.json({ runId }, { status: 201 });
  } catch (error) {
    // Only "could not create the run" — a later fold failure lands on the run row instead.
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: detail }, { status: 500 });
  }
}
