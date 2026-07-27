import { NextResponse } from "next/server";
import { BACKENDS, foldInBackground, listExamples, queueRun } from "@/lib/vizfold";

export async function POST(request: Request) {
  const { inputId, attn = true, backend = "" } = await request.json();

  // Trust boundary: only an id the CLI itself listed is ever handed back to it.
  const example = (await listExamples()).find((one) => one.id === inputId);
  if (!example) {
    return NextResponse.json(
      { error: `Unknown example "${inputId}".` },
      { status: 400 },
    );
  }
  // Same: only a backend this dashboard serves reaches the CLI's argv.
  if (backend && !BACKENDS.includes(backend)) {
    return NextResponse.json(
      { error: `Backend "${backend}" is not being served.` },
      { status: 400 },
    );
  }

  try {
    const runId = await queueRun(example, Boolean(attn), backend);
    foldInBackground(runId);
    return NextResponse.json({ runId }, { status: 201 });
  } catch (error) {
    // Only "could not create the run" — a later fold failure lands on the run row instead.
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: detail }, { status: 500 });
  }
}
