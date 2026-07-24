import { NextResponse } from "next/server";
import { foldInBackground, listExamples, queueRun } from "@/lib/vizfold";

export async function POST(request: Request) {
  const { inputId, attn = true } = await request.json();

  // Trust boundary: only an id the CLI itself listed is ever handed back to it.
  const example = (await listExamples()).find((one) => one.id === inputId);
  if (!example) {
    return NextResponse.json(
      { error: `Unknown example "${inputId}".` },
      { status: 400 },
    );
  }

  try {
    const runId = await queueRun(example, Boolean(attn));
    foldInBackground(runId);
    return NextResponse.json({ runId }, { status: 201 });
  } catch (error) {
    // Only "could not create the run" — a later fold failure lands on the run row instead.
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: detail }, { status: 500 });
  }
}
