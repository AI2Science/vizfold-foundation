import { NextResponse } from "next/server";
import { foldInBackground, listExamples, queueRun } from "@/lib/vizfold";

export async function POST(request: Request) {
  const { inputId, attn = true } = await request.json();

  // The trust boundary: only an id the CLI itself listed reaches the CLI, so nothing a caller
  // typed is ever passed through as a sequence or a path.
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
    // A fold that fails later is not an error here — the run row carries its own status and
    // error_message, and the run page shows them. This is only "the run could not be created".
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: detail }, { status: 500 });
  }
}
