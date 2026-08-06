import { useCallback, useEffect, useRef, useState } from "react";

import type {
  AttentionMap,
  Environment,
  Protein,
  RunDetail,
  RunRow,
} from "../shared/types.ts";

async function get<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal, headers: { accept: "application/json" } });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const message = (body as { error?: string } | null)?.error;
    throw new Error(message ?? `${response.status} ${response.statusText}`);
  }
  return body as T;
}

export const fetchEnvironment = (signal?: AbortSignal) => get<Environment>("/api/environment", signal);
export const fetchProteins = (signal?: AbortSignal) => get<Protein[]>("/api/proteins", signal);
export const fetchRuns = (signal?: AbortSignal) => get<RunRow[]>("/api/runs", signal);
export const fetchRun = (id: number, signal?: AbortSignal) => get<RunDetail>(`/api/runs/${id}`, signal);

export const fetchAttention = (id: number, path: string, topK: number, signal?: AbortSignal) =>
  get<AttentionMap>(
    `/api/runs/${id}/attention?path=${encodeURIComponent(path)}&topK=${topK}`,
    signal,
  );

export const fileUrl = (runId: number, path: string) =>
  `/api/runs/${runId}/file?path=${encodeURIComponent(path)}`;

export async function startFold(body: {
  ids: string[];
  attn: boolean;
  backend?: string;
}): Promise<number> {
  const response = await fetch("/api/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = (await response.json().catch(() => null)) as
    | { runId?: number; error?: string }
    | null;
  if (!response.ok || typeof payload?.runId !== "number") {
    throw new Error(payload?.error ?? "Could not start the fold.");
  }
  return payload.runId;
}

type Async<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
};

/**
 * Load once, then re-load on an interval while `intervalMs` is a number. The executor's database
 * is the state, so a poll is the whole refresh: nothing is cached client-side between reads.
 */
export function useAsync<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  intervalMs: number | null,
): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const run = useRef(load);
  run.current = load;

  useEffect(() => {
    const controller = new AbortController();
    let live = true;
    let timer: ReturnType<typeof setInterval> | null = null;

    const read = async (first: boolean) => {
      if (first) setLoading(true);
      try {
        const next = await run.current(controller.signal);
        if (!live) return;
        setData(next);
        setError(null);
      } catch (problem) {
        if (!live || controller.signal.aborted) return;
        setError(problem instanceof Error ? problem.message : String(problem));
      } finally {
        if (live && first) setLoading(false);
      }
    };

    void read(true);
    if (intervalMs !== null) timer = setInterval(() => void read(false), intervalMs);

    return () => {
      live = false;
      controller.abort();
      if (timer) clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, intervalMs, nonce]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);
  return { data, error, loading, reload };
}

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

/** A run nobody is waiting on does not need polling; one in flight is re-read every few seconds. */
export const isActive = (status: string) => !TERMINAL.has(status);
