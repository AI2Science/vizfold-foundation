"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

const TERMINAL = new Set(["completed", "failed"]);

/**
 * Re-render the server component while any run is still going. Both pages are `force-dynamic`, so
 * a refresh re-reads the executor's database — no client-side run state to keep in sync.
 */
export default function Poller({ statuses }: { statuses: string[] }) {
  const router = useRouter();
  const active = statuses.some((status) => !TERMINAL.has(status));

  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => router.refresh(), 3000);
    return () => clearInterval(timer);
  }, [active, router]);

  return null;
}
