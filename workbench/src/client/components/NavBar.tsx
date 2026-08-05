import { useTheme } from "./theme.tsx";
import { Link, useRoute } from "../router.tsx";
import type { Environment } from "../../shared/types.ts";

function Mark() {
  return (
    <svg className="brand-mark" width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <path
        d="M3 15.5c3.2 0 3.2-9 6.4-9s3.2 9 6.4 9"
        stroke="var(--accent)"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx="3" cy="15.5" r="1.9" fill="var(--accent)" />
      <circle cx="19" cy="15.5" r="1.9" fill="var(--ink-2)" />
    </svg>
  );
}

export default function NavBar({ environment }: { environment: Environment | null }) {
  const { theme, toggle } = useTheme();
  const route = useRoute();

  return (
    <nav className="nav">
      <div className="nav-inner">
        <Link href="/" className="brand">
          <Mark />
          VizFold
        </Link>
        <div className="nav-links">
          <Link href="/" className="nav-link" aria-current={route.name === "home" ? "page" : undefined}>
            Fold
          </Link>
          <Link
            href="/runs"
            className="nav-link"
            aria-current={route.name === "runs" || route.name === "run" ? "page" : undefined}
          >
            Runs
          </Link>
        </div>
        <div className="nav-right">
          {environment ? (
            <span className="env-chip" title={environment.prefix || "no OPENFOLD_PREFIX"}>
              <span className="status" data-tone={environment.cli.ok ? "good" : "critical"}>
                <span className="dot" />
                {environment.cli.ok ? "cli ready" : "cli unreachable"}
              </span>
              {environment.backends.length ? environment.backends.join(" · ") : "no backend served"}
            </span>
          ) : null}
          <button
            type="button"
            className="icon-button"
            onClick={toggle}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <circle cx="8" cy="8" r="3.2" stroke="currentColor" strokeWidth="1.4" />
                <path
                  d="M8 1.6v1.5M8 12.9v1.5M1.6 8h1.5M12.9 8h1.5M3.5 3.5l1 1M11.5 11.5l1 1M12.5 3.5l-1 1M4.5 11.5l-1 1"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path
                  d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8Z"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinejoin="round"
                />
              </svg>
            )}
          </button>
        </div>
      </div>
    </nav>
  );
}
