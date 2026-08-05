import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { AnchorHTMLAttributes, ReactNode } from "react";

/* Three pages and deep links that survive a reload — the History API is the whole router. */

export type Route =
  | { name: "home" }
  | { name: "runs" }
  | { name: "run"; id: number }
  | { name: "missing" };

export function parse(pathname: string): Route {
  if (pathname === "/" || pathname === "") return { name: "home" };
  if (pathname === "/runs") return { name: "runs" };
  const run = /^\/runs\/(\d+)\/?$/.exec(pathname);
  if (run) return { name: "run", id: Number(run[1]) };
  return { name: "missing" };
}

const RouterContext = createContext<{ route: Route; navigate: (href: string) => void }>({
  route: { name: "home" },
  navigate: () => {},
});

export function Router({ children }: { children: ReactNode }) {
  const [path, setPath] = useState(() => location.pathname);

  useEffect(() => {
    const onPop = () => setPath(location.pathname);
    addEventListener("popstate", onPop);
    return () => removeEventListener("popstate", onPop);
  }, []);

  const navigate = useCallback((href: string) => {
    if (href === location.pathname) return;
    history.pushState(null, "", href);
    setPath(href);
    scrollTo({ top: 0 });
  }, []);

  const value = useMemo(() => ({ route: parse(path), navigate }), [path, navigate]);
  return <RouterContext value={value}>{children}</RouterContext>;
}

export const useRoute = () => useContext(RouterContext).route;
export const useNavigate = () => useContext(RouterContext).navigate;

export function Link({
  href,
  children,
  ...rest
}: { href: string; children: ReactNode } & AnchorHTMLAttributes<HTMLAnchorElement>) {
  const navigate = useNavigate();
  return (
    <a
      href={href}
      onClick={(event) => {
        // Modified clicks belong to the browser: new tab, download, new window.
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
        event.preventDefault();
        navigate(href);
      }}
      {...rest}
    >
      {children}
    </a>
  );
}
