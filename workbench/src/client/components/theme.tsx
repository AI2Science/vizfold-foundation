import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export type Theme = "light" | "dark";

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({
  theme: "light",
  toggle: () => {},
});

const KEY = "vizfold-theme";

function initial(): Theme {
  const stored = localStorage.getItem(KEY);
  if (stored === "light" || stored === "dark") return stored;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initial);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(KEY, theme);
  }, [theme]);

  const toggle = useCallback(() => setTheme((current) => (current === "dark" ? "light" : "dark")), []);
  const value = useMemo(() => ({ theme, toggle }), [theme, toggle]);
  return <ThemeContext value={value}>{children}</ThemeContext>;
}

export const useTheme = () => useContext(ThemeContext);

/** Chart marks are drawn with the same tokens the page is built from: read them off the document
 *  so a mode switch repaints the SVG with the mode's own validated steps. */
export function useToken(...names: string[]): string[] {
  const { theme } = useTheme();
  return useMemo(() => {
    const styles = getComputedStyle(document.documentElement);
    return names.map((name) => styles.getPropertyValue(name).trim());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, names.join(",")]);
}
