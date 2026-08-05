import { useEffect, useRef, useState } from "react";

import { useTheme } from "./theme.tsx";
import { Segmented } from "./ui.tsx";

// 3Dmol reads `window` at import time, so it is imported inside the effect and never at module load.

type Viewer = ReturnType<(typeof import("3dmol"))["createViewer"]>;
type Representation = "cartoon" | "stick" | "line" | "sphere";
type Scheme = "spectrum" | "chain" | "residue" | "bfactor";

/** pLDDT rides in the B-factor column of a predicted structure, so "confidence" is a real colour
 *  scheme here, not a decoration. */
const STYLE: Record<Scheme, Record<string, unknown>> = {
  spectrum: { color: "spectrum" },
  chain: { colorscheme: "chainHetatm" },
  residue: { colorscheme: "amino" },
  bfactor: { colorscheme: { prop: "b", gradient: "roygb", min: 50, max: 90 } },
};

const styleFor = (representation: Representation, scheme: Scheme) => ({
  [representation]: STYLE[scheme],
});

export default function StructureViewer({ url, name }: { url: string; name: string }) {
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<Viewer | null>(null);
  const watcher = useRef<ResizeObserver | null>(null);
  const [representation, setRepresentation] = useState<Representation>("cartoon");
  const [scheme, setScheme] = useState<Scheme>("spectrum");
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { theme } = useTheme();

  useEffect(() => {
    let disposed = false;
    setReady(false);
    setError(null);
    void (async () => {
      try {
        const $3Dmol = await import("3dmol");
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const text = await response.text();
        if (disposed || !host.current) return;
        const created = $3Dmol.createViewer(host.current, { backgroundAlpha: 0 });
        created.addModel(text, name.endsWith(".cif") ? "cif" : "pdb");
        created.setStyle({}, styleFor("cartoon", "spectrum"));
        created.zoomTo();
        created.render();
        viewer.current = created;
        setReady(true);
        // 3Dmol sizes its canvas from the container it was handed. The panel is still settling on
        // the frame the model lands, and it reflows again on rotate or resize, so follow it.
        const fit = () => {
          created.resize();
          created.zoomTo();
          created.render();
        };
        requestAnimationFrame(fit);
        const observer = new ResizeObserver(() => {
          created.resize();
          created.render();
        });
        observer.observe(host.current);
        watcher.current = observer;
      } catch (problem) {
        if (!disposed) setError(problem instanceof Error ? problem.message : String(problem));
      }
    })();
    return () => {
      disposed = true;
      watcher.current?.disconnect();
      watcher.current = null;
      viewer.current?.clear();
      viewer.current = null;
    };
  }, [url, name]);

  useEffect(() => {
    if (!ready || !viewer.current) return;
    viewer.current.setStyle({}, styleFor(representation, scheme));
    viewer.current.render();
  }, [representation, scheme, ready, theme]);

  return (
    <div className="structure">
      <div className="structure-canvas">
        <div ref={host} style={{ position: "absolute", inset: 0 }} />
        {error ? (
          <div className="viewer-overlay">Could not render {name}: {error}</div>
        ) : !ready ? (
          <div className="viewer-overlay">
            <span className="spinner" />
          </div>
        ) : null}
      </div>
      <div className="row">
        <Segmented
          label="Style"
          value={representation}
          onChange={setRepresentation}
          options={[
            { value: "cartoon", label: "Cartoon" },
            { value: "stick", label: "Stick" },
            { value: "line", label: "Line" },
            { value: "sphere", label: "Sphere" },
          ]}
        />
        <Segmented
          label="Colour"
          value={scheme}
          onChange={setScheme}
          options={[
            { value: "spectrum", label: "Chain spectrum" },
            { value: "residue", label: "Residue" },
            { value: "chain", label: "Chain" },
            { value: "bfactor", label: "Confidence (B)" },
          ]}
        />
        <div className="control">
          <span className="control-label">View</span>
          <button
            type="button"
            className="button"
            onClick={() => {
              viewer.current?.zoomTo();
              viewer.current?.render();
            }}
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
