"use client";

import { useEffect, useRef, useState } from "react";

// 3Dmol drives WebGL and reads `window`, so the library is imported dynamically inside the
// effect (never during SSR) and the viewer is built after mount. Ported from the retired Flask
// web_app demo's 3Dmol viewer: cartoon/stick/line representations + default/chain/residue coloring.

type Viewer = ReturnType<(typeof import("3dmol"))["createViewer"]>;
type Representation = "cartoon" | "stick" | "line";
type Scheme = "default" | "chain" | "residue";

const SCHEME_COLOR: Record<Scheme, string | undefined> = {
  default: undefined,
  chain: "chainHetatm",
  residue: "shapely",
};

function styleFor(rep: Representation, scheme: Scheme) {
  const spec = SCHEME_COLOR[scheme] ? { colorscheme: SCHEME_COLOR[scheme] } : {};
  return rep === "cartoon" ? { cartoon: spec } : rep === "stick" ? { stick: spec } : { line: spec };
}

export default function StructureViewer({ url, name }: { url: string; name: string }) {
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<Viewer | null>(null);
  const [rep, setRep] = useState<Representation>("cartoon");
  const [scheme, setScheme] = useState<Scheme>("default");
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    (async () => {
      try {
        const $3Dmol = await import("3dmol");
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const pdb = await res.text();
        if (disposed || !host.current) return;
        const v = $3Dmol.createViewer(host.current, { backgroundColor: "white" });
        v.addModel(pdb, "pdb");
        v.setStyle({}, { cartoon: {} });
        v.zoomTo();
        v.render();
        viewer.current = v;
        setReady(true);
      } catch (e) {
        if (!disposed) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      disposed = true;
      viewer.current?.clear();
      viewer.current = null;
    };
  }, [url]);

  useEffect(() => {
    if (!ready || !viewer.current) return;
    viewer.current.setStyle({}, styleFor(rep, scheme));
    viewer.current.render();
  }, [rep, scheme, ready]);

  if (error) return <p className="field-note">Could not render {name}: {error}</p>;

  return (
    <div className="structure-viewer">
      <div ref={host} className="structure-canvas" />
      <div className="structure-controls">
        <Segmented label="Style" value={rep} onChange={setRep}
          options={[["cartoon", "Cartoon"], ["stick", "Stick"], ["line", "Line"]]} />
        <Segmented label="Color" value={scheme} onChange={setScheme}
          options={[["default", "Default"], ["chain", "Chain"], ["residue", "Residue"]]} />
        <button type="button" className="seg-btn" onClick={() => {
          viewer.current?.zoomTo();
          viewer.current?.render();
        }}>Reset view</button>
      </div>
    </div>
  );
}

function Segmented<T extends string>({ label, value, onChange, options }: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: [T, string][];
}) {
  return (
    <div className="seg">
      <span className="seg-label">{label}</span>
      {options.map(([val, text]) => (
        <button key={val} type="button"
          className={val === value ? "seg-btn seg-on" : "seg-btn"}
          onClick={() => onChange(val)}>{text}</button>
      ))}
    </div>
  );
}
