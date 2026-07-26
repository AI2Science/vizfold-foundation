"use client";

import { useState } from "react";

// ============================================================
// Types
// ============================================================

export type AttentionType = "msa_row" | "triangle_start";

export type Artifact = {
  attentionType: AttentionType;
  layer: number;
  head: number;
  residueIndex?: number; // only for triangle_start
  kind: "arc" | "3d" | "combined";
  url: string;           // path to the PNG file
  label: string;         // human-readable label
};

export type ArtifactViewerProps = {
  artifacts: Artifact[];
  proteinName: string;
};

// ============================================================
// Helper: filter artifacts by current selections
// ============================================================

function filterArtifacts(
  artifacts: Artifact[],
  attentionType: AttentionType,
  layer: number,
  head: number,
  kind: "arc" | "3d" | "combined"
): Artifact[] {
  return artifacts.filter(
    (a) =>
      a.attentionType === attentionType &&
      a.layer === layer &&
      a.head === head &&
      a.kind === kind
  );
}

function unique<T>(arr: T[]): T[] {
  return Array.from(new Set(arr));
}

// ============================================================
// Main Component
// ============================================================

export function ArtifactViewer({ artifacts, proteinName }: ArtifactViewerProps) {
  // Derive available options from the artifact list
  const availableLayers = unique(artifacts.map((a) => a.layer)).sort((a, b) => a - b);
  const availableHeads = unique(artifacts.map((a) => a.head)).sort((a, b) => a - b);

  const [attentionType, setAttentionType] = useState<AttentionType>("msa_row");
  const [selectedLayer, setSelectedLayer] = useState<number>(availableLayers[0] ?? 0);
  const [selectedHead, setSelectedHead] = useState<number>(availableHeads[0] ?? 0);
  const [selectedKind, setSelectedKind] = useState<"arc" | "3d" | "combined">("arc");

  const filtered = filterArtifacts(
    artifacts,
    attentionType,
    selectedLayer,
    selectedHead,
    selectedKind
  );

  const currentArtifact = filtered[0] ?? null;

  if (artifacts.length === 0) {
    return (
      <div className="empty-state">
        <p>No artifacts available.</p>
        <p>Run a job to generate visualizations.</p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>

      {/* ── Controls ── */}
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "1rem",
        padding: "1rem",
        background: "#f8f9fc",
        borderRadius: "0.75rem",
        border: "1px solid #e2e8f0"
      }}>

        {/* Attention type */}
        <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem", fontSize: "0.85rem", fontWeight: 600 }}>
          Attention type
          <select
            value={attentionType}
            onChange={(e) => setAttentionType(e.target.value as AttentionType)}
            style={selectStyle}
          >
            <option value="msa_row">MSA Row</option>
            <option value="triangle_start">Triangle Start</option>
          </select>
        </label>

        {/* Layer */}
        <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem", fontSize: "0.85rem", fontWeight: 600 }}>
          Layer
          <select
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(Number(e.target.value))}
            style={selectStyle}
          >
            {availableLayers.map((l) => (
              <option key={l} value={l}>Layer {l}</option>
            ))}
          </select>
        </label>

        {/* Head */}
        <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem", fontSize: "0.85rem", fontWeight: 600 }}>
          Head
          <select
            value={selectedHead}
            onChange={(e) => setSelectedHead(Number(e.target.value))}
            style={selectStyle}
          >
            {availableHeads.map((h) => (
              <option key={h} value={h}>Head {h}</option>
            ))}
          </select>
        </label>

        {/* View kind */}
        <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem", fontSize: "0.85rem", fontWeight: 600 }}>
          View
          <select
            value={selectedKind}
            onChange={(e) => setSelectedKind(e.target.value as "arc" | "3d" | "combined")}
            style={selectStyle}
          >
            <option value="arc">Arc diagram</option>
            <option value="3d">3D overlay</option>
            <option value="combined">Combined panel</option>
          </select>
        </label>
      </div>

      {/* ── Image display ── */}
      <div style={{
        borderRadius: "0.75rem",
        border: "1px solid #e2e8f0",
        overflow: "hidden",
        background: "#fff",
        minHeight: "300px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}>
        {currentArtifact ? (
          <div style={{ width: "100%", padding: "1rem" }}>
            <p style={{ fontSize: "0.8rem", color: "#64748b", marginBottom: "0.5rem" }}>
              {proteinName} · {currentArtifact.label}
            </p>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentArtifact.url}
              alt={currentArtifact.label}
              style={{ width: "100%", borderRadius: "0.5rem" }}
            />
          </div>
        ) : (
          <p style={{ color: "#94a3b8", fontSize: "0.9rem" }}>
            No artifact found for this combination of attention type, layer, head, and view.
          </p>
        )}
      </div>

      {/* ── Download link ── */}
      {currentArtifact && (
        <a
          href={currentArtifact.url}
          download
          style={{
            display: "inline-block",
            fontSize: "0.85rem",
            color: "#3b82f6",
            textDecoration: "underline",
            cursor: "pointer"
          }}
        >
          Download this image
        </a>
      )}
    </div>
  );
}

// ============================================================
// Styles
// ============================================================

const selectStyle: React.CSSProperties = {
  padding: "0.4rem 0.6rem",
  borderRadius: "0.5rem",
  border: "1px solid #cbd5e1",
  fontSize: "0.85rem",
  background: "#fff",
  cursor: "pointer",
  minWidth: "130px"
};


// ============================================================
// Mock data helper — use this to test the viewer locally
// before real artifacts are wired in from the API.
// Replace these URLs with real artifact paths once available.
// ============================================================

export function buildMockArtifacts(proteinName: string): Artifact[] {
  // These point to placeholder images in /public/mock-artifacts/
  // Drop real PNGs there to test rendering locally.
  const layers = [0, 18, 47];
  const heads = [0, 1, 2, 3];
  const kinds: Array<"arc" | "3d" | "combined"> = ["arc", "3d", "combined"];
  const types: AttentionType[] = ["msa_row", "triangle_start"];

  const artifacts: Artifact[] = [];

  for (const attentionType of types) {
    for (const layer of layers) {
      for (const head of heads) {
        for (const kind of kinds) {
          artifacts.push({
            attentionType,
            layer,
            head,
            kind,
            // Points to a placeholder image; swap with real paths later
            url: `/mock-artifacts/placeholder.png`,
            label: `${attentionType === "msa_row" ? "MSA Row" : "Triangle Start"} · Layer ${layer} · Head ${head} · ${kind}`,
          });
        }
      }
    }
  }

  return artifacts;
}
