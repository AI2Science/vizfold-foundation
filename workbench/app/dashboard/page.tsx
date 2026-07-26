"use client";

import { useState, useEffect } from "react";
import styles from "./dashboard.module.css";

type AttentionType = "msa_row" | "triangle_start";

type Artifact = {
  attentionType: AttentionType | null;
  layer: number | null;
  head: number | null;
  residueIndex: number | null;
  url: string;
  label: string;
  artifactType?: string;
  displayMode?: string;
};

type RunData = {
  runId: string;
  protein: string;
  model: string;
  status: string;
  artifacts: Artifact[];
};

export default function DashboardPage() {
  const [runData, setRunData] = useState<RunData | null>(null);
  const [loading, setLoading] = useState(true);
  const [attentionType, setAttentionType] = useState<AttentionType>("msa_row");
  const [selectedHead, setSelectedHead] = useState(0);

  useEffect(() => {
    fetch("/api/artifacts")
      .then((res) => res.json())
      .then((data: RunData) => {
        setRunData(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <main className={styles.page}>
        <div className={styles.container}>
          <p style={{ color: "#64748b" }}>Loading artifacts...</p>
        </div>
      </main>
    );
  }

  if (!runData) {
    return (
      <main className={styles.page}>
        <div className={styles.container}>
          <p style={{ color: "#ef4444" }}>Failed to load artifacts.</p>
        </div>
      </main>
    );
  }

  const attentionArtifacts = runData.artifacts.filter(
    (a) => a.attentionType === attentionType
  );

  const availableHeads = Array.from(
    new Set(attentionArtifacts.map((a) => a.head).filter((h): h is number => h !== null))
  ).sort((a, b) => a - b);

  const availableLayers = Array.from(
    new Set(attentionArtifacts.map((a) => a.layer).filter((l): l is number => l !== null))
  ).sort((a, b) => a - b);

  const current = attentionArtifacts.find((a) => a.head === selectedHead);

  const downloadArtifacts = runData.artifacts.filter(
    (a) => a.displayMode === "download"
  );

  function switchType(type: AttentionType) {
    setAttentionType(type);
    setSelectedHead(0);
  }

  return (
    <main className={styles.page}>
      <div className={styles.container}>

        <h1 className={styles.title}>VizFold - Visualization Dashboard</h1>
        <p className={styles.subtitle}>
          {runData.protein} - {runData.model} - {runData.runId}
        </p>

        <div className={styles.metaCard}>
          <div><span className={styles.metaLabel}>Job</span><span className={styles.metaValue}>{runData.runId}</span></div>
          <div><span className={styles.metaLabel}>Model</span><span className={styles.metaValue}>{runData.model}</span></div>
          <div><span className={styles.metaLabel}>Protein</span><span className={styles.metaValue}>{runData.protein}</span></div>
          <div><span className={styles.metaLabel}>Status</span><span className={styles.metaValue}>{runData.status}</span></div>
          <div><span className={styles.metaLabel}>Layers</span><span className={styles.metaValue}>{availableLayers.join(", ")}</span></div>
        </div>

        <div className={styles.viewerCard}>
          <div className={styles.controls}>
            <div>
              <p className={styles.controlLabel}>Attention type</p>
              <div className={styles.toggle}>
                <button
                  onClick={() => switchType("msa_row")}
                  className={attentionType === "msa_row" ? styles.toggleActive : styles.toggleInactive}
                >
                  MSA Row
                </button>
                <button
                  onClick={() => switchType("triangle_start")}
                  className={attentionType === "triangle_start" ? styles.toggleActive : styles.toggleInactive}
                >
                  Triangle Start
                </button>
              </div>
            </div>

            <div>
              <p className={styles.controlLabel}>Head</p>
              <select
                className={styles.select}
                value={selectedHead}
                onChange={(e) => setSelectedHead(Number(e.target.value))}
              >
                {availableHeads.map((h) => (
                  <option key={h} value={h}>Head {h}</option>
                ))}
              </select>
            </div>

            {current && (
              <p className={styles.currentLabel}>{current.label}</p>
            )}
          </div>

          <div className={styles.imageArea}>
            {current ? (
              <div>
                <img
                  src={current.url}
                  alt={current.label}
                  className={styles.mainImage}
                />
                <div className={styles.downloadRow}>
                  <a href={current.url} download className={styles.downloadLink}>
                    Download image
                  </a>
                </div>
              </div>
            ) : (
              <p style={{ color: "#94a3b8", textAlign: "center" }}>No artifact found.</p>
            )}
          </div>
        </div>

        <p className={styles.galleryLabel}>
          All heads - {attentionType === "msa_row" ? "MSA Row" : "Triangle Start"}
        </p>
        <div className={styles.gallery}>
          {availableHeads.map((h) => {
            const a = attentionArtifacts.find((x) => x.head === h);
            if (!a) return null;
            return (
              <div
                key={h}
                onClick={() => setSelectedHead(h)}
                className={selectedHead === h ? styles.thumbActive : styles.thumb}
              >
                <img src={a.url} alt={"Head " + h} className={styles.thumbImage} />
                <p className={styles.thumbLabel}>Head {h}</p>
              </div>
            );
          })}
        </div>

        {downloadArtifacts.length > 0 && (
          <div style={{ marginTop: "1.5rem" }}>
            <p className={styles.galleryLabel}>Downloads</p>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {downloadArtifacts.map((a) => (
                <a key={a.url} href={a.url} download className={styles.downloadLink}>
                  Download: {a.label}
                </a>
              ))}
            </div>
          </div>
        )}

      </div>
    </main>
  );
}
