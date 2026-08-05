import Link from "next/link";
import { readdirSync } from "node:fs";
import path from "node:path";
import AttentionViewer from "@/app/AttentionViewer";
import { IS_ARC } from "@/lib/vizfold";

// Committed sample output, so the attention view is browsable without a GPU and a folded run.
// Produced by examples/visualize_attention_arc_diagram_demo_utils.py over an OpenFold run of 6KWC.
const DIR = path.join(process.cwd(), "public", "demo-run");

export default function DemoPage() {
  const images = readdirSync(DIR)
    .filter((name) => IS_ARC.test(name))
    .map((name) => ({ name, url: `/demo-run/${name}` }));

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">
            <Link href="/">← All runs</Link>
          </p>
          <h1 className="brand-title">Attention demo</h1>
          <p className="subtitle">
            Arc diagrams from a finished OpenFold run of 6KWC, at layer 47. Each arc joins two
            residues the model attended between; thickness is the attention weight.
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Arc diagrams</h2>
          <p>{images.length} sample diagrams.</p>
        </div>
        <AttentionViewer images={images} />
      </section>
    </main>
  );
}
