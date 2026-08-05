import { useState } from "react";

import { useToken } from "./theme.tsx";
import { HoverTip, Reading } from "./ui.tsx";
import type { AttentionHead, AttentionMap } from "../../shared/types.ts";

/* An arc joins the two residues one attention head weighted most. Everything drawn here comes
   from the run's own attention dump, parsed on request — nothing is pre-rendered. */

type Hover = { x: number; y: number; from: number; to: number; weight: number } | null;

const PAD_X = 26;
const PAD_TOP = 20;
const AXIS_GAP = 16;
const PAD_BOTTOM = 34;

/** Mix two sRGB hex colours. The endpoints are the mode's own sequential steps. */
function mix(low: string, high: string, t: number): string {
  const parse = (hex: string): [number, number, number] => {
    const value = hex.replace("#", "");
    const full =
      value.length === 3
        ? value
            .split("")
            .map((char) => char + char)
            .join("")
        : value;
    return [
      Number.parseInt(full.slice(0, 2), 16),
      Number.parseInt(full.slice(2, 4), 16),
      Number.parseInt(full.slice(4, 6), 16),
    ];
  };
  const [r1, g1, b1] = parse(low || "#86b6ef");
  const [r2, g2, b2] = parse(high || "#0d366b");
  const clamp = Math.min(1, Math.max(0, t));
  const channel = (a: number, b: number) => Math.round(a + (b - a) * clamp);
  return `rgb(${channel(r1, r2)} ${channel(g1, g2)} ${channel(b1, b2)})`;
}

const AMINO: Record<string, string> = {
  A: "Ala", R: "Arg", N: "Asn", D: "Asp", C: "Cys", Q: "Gln", E: "Glu", G: "Gly",
  H: "His", I: "Ile", L: "Leu", K: "Lys", M: "Met", F: "Phe", P: "Pro", S: "Ser",
  T: "Thr", W: "Trp", Y: "Tyr", V: "Val",
};

const residueLabel = (sequence: string | null, index: number): string => {
  const letter = sequence?.[index];
  return letter ? `${index} ${AMINO[letter] ?? letter}` : String(index);
};

export default function ArcDiagram({
  map,
  head,
}: {
  map: AttentionMap;
  head: AttentionHead;
}) {
  const [hover, setHover] = useState<Hover>(null);
  const [low = "", high = "", muted = ""] = useToken("--seq-low", "--seq-high", "--ink-muted");

  const residues = Math.max(map.residues, map.sequence?.length ?? 0, 2);
  const width = Math.max(880, Math.min(residues * 11, 2400));
  const plot = width - PAD_X * 2;
  // Arcs fan out above (i → j forward) and below (backward), so direction is visible without colour.
  const forward = head.edges.filter(([i, j]) => i < j);
  const backward = head.edges.filter(([i, j]) => i > j);
  // Attention is mostly local, so a span scaled against the whole chain would draw as a flat line.
  // The widest pair in view sets the top of the plot; every other arc is read against it.
  const widest = Math.max(1, ...head.edges.map(([i, j]) => Math.abs(j - i)));
  const ceiling = Math.min(230, Math.max(120, plot * 0.17));

  const x = (residue: number) => PAD_X + (residues <= 1 ? plot / 2 : (residue / (residues - 1)) * plot);

  // Height grows with the residue separation, capped against the chord so a one-residue hop stays
  // a semicircle instead of a spike.
  const reachOf = (from: number, to: number) =>
    Math.min((Math.abs(to - from) / widest) ** 0.65 * ceiling, Math.abs(x(to) - x(from)) * 0.6);

  const tallest = (edges: [number, number, number][]) =>
    edges.reduce((most, [from, to]) => Math.max(most, reachOf(from, to)), 0);

  // The band each side actually needs — no empty half-panel above a run of short-range heads.
  const arcTop = Math.max(36, tallest(forward));
  const arcBottom = backward.length ? Math.max(36, tallest(backward)) : 0;
  const axisY = PAD_TOP + arcTop;
  const height = axisY + AXIS_GAP + arcBottom + PAD_BOTTOM;

  const span = head.max - head.min;
  const strength = (weight: number) => (span > 1e-12 ? (weight - head.min) / span : 1);

  // One label per residue while they fit; past that, numbered stops — every fourth letter of a
  // sequence reads as a sequence, and is not one.
  const ticks: number[] = [];
  const every = residues <= 90 ? 1 : Math.ceil(residues / 45);
  for (let index = 0; index < residues; index += every) ticks.push(index);

  const arc = (from: number, to: number, up: boolean) => {
    const [x1, x2] = [x(from), x(to)];
    const reach = reachOf(from, to);
    const tip = up ? axisY - reach * 2 : axisY + AXIS_GAP + reach * 2;
    const start = up ? axisY : axisY + AXIS_GAP;
    return `M ${x1.toFixed(2)} ${start} Q ${((x1 + x2) / 2).toFixed(2)} ${tip.toFixed(2)} ${x2.toFixed(2)} ${start}`;
  };

  const edgeMarks = (edges: [number, number, number][], up: boolean) =>
    edges.map(([from, to, weight], index) => {
      const t = strength(weight);
      const stroke = mix(low, high, 0.25 + t * 0.75);
      const enter = (event: React.MouseEvent) =>
        setHover({ x: event.clientX, y: event.clientY, from, to, weight });
      if (from === to) {
        // A residue attending itself has no span to arc over; it reads as a dot on the axis.
        return (
          <circle
            key={`self-${from}-${index}`}
            cx={x(from)}
            cy={up ? axisY - 4 : axisY + AXIS_GAP + 4}
            r={2 + t * 2}
            fill={stroke}
            onMouseEnter={enter}
            onMouseMove={enter}
            onMouseLeave={() => setHover(null)}
          />
        );
      }
      const d = arc(from, to, up);
      return (
        <g key={`${from}-${to}-${index}`}>
          <path d={d} fill="none" stroke={stroke} strokeWidth={1 + t * 2.5} strokeLinecap="round" opacity={0.85} />
          {/* A hit target wider than the mark: 2px strokes are hard to hover exactly. */}
          <path
            d={d}
            fill="none"
            stroke="transparent"
            strokeWidth={12}
            onMouseEnter={enter}
            onMouseMove={enter}
            onMouseLeave={() => setHover(null)}
          />
        </g>
      );
    });

  if (head.edges.length === 0) {
    return <p className="note">This head recorded no attention edges.</p>;
  }

  return (
    <figure className="chart-figure">
      {/* A short chain stretches to the panel; a long one keeps its width and scrolls. */}
      <div className="chart" data-fill={width <= 1200}>
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`Arc diagram of ${head.edges.length} attention edges for head ${head.head}`}
        >
          {edgeMarks(forward, true)}
          {edgeMarks(backward, false)}

          <line x1={PAD_X} y1={axisY} x2={width - PAD_X} y2={axisY} className="baseline" />
          <line x1={PAD_X} y1={axisY + AXIS_GAP} x2={width - PAD_X} y2={axisY + AXIS_GAP} className="baseline" />

          {ticks.map((index) => (
            <text
              key={index}
              x={x(index)}
              y={axisY + AXIS_GAP - 5}
              textAnchor="middle"
              className="axis-text"
            >
              {every === 1 ? (map.sequence?.[index] ?? index) : index}
            </text>
          ))}

          {map.source.residue !== null && map.source.residue !== "avg" ? (
            <g>
              <line
                x1={x(map.source.residue)}
                y1={PAD_TOP}
                x2={x(map.source.residue)}
                y2={axisY + AXIS_GAP + arcBottom}
                stroke={muted}
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <text x={x(map.source.residue)} y={PAD_TOP - 6} textAnchor="middle" className="axis-text">
                query {map.source.residue}
              </text>
            </g>
          ) : null}
        </svg>
      </div>

      <figcaption className="chart-caption row">
        <span className="legend">
          <span>weaker</span>
          <span className="legend-ramp" />
          <span>stronger</span>
          <span className="muted">
            {head.min.toFixed(4)} – {head.max.toFixed(4)}
          </span>
        </span>
        <span className="muted">
          {forward.length} forward (above) · {backward.length} backward (below)
        </span>
      </figcaption>

      {hover ? (
        <HoverTip x={hover.x} y={hover.y}>
          <Reading label="from" value={residueLabel(map.sequence, hover.from)} />
          <Reading label="to" value={residueLabel(map.sequence, hover.to)} />
          <Reading label="weight" value={hover.weight.toFixed(6)} />
        </HoverTip>
      ) : null}
    </figure>
  );
}
