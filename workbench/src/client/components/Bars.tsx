import { useState } from "react";

import { useToken } from "./theme.tsx";

export type Bar = { label: string; value: number; detail?: string };

/** One measure, one axis: horizontal bars from zero, in the sequential hue's mid step. Values are
 *  direct-labelled while there are few enough rows to read, and every bar carries a hover read-out. */
export default function Bars({
  bars,
  format = (value: number) => value.toPrecision(3),
  unitLabel,
  height = 18,
  gap = 6,
}: {
  bars: Bar[];
  format?: (value: number) => string;
  unitLabel: string;
  height?: number;
  gap?: number;
}) {
  const [hover, setHover] = useState<{ x: number; y: number; bar: Bar } | null>(null);
  const [fill, grid] = useToken("--accent", "--grid");

  if (bars.length === 0) return null;

  const width = 720;
  const labelWidth = Math.min(190, Math.max(72, ...bars.map((bar) => bar.label.length * 6.5)));
  const valueWidth = bars.length <= 24 ? 78 : 0;
  const plot = width - labelWidth - valueWidth - 8;
  const max = Math.max(...bars.map((bar) => Math.abs(bar.value)), Number.EPSILON);
  const chartHeight = bars.length * (height + gap);

  return (
    <div className="chart" data-fill="true">
      <svg
        width={width}
        height={chartHeight + 18}
        viewBox={`0 0 ${width} ${chartHeight + 18}`}
        role="img"
        aria-label={`${unitLabel} by layer`}
      >
        {[0.25, 0.5, 0.75, 1].map((fraction) => (
          <line
            key={fraction}
            x1={labelWidth + plot * fraction}
            y1={0}
            x2={labelWidth + plot * fraction}
            y2={chartHeight}
            className="grid-line"
            stroke={grid}
          />
        ))}

        {bars.map((bar, index) => {
          const y = index * (height + gap);
          const length = Math.max(2, (Math.abs(bar.value) / max) * plot);
          return (
            <g
              key={bar.label}
              onMouseEnter={(event) => setHover({ x: event.clientX, y: event.clientY, bar })}
              onMouseMove={(event) => setHover({ x: event.clientX, y: event.clientY, bar })}
              onMouseLeave={() => setHover(null)}
            >
              <rect x={0} y={y - gap / 2} width={width} height={height + gap} fill="transparent" />
              <text x={0} y={y + height * 0.72} className="bar-label">
                {bar.label}
              </text>
              <rect
                x={labelWidth}
                y={y}
                width={length}
                height={height}
                rx={4}
                fill={fill}
                opacity={hover && hover.bar.label !== bar.label ? 0.55 : 1}
              />
              {valueWidth ? (
                <text
                  x={labelWidth + plot + 8}
                  y={y + height * 0.72}
                  className="bar-label"
                  textAnchor="start"
                >
                  {format(bar.value)}
                </text>
              ) : null}
            </g>
          );
        })}

        <line x1={labelWidth} y1={0} x2={labelWidth} y2={chartHeight} className="baseline" />
        <text x={labelWidth} y={chartHeight + 14} className="axis-text">
          0
        </text>
        <text x={labelWidth + plot} y={chartHeight + 14} className="axis-text" textAnchor="end">
          {format(max)} {unitLabel}
        </text>
      </svg>

      {hover ? (
        <div
          className="tooltip"
          style={{ left: Math.min(hover.x + 14, window.innerWidth - 250), top: hover.y + 14 }}
        >
          <div className="row">
            <strong>{hover.bar.label}</strong>
          </div>
          <div className="row">
            <span>{unitLabel}</span>
            <strong>{format(hover.bar.value)}</strong>
          </div>
          {hover.bar.detail ? <div className="muted">{hover.bar.detail}</div> : null}
        </div>
      ) : null}
    </div>
  );
}
