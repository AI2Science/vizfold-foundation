import { Checkbox } from "@base-ui-components/react/checkbox";
import { Select } from "@base-ui-components/react/select";
import { Slider } from "@base-ui-components/react/slider";
import { Switch } from "@base-ui-components/react/switch";
import { Toast } from "@base-ui-components/react/toast";
import { Toggle } from "@base-ui-components/react/toggle";
import { ToggleGroup } from "@base-ui-components/react/toggle-group";
import type { ReactNode } from "react";

/* Base UI ships behaviour, not looks: every part below is styled by `app.css` alone, so the
   dashboard has one design system rather than a component library's plus ours. */

export function Chevron() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M3 4.5 6 7.5 9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function Check({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M2.5 6.3 4.8 8.6 9.5 3.6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export type Option<T extends string | number> = { value: T; label: string; hint?: string };

export function Picker<T extends string | number>({
  label,
  value,
  options,
  onChange,
  disabled,
  width,
}: {
  label: string;
  value: T;
  options: Option<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
  width?: number;
}) {
  const selected = options.find((option) => option.value === value);
  return (
    <div className="control">
      <span className="control-label">{label}</span>
      <Select.Root
        value={value}
        onValueChange={(next) => onChange(next as T)}
        disabled={disabled || options.length === 0}
      >
        <Select.Trigger className="select-trigger" style={width ? { minWidth: width } : undefined}>
          <Select.Value className="select-value">{selected?.label ?? "—"}</Select.Value>
          <Select.Icon className="select-icon">
            <Chevron />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Positioner sideOffset={6} alignItemWithTrigger={false}>
            <Select.Popup className="select-popup">
              {options.map((option) => (
                <Select.Item key={String(option.value)} value={option.value} className="select-item">
                  <Select.ItemText>{option.label}</Select.ItemText>
                  {option.hint ? <span className="muted">{option.hint}</span> : null}
                  <Select.ItemIndicator className="select-item-indicator">
                    <Check />
                  </Select.ItemIndicator>
                </Select.Item>
              ))}
            </Select.Popup>
          </Select.Positioner>
        </Select.Portal>
      </Select.Root>
    </div>
  );
}

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label?: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="control">
      {label ? <span className="control-label">{label}</span> : null}
      <ToggleGroup
        className="toggle-group"
        value={[value]}
        onValueChange={(next) => {
          const picked = next[0];
          // Toggling the pressed item off would leave no view selected; keep the current one.
          if (typeof picked === "string") onChange(picked as T);
        }}
      >
        {options.map((option) => (
          <Toggle key={option.value} value={option.value} className="toggle">
            {option.label}
          </Toggle>
        ))}
      </ToggleGroup>
    </div>
  );
}

export function Amount({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  return (
    <div className="control">
      <span className="control-label">{label}</span>
      <Slider.Root
        className="slider"
        value={value}
        min={min}
        max={max}
        step={step}
        onValueChange={(next) => onChange(typeof next === "number" ? next : (next[0] ?? min))}
      >
        <Slider.Control className="slider-control">
          <Slider.Track className="slider-track">
            <Slider.Indicator className="slider-indicator" />
            <Slider.Thumb className="slider-thumb" />
          </Slider.Track>
        </Slider.Control>
        <span className="slider-value">{format ? format(value) : value}</span>
      </Slider.Root>
    </div>
  );
}

export function Toggler({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  hint?: string;
  disabled?: boolean;
}) {
  return (
    <label className="switch-field">
      <Switch.Root
        className="switch"
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
      >
        <Switch.Thumb className="switch-thumb" />
      </Switch.Root>
      <span className="label">
        {label}
        {hint ? <span className="hint">{hint}</span> : null}
      </span>
    </label>
  );
}

export function Tick({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <Checkbox.Root className="checkbox" checked={checked} onCheckedChange={onChange} disabled={disabled}>
      <Checkbox.Indicator>
        <Check />
      </Checkbox.Indicator>
    </Checkbox.Root>
  );
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
  flush,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  flush?: boolean;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {actions ? <div className="spacer row">{actions}</div> : null}
      </div>
      <div className={flush ? "panel-body flush" : "panel-body"}>{children}</div>
    </section>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {children}
    </div>
  );
}

export type Tone = "neutral" | "good" | "warning" | "critical";

/** Status is never colour alone: the dot is paired with the executor's own word for the state. */
export function Status({ status }: { status: string }) {
  const tone: Tone =
    status === "completed"
      ? "good"
      : status === "failed" || status === "cancelled"
        ? "critical"
        : "warning";
  return (
    <span className="status" data-tone={tone}>
      <span className="dot" />
      {status}
    </span>
  );
}

export function Banner({
  tone = "neutral",
  title,
  children,
}: {
  tone?: Tone;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="banner" data-tone={tone} role={tone === "critical" ? "alert" : undefined}>
      <svg className="banner-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <circle cx="8" cy="8" r="6.6" stroke="currentColor" strokeWidth="1.4" />
        <path d="M8 4.8v4M8 11.1v.1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
      <div>
        <strong>{title}</strong>
        {children ? <div>{children}</div> : null}
      </div>
    </div>
  );
}

export function Toasts() {
  const { toasts } = Toast.useToastManager();
  return (
    <Toast.Portal>
      <Toast.Viewport className="toast-viewport">
        {toasts.map((toast) => (
          <Toast.Root key={toast.id} toast={toast} className="toast" data-tone={toast.type}>
            <Toast.Title className="toast-title" />
            <Toast.Description className="toast-description" />
          </Toast.Root>
        ))}
      </Toast.Viewport>
    </Toast.Portal>
  );
}

export const bytes = (size: number): string => {
  if (size < 1024) return `${size} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = size / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
};

/** The executor writes SQLite timestamps in UTC without a zone; read them as UTC, show them local. */
export function when(stamp: string | null): string {
  if (!stamp) return "—";
  const iso = /(Z|[+-]\d\d:?\d\d)$/.test(stamp) ? stamp : `${stamp.replace(" ", "T")}Z`;
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? stamp
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
