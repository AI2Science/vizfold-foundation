import type {
  ExecutionTargetOption,
  ModelOption,
} from "@/components/home-page-client";

export const modelOptions: ModelOption[] = [
  {
    slug: "openfold-local",
    label: "OpenFold (Local Prototype)",
    summary: "Reference structure prediction flow for local executor integration experiments.",
    capabilities: ["Monomer folding", "Prototype visualization handoff", "Local mock execution"],
  },
  {
    slug: "esmfold-fast",
    label: "ESMFold (Fast Preview)",
    summary: "Fast iteration path for trying alternate folding backends in the prototype workbench.",
    capabilities: ["Fast preview runs", "Sequence-only input", "Prototype comparison workflow"],
  },
  {
    slug: "boltz-experimental",
    label: "Boltz (Experimental)",
    summary: "Reserved slot for future backend comparison once the executor and CLI flow settle.",
    capabilities: ["Future adapter target", "Comparative benchmarking", "WIP placeholder"],
  },
];

export const executionTargets: ExecutionTargetOption[] = [
  {
    slug: "local-mock",
    label: "Local Mock",
    summary: "Pure frontend placeholder mode for layout and interaction testing.",
    kind: "local_mock",
    status: "available",
  },
  {
    slug: "executor-http",
    label: "Rust Executor HTTP",
    summary: "Planned handoff to the Rust executor API after the CLI-first flow is finalized.",
    kind: "gateway",
    status: "planned",
  },
  {
    slug: "executor-cli",
    label: "Rust Executor CLI",
    summary: "Future direct invocation path once the CLI adapter is implemented.",
    kind: "local_runtime",
    status: "planned",
  },
];
