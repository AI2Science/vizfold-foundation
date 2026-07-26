# Science Gateway Architecture

![Science Gateway Architecture](VizfoldGateway-1.png)

# VizFold Executor MVP Data Model

![Science Gateway Metadata Model](ERModel.png)

This diagram describes the MVP data model for the Rust executor core. The goal is to separate model definition, execution environment, invocation configuration, concrete runs, artifact classification, and produced artifact instances.

`MODEL_BACKEND` is a registered model implementation. Two are seeded: `openfold` and `esmfold`. It owns model-level metadata, the artifact types the model can produce, and — in `parameter_schema_json` — the canonical contract for the model-native arguments the planner emits.

`EXECUTION_TARGET` is an environment where execution can happen, described by its resources and capabilities: supported devices, CPU bounds, resource constraints. Both seeded targets are `target_type: "local"`. HPC is not a separate target — it is the same local profile with an `srun` prefix wrapped on at execution time when a GPU partition is configured. The target stores no model-specific paths or installation details.

`MODEL_INVOCATION_PROFILE` connects a specific model backend to a specific execution target and owns the invocation configuration for that pair in `config_json`: `program`, `script`, `working_dir`, environment variables, and `output_location`. This keeps model-specific paths and command templates out of the generic execution target. `local_subprocess` is the only supported `invocation_kind`; the planner rejects any other value.

`RUN` is one concrete execution request. It selects a model backend, execution target, and invocation profile, then records the concrete model and runtime choices for that run, the sequence folded (`input_sequence`), and an immutable `provenance_json` snapshot of the backend/target/profile and resolved install paths as they were at queue time. `RUN.input_id` is the stable model-facing input identity. For OpenFold this is enforced: preflight fails a run whose FASTA header tag differs from `input_id`, and the precomputed-alignment key is `alignment_dir/<input_id>`. It should not be mutated for workspace/output collision handling; use run/workspace identifiers for that instead.

`ARTIFACT_TYPE` is catalog/reference data for known artifact kinds — protein structures, attention heatmaps, PyMOL sessions, trace archives, manifests. It stores a slug, default format, display mode, viewer kind, description, and an optional metadata schema, separating classification and visualization hints from produced instances.

`ARTIFACT` is a manifest entry for one concrete output of a run: its `ARTIFACT_TYPE`, the produced format, storage URI, and metadata. The database records what artifact exists and where it is stored; the heavy scientific output files remain in external storage. Artifact capabilities stay model-level — the MVP has no model-target artifact constraint logic.

Post-run artifact discovery runs inline: a run that exits 0 registers its output directories (`run_output_directory`, `attention_output_directory`) through `services::run_artifacts::register_known_run_artifacts`, idempotently, and `vizfold register-artifacts <id>` re-runs it. The workbench (`vizfold serve`) reads those rows, lists the registered directories, and renders each run's structure and attention images. Discovery below directory granularity — classifying individual files by artifact type — is still deferred.

## Architecture note: parameter and resource ownership

`Run` has two parameter buckets, resolved against `ModelBackend.parameter_schema_json` and `ExecutionTarget.available_resources_json`:

| Column | Intended meaning |
| --- | --- |
| `Run.model_parameters_json` | Explicit model-argument values or overrides selected for this run, such as an OpenFold preset or model feature flag. The allowed arguments, types, CLI flags, and defaults are defined by `ModelBackend.parameter_schema_json`; schema defaults may be applied when a value is omitted here. This column is not redundant with the schema: it preserves the run's chosen model values. |
| `Run.execution_parameters_json` | Run-scoped input, runtime, and resource choices consumed by schema entries sourced from `execution_parameters`, such as FASTA/alignment inputs, the current `data_dir`, device selection, or CPU count. It must not carry invocation-profile configuration or normalized output paths. |

Resolved output locations are derived from the run's `provenance_json` snapshot of `output_location` — falling back to the live `ModelInvocationProfile.config_json.output_location` for runs queued before snapshots existed — joined with `Run.id`, rather than being stored as `output_dir` or `attn_map_dir` execution parameters.

`ExecutionTarget.available_resources_json` both constrains the values in `Run.execution_parameters_json` and emits its own flags: `append_available_resources_args` appends `--model_device` and `--cpus` for the OpenFold target, `--device` for the ESMFold one. Model-native flags such as OpenFold's `--attn_map_dir` live in the backend's parameter schema, not on the target.

## Executor Architecture Flow

![Executor Architecture Flow](ExecutionFlow.png)

The executor separates registration, planning, preflight, execution, and artifact recording. For a concrete `RUN` it loads the selected model, target, invocation profile, and parameters, and a planner converts those records into a `CommandSpec` — the final resolved plan holding program, arguments, working directory, and environment variables.

Before execution the command always passes through a preflight, selected by the run's backend slug — `preflight_openfold` or `preflight_esmfold`, with unknown slugs falling back to OpenFold's. Each performs model-specific readiness checks against the planned (unwrapped) command and returns a `PreflightReport` of passed checks, warnings, or failures. If failures are reported, execution is skipped and the run is marked `failed` with the failure messages.

`services::run_execution::execute_run` coordinates this flow: create the run workspace → plan a `CommandSpec` → wrap it for the backend's environment (micromamba activation for OpenFold; for ESMFold its env's own `bin/python`, which needs no activate.d hook; `srun` outermost when a GPU partition is configured) → preflight the unwrapped command → `CommandRunner`. The runner executes the command and returns a `CommandOutput` with exit code, stdout, and stderr; `execute_run` moves the run through `running` → `completed`/`failed` and registers its artifacts on success.

One built-in schema-driven Rust planner serves every backend — it emits OpenFold's or ESMFold's CLI from that backend's `parameter_schema_json` — and each backend contributes its own preflight. The same abstractions can later support DB-driven command templates, external model plugins, richer preflight checks, and additional execution targets without changing the core execution flow. Produced outputs are never stored in the database; they remain in external storage and are registered as `ARTIFACT` manifest entries classified by the `ARTIFACT_TYPE` catalog.

### Schema parameter sources

`ModelBackend.parameter_schema_json` declares argument types, CLI flags, defaults, and where the planner obtains a value. A schema describes what a backend accepts; the selected run values remain in the `RUN` record for reproducibility.

The planner's source vocabulary, shared by every backend schema, is:

| Source | Resolution |
| --- | --- |
| model parameter (no `source`) | Read from `Run.model_parameters_json`, falling back to a schema default when present. |
| `execution_parameters` | Read the named `parameter` from `Run.execution_parameters_json`. |
| `data_dir` | Read `data_dir` from `Run.execution_parameters_json` and join the declaration's `relative_path`. |
| `invocation_profile_config` | Read the named `parameter` from `ModelInvocationProfile.config_json`. An optional `relative_path` is joined after resolution. |
| `run_output_workspace` | Resolve the run's output workspace — the provenance snapshot's `output_location`, else the live profile's, joined with `Run.id`. An optional `relative_path` is joined after resolution. |

For example, OpenFold's normalized output arguments are declared in the model schema rather than emitted as planner-specific special cases:

```json
{
  "output_dir": {
    "type": "path",
    "source": "run_output_workspace",
    "cli_flag": "--output_dir"
  },
  "attn_map_dir": {
    "type": "path",
    "source": "run_output_workspace",
    "relative_path": "attention",
    "cli_flag": "--attn_map_dir"
  }
}
```

`invocation_profile_config` enables direct profile-owned values such as a target-specific `data_dir`. It is available to schema declarations but is not yet used by the OpenFold parameter schema; current `data_dir` behavior remains unchanged.

### Output location resolution

`ModelInvocationProfile.config_json.output_location` is the base output location for a backend-target pair, and each run snapshots it into `provenance_json` at queue time so a later edit to the profile cannot move a finished run's outputs. The resolved workspace is that snapshot (else the live profile) joined with `Run.id`. OpenFold maps it to `--output_dir`, with `<workspace>/attention` for `--attn_map_dir`; secondary output paths should be derived this way rather than supplied as unrelated top-level paths.
