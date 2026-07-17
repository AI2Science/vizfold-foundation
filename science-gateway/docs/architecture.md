# Science Gateway Architecture

![Science Gateway Architecture](img/VizfoldGateway-1.png)

# VizFold Executor MVP Data Model

![Science Gateway Metadata Model](img/ERModel.png)

This diagram describes the MVP data model for the Rust executor core. The goal is to separate model definition, execution environment, invocation configuration, concrete runs, and produced artifacts.

`MODEL_BACKEND` represents a registered model implementation, such as OpenFold, ESMFold, or Boltz. It stores model-level metadata, the model parameter schema, and the artifact types the model can theoretically produce.

`EXECUTION_TARGET` represents an environment where execution can happen, such as local runtime, Docker, HPC, or a science gateway. It stores target-level metadata and execution parameter schema, but does not store model-specific installation details.

`MODEL_INVOCATION_PROFILE` connects a specific model backend to a specific execution target. It owns the model-target-specific invocation configuration, such as subprocess, Docker, SLURM, or gateway invocation details. This prevents model-specific paths or command templates from leaking into the generic execution target definition.

`RUN` represents one concrete execution request. It selects a model backend, execution target, and invocation profile, then stores the selected model and execution parameters for that specific run. `RUN.input_id` is the stable model-facing input identity for the run. For OpenFold, this is intended to become the canonical FASTA record ID/header tag, default alignment key, and artifact filename prefix. It should not be mutated for workspace/output collision handling; use run/workspace identifiers for that instead.

`ARTIFACT` represents a manifest entry for an output produced by a run. The database records what artifact exists and where it is stored, while the heavy scientific output files remain in external storage such as the filesystem, HPC storage, or object storage.

This model intentionally does not include model-target artifact constraint logic in the MVP. Artifact capabilities remain model-level, and actual produced outputs are recorded through the artifact manifest.

## Architecture note: parameter schema ownership

The current MVP has three places that can participate in command planning:

- `ModelBackend.parameter_schema_json`
- `ExecutionTarget.parameter_schema_json`
- `ModelInvocationProfile.parameter_schema_json`

This has created some ambiguity because `Run` currently has only two concrete parameter buckets:

- `Run.model_parameters_json`
- `Run.execution_parameters_json`

The intended future direction is to avoid treating all three schema locations as competing sources of command parameters.

A cleaner model may be:

1. `ModelBackend` owns the canonical model parameter contract. It defines the parameters a backend supports, their meaning, and where their values should be sourced from.

2. `ModelInvocationProfile` provides target-specific invocation context and defaults, such as local paths, remote paths, script/program configuration, or target-specific output locations.

3. `ExecutionTarget` describes runtime capabilities/resources rather than model-specific command parameters. For example, GPU availability, CPU limits, supported execution type, or resource constraints.

Under this direction, model-specific CLI flags such as OpenFold `--attn_map_dir` should not live on a generic execution target. Likewise, concrete run values such as `output_dir` and `attn_map_dir` should remain part of the run’s selected execution parameters, while the model/backend contract defines that those values are needed and how they are interpreted.

This also suggests that `ExecutionTarget.parameter_schema_json` may eventually need to be renamed to something like `runtime_capabilities_json` or `resource_schema_json`, because it is not meant to become a model-specific command parameter schema.

This is deferred for now. The current MVP keeps the existing structure while the OpenFold workflow example validates the executor flow end-to-end.

## Executor Architecture Flow

![Science Gateway Metadata Model](img/ExecutionFlow.png)

The executor separates registration, planning, optional preflight, execution, and artifact recording. `MODEL_BACKEND` defines what model exists, `EXECUTION_TARGET` defines where execution can happen, and `MODEL_INVOCATION_PROFILE` defines how a specific model runs on a specific target.

For a concrete `RUN`, the executor loads the selected model, target, invocation profile, and parameters. A planner then converts those records into a `CommandSpec`, which is the final resolved execution plan containing the program, arguments, working directory, and environment variables.

Before execution, the command may pass through an optional `PreflightRunner`. A preflight runner performs model-specific readiness checks for the selected execution target environment and returns a `PreflightReport` with passed checks, warnings, or failures. If no preflight runner is available, the workflow can proceed directly to execution. If preflight failures are reported, execution is skipped and the workflow returns the report.

The `ExecutionWorkflow` coordinates this flow: `CommandSpec` → optional `PreflightRunner` → `CommandRunner`. The `CommandRunner` executes the command and returns a `CommandOutput` containing the exit code, stdout, and stderr.

For the MVP, OpenFold can be supported through a built-in Rust planner and an optional OpenFold preflight runner. Later, the same abstractions can support DB-driven command templates, external model plugins, richer preflight checks, and additional execution targets without changing the core execution flow. Produced outputs are not stored directly in the database; they remain in external storage and are registered as `ARTIFACT` manifest entries.

### Output location resolution

The executor should maintain a normalized output location concept that is independent of model-native argument names.

The preferred future pattern is for `ModelInvocationProfile` to define a base `output_location` for a specific backend-target invocation. A concrete run then resolves its output workspace using the run identifier:

```text
resolved_output_location = invocation_profile.output_location / run.id
```
Model-specific command planning maps this resolved location to the backend-native argument. For OpenFold, the resolved output location maps to --output_dir.

Model-specific secondary output paths, such as OpenFold attention map directories, should be derived from the resolved run output location where possible rather than supplied as unrelated top-level paths.

This keeps:

- ModelInvocationProfile responsible for target-specific location conventions;
- Run responsible for the concrete run identity and selected parameters;
- artifact registration responsible for indexing produced files under the resolved output location.