# Science Gateway Architecture

![Science Gateway Architecture](img/VizfoldGateway-1.png)

# VizFold Executor MVP Data Model

![Science Gateway Metadata Model](img/ERModel.png)

This diagram describes the MVP data model for the Rust executor core. The goal is to separate model definition, execution environment, invocation configuration, concrete runs, and produced artifacts.

`MODEL_BACKEND` represents a registered model implementation, such as OpenFold, ESMFold, or Boltz. It stores model-level metadata, the model parameter schema, and the artifact types the model can theoretically produce.

`EXECUTION_TARGET` represents an environment where execution can happen, such as local runtime, Docker, HPC, or a science gateway. It stores target-level metadata and execution parameter schema, but does not store model-specific installation details.

`MODEL_INVOCATION_PROFILE` connects a specific model backend to a specific execution target. It owns the model-target-specific invocation configuration, such as subprocess, Docker, SLURM, or gateway invocation details. This prevents model-specific paths or command templates from leaking into the generic execution target definition.

`RUN` represents one concrete execution request. It selects a model backend, execution target, and invocation profile, then stores the selected model and execution parameters for that specific run.

`ARTIFACT` represents a manifest entry for an output produced by a run. The database records what artifact exists and where it is stored, while the heavy scientific output files remain in external storage such as the filesystem, HPC storage, or object storage.

This model intentionally does not include model-target artifact constraint logic in the MVP. Artifact capabilities remain model-level, and actual produced outputs are recorded through the artifact manifest.

## Executor Architecture Flow

![Science Gateway Metadata Model](img/ExecutionFlow.png)

The executor separates registration, planning, execution, and artifact recording. `MODEL_BACKEND` defines what model exists, `EXECUTION_TARGET` defines where execution can happen, and `MODEL_INVOCATION_PROFILE` defines how a specific model runs on a specific target.

For a concrete `RUN`, the executor loads the selected model, target, invocation profile, and parameters. A planner then converts those records into a `CommandSpec`, which is the final resolved execution plan containing the program, arguments, working directory, and environment variables. The `CommandRunner` executes that command and returns a `CommandOutput`.

For the MVP, OpenFold can be supported through a built-in Rust planner. Later, the same abstraction can support DB-driven command templates or external model plugins without changing the core execution flow. Produced outputs are not stored directly in the database; they remain in external storage and are registered as `ARTIFACT` manifest entries.