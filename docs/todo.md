## Model / Execution Target Compatibility

A profile exists only for a supported backend/target pair (cli/src/core/seed.rs), and `submit_run` rejects a run whose profile does not match its backend and target (cli/src/core/services/runs.rs).

Still missing: a status and notes per pair, so an unsupported combination can be listed and explained instead of merely having no profile.


## Current JSON validation approach

Artifact capabilities, parameter schemas, target resources, profile config, and run parameters are stored as JSON strings in SQLite. The validation helpers live in cli/src/core/services/validation.rs and are called from both the services and the model runners.

The CLI adapter writes only through the core services and touches entities directly only for reads (cli/src/adapters/cli.rs:1470, 1481, 1732, 1737). Still open: move JSON parsing to the adapter boundary so the core receives `serde_json::Value`.
