## Model / Execution Target Compatibility

`MODEL_INVOCATION_PROFILE` already is the compatibility link: a profile exists only for a supported backend/target pair, `submit_run` rejects a run whose profile does not match the selected pair (cli/src/core/services/runs.rs), and `queue <backend>` resolves the target and profile from the backend rather than offering them independently. What is still missing is compatibility *metadata* — a status and notes per pair — so an unsupported combination can be listed and explained instead of merely having no profile.


## Current JSON validation approach

For the MVP, model capabilities, artifact capabilities, parameter schemas, and selected run parameters are stored as JSON strings in SQLite and validated in the Rust core service layer.

The CLI adapter already writes only through the core services (`services::runs::submit_run` and friends) and touches entities directly only for reads. Still open: once CLI/API interfaces are formalized, JSON parsing can move to those adapter boundaries so the core receives structured `serde_json::Value` inputs.