## Model / Execution Target Compatibility

The current prototype treats model backends and execution targets as independently selectable. In a production version, not all model backends will be compatible with all execution targets. This should be represented with a compatibility table, likely `ModelExecutionTarget`, including status, notes, and target-specific configuration.


## Current JSON validation approach

For the MVP, model capabilities, artifact capabilities, parameter schemas, and selected run parameters are stored as JSON strings in SQLite and validated in the Rust core service layer.

Invocation layers should call the core services rather than writing through repositories directly. Once CLI/API interfaces are formalized, JSON parsing can move closer to those adapter boundaries while the core receives structured `serde_json::Value` inputs.