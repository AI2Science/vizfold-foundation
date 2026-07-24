use std::path::PathBuf;

use sea_orm::DbErr;
use serde_json::Value;

use crate::core::entities::{model_invocation_profiles, runs};

/// Resolves the workspace where a run's outputs are stored.
pub fn resolve_output_location(
    invocation_profile: &model_invocation_profiles::Model,
    run: &runs::Model,
) -> Result<PathBuf, DbErr> {
    let output_location = output_location_from(
        run.provenance_json.as_deref(),
        &invocation_profile.config_json,
    )?;
    Ok(PathBuf::from(output_location).join(run.id.to_string()))
}

/// Prefer the run's immutable snapshot; fall back to the live profile for runs queued before
/// snapshots existed.
pub fn output_location_from(
    provenance_json: Option<&str>,
    profile_config_json: &str,
) -> Result<String, DbErr> {
    if let Some(raw) = provenance_json
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(raw)
        && let Some(location) = value["profile"]["config"]["output_location"].as_str()
    {
        return Ok(location.to_owned());
    }

    let config: Value = serde_json::from_str(profile_config_json).map_err(|error| {
        DbErr::Custom(format!(
            "model invocation profile config_json must be valid JSON: {error}"
        ))
    })?;

    if !config.is_object() {
        return Err(DbErr::Custom(
            "model invocation profile config_json must be a JSON object".into(),
        ));
    }

    let output_location = config
        .get("output_location")
        .ok_or_else(|| DbErr::Custom("output_location is required".into()))?
        .as_str()
        .ok_or_else(|| DbErr::Custom("output_location must be a string".into()))?;

    if output_location.trim().is_empty() {
        return Err(DbErr::Custom("output_location must be non-empty".into()));
    }

    Ok(output_location.to_owned())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use chrono::Utc;
    use serde_json::json;

    use super::resolve_output_location;
    use crate::core::entities::{model_invocation_profiles, runs};

    fn invocation_profile(config_json: &str) -> model_invocation_profiles::Model {
        let now = Utc::now();
        model_invocation_profiles::Model {
            id: 1,
            model_backend_id: 2,
            execution_target_id: 3,
            invocation_kind: "mock".into(),
            config_json: config_json.into(),
            created_at: now,
            updated_at: now,
        }
    }

    fn run() -> runs::Model {
        runs::Model {
            id: 42,
            model_backend_id: 2,
            execution_target_id: 3,
            invocation_profile_id: 1,
            status: "submitted".into(),
            input_id: "input-1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: json!({}).to_string(),
            execution_parameters_json: json!({}).to_string(),
            provenance_json: None,
            submitted_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error_message: None,
        }
    }

    #[test]
    fn resolves_output_location_with_run_id() {
        let output_location = resolve_output_location(
            &invocation_profile(r#"{"output_location":"/work/outputs"}"#),
            &run(),
        )
        .expect("output location should resolve");

        assert_eq!(output_location, PathBuf::from("/work/outputs").join("42"));
    }

    #[test]
    fn rejects_missing_output_location() {
        let error = resolve_output_location(&invocation_profile("{}"), &run())
            .expect_err("missing output location should fail");

        assert!(error.to_string().contains("output_location is required"));
    }

    #[test]
    fn rejects_empty_output_location() {
        let error =
            resolve_output_location(&invocation_profile(r#"{"output_location":"   "}"#), &run())
                .expect_err("empty output location should fail");

        assert!(
            error
                .to_string()
                .contains("output_location must be non-empty")
        );
    }

    #[test]
    fn rejects_non_string_output_location() {
        let error =
            resolve_output_location(&invocation_profile(r#"{"output_location":123}"#), &run())
                .expect_err("non-string output location should fail");

        assert!(
            error
                .to_string()
                .contains("output_location must be a string")
        );
    }

    #[test]
    fn provenance_snapshot_wins_over_a_mutated_profile() {
        let profile_config = r#"{"output_location":"/work/relocated"}"#;
        let snapshot =
            Some(r#"{"profile":{"config":{"output_location":"/work/original"}}}"#.to_owned());

        let resolved =
            super::output_location_from(snapshot.as_deref(), profile_config).expect("resolved");

        assert_eq!(resolved, "/work/original");
    }

    #[test]
    fn falls_back_to_the_profile_when_a_run_has_no_snapshot() {
        let resolved = super::output_location_from(None, r#"{"output_location":"/work/live"}"#)
            .expect("resolved");
        assert_eq!(resolved, "/work/live");
    }

    #[test]
    fn rejects_invalid_or_non_object_config_json() {
        let invalid_error = resolve_output_location(&invocation_profile("not-json"), &run())
            .expect_err("invalid config JSON should fail");
        let non_object_error = resolve_output_location(&invocation_profile("[]"), &run())
            .expect_err("non-object config JSON should fail");

        assert!(
            invalid_error
                .to_string()
                .contains("model invocation profile config_json must be valid JSON")
        );
        assert!(
            non_object_error
                .to_string()
                .contains("model invocation profile config_json must be a JSON object")
        );
    }
}
