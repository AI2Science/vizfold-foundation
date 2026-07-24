use std::path::Path;

use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, DbErr, EntityTrait,
    QueryFilter,
};

use crate::core::entities::{
    artifacts, execution_targets, model_backends, model_invocation_profiles, runs,
};

use super::validation::{reject_unknown_keys, require_json_object};

#[derive(Clone, Debug)]
pub struct SubmitRunInput {
    pub model_backend_id: i32,
    pub execution_target_id: i32,
    pub invocation_profile_id: i32,
    pub status: String,
    pub input_id: String,
    pub input_sequence: String,
    pub model_parameters_json: String,
    pub execution_parameters_json: String,
    pub provenance_json: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct UpdateRunStatusInput {
    pub status: String,
    pub started_at: Option<Option<chrono::DateTime<Utc>>>,
    pub completed_at: Option<Option<chrono::DateTime<Utc>>>,
    pub error_message: Option<Option<String>>,
}

#[derive(Clone, Debug)]
pub struct RunWithArtifacts {
    pub run: runs::Model,
    pub artifacts: Vec<artifacts::Model>,
}

pub async fn list_runs(db: &DatabaseConnection) -> Result<Vec<runs::Model>, DbErr> {
    runs::Entity::find().all(db).await
}

/// Immutable record of what produced a run. Catalog rows can be edited later; this cannot.
/// Takes resolved paths as parameters rather than resolving them itself, matching the
/// `gpu_launch`/`gpu_launch_args` split -- callers resolve from the environment.
#[allow(clippy::too_many_arguments)]
pub fn provenance_snapshot(
    backend_slug: &str,
    backend_version: Option<&str>,
    target_slug: &str,
    invocation_kind: &str,
    profile_config_json: &str,
    openfold_home: &Path,
    prefix: &Path,
    env_prefix: &Path,
) -> String {
    let config: serde_json::Value =
        serde_json::from_str(profile_config_json).unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "backend": { "slug": backend_slug, "version": backend_version },
        "target": { "slug": target_slug },
        "profile": { "invocation_kind": invocation_kind, "config": config },
        "resolved": {
            "openfold_home": openfold_home.display().to_string(),
            "prefix": prefix.display().to_string(),
            "env_prefix": env_prefix.display().to_string(),
        },
    })
    .to_string()
}

pub async fn submit_run(
    db: &DatabaseConnection,
    input: SubmitRunInput,
) -> Result<runs::Model, DbErr> {
    require_non_empty("input_id", &input.input_id)?;
    require_non_empty("input_sequence", &input.input_sequence)?;

    let backend = model_backends::Entity::find_by_id(input.model_backend_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("model backend does not exist".into()))?;
    let target = execution_targets::Entity::find_by_id(input.execution_target_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("execution target does not exist".into()))?;
    let profile = model_invocation_profiles::Entity::find_by_id(input.invocation_profile_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("model invocation profile does not exist".into()))?;

    if profile.model_backend_id != input.model_backend_id
        || profile.execution_target_id != input.execution_target_id
    {
        return Err(DbErr::Custom(
            "model invocation profile does not match selected model backend and execution target"
                .into(),
        ));
    }

    let model_schema = require_json_object(
        "model backend parameter_schema",
        &backend.parameter_schema_json,
    )?;
    let _available_resources = require_json_object(
        "execution target available_resources",
        &target.available_resources_json,
    )?;
    let model_params = require_json_object("model_parameters", &input.model_parameters_json)?;
    let _execution_params =
        require_json_object("execution_parameters", &input.execution_parameters_json)?;
    reject_unknown_keys("model_parameters", &model_schema, &model_params)?;

    // TODO: Execution parameter validation should eventually distinguish target
    // available resources/capabilities from concrete per-run execution values and
    // invocation-profile-specific requirements. For now, submit_run only requires
    // execution_parameters_json to be a JSON object; model-specific planning performs
    // additional validation where needed.

    runs::ActiveModel {
        model_backend_id: Set(input.model_backend_id),
        execution_target_id: Set(input.execution_target_id),
        invocation_profile_id: Set(input.invocation_profile_id),
        status: Set(input.status),
        input_id: Set(input.input_id),
        input_sequence: Set(input.input_sequence),
        model_parameters_json: Set(input.model_parameters_json),
        execution_parameters_json: Set(input.execution_parameters_json),
        provenance_json: Set(input.provenance_json),
        ..Default::default()
    }
    .insert(db)
    .await
}

fn require_non_empty(field_name: &str, value: &str) -> Result<(), DbErr> {
    if value.trim().is_empty() {
        return Err(DbErr::Custom(format!("{field_name} must be non-empty")));
    }

    Ok(())
}

pub async fn update_run_status(
    db: &DatabaseConnection,
    run_id: i32,
    update: UpdateRunStatusInput,
) -> Result<runs::Model, DbErr> {
    let model = runs::Entity::find_by_id(run_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("run does not exist".into()))?;

    let mut active_model: runs::ActiveModel = model.into();
    active_model.status = Set(update.status);

    if let Some(started_at) = update.started_at {
        active_model.started_at = Set(started_at);
    }

    if let Some(completed_at) = update.completed_at {
        active_model.completed_at = Set(completed_at);
    }

    if let Some(error_message) = update.error_message {
        active_model.error_message = Set(error_message);
    }

    active_model.update(db).await
}

pub async fn get_run_with_artifacts(
    db: &DatabaseConnection,
    run_id: i32,
) -> Result<Option<RunWithArtifacts>, DbErr> {
    let Some(run) = runs::Entity::find_by_id(run_id).one(db).await? else {
        return Ok(None);
    };

    let artifacts = artifacts::Entity::find()
        .filter(artifacts::Column::RunId.eq(run_id))
        .all(db)
        .await?;
    Ok(Some(RunWithArtifacts { run, artifacts }))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    #[test]
    fn snapshot_records_every_catalog_payload_and_the_resolved_paths() {
        let snapshot = super::provenance_snapshot(
            "openfold",
            Some("v2.1"),
            "local-openfold",
            "local_subprocess",
            r#"{"output_location":"/work/runs"}"#,
            Path::new("/opt/openfold"),
            Path::new("/opt/prefix"),
            Path::new("/opt/prefix/mamba/envs/openfold-env"),
        );
        let value: serde_json::Value = serde_json::from_str(&snapshot).expect("valid json");

        assert_eq!(value["backend"]["slug"], "openfold");
        assert_eq!(value["backend"]["version"], "v2.1");
        assert_eq!(value["target"]["slug"], "local-openfold");
        assert_eq!(value["profile"]["invocation_kind"], "local_subprocess");
        assert_eq!(value["profile"]["config"]["output_location"], "/work/runs");
        assert_eq!(value["resolved"]["openfold_home"], "/opt/openfold");
        assert_eq!(value["resolved"]["prefix"], "/opt/prefix");
        assert_eq!(
            value["resolved"]["env_prefix"],
            "/opt/prefix/mamba/envs/openfold-env"
        );
    }
}
