use sea_orm::{ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter};

use crate::core::{
    entities::{artifact_types, artifacts, model_invocation_profiles, runs},
    output_locations::resolve_output_location,
};

use super::artifacts::{self as artifact_service, RecordArtifactByTypeSlugInput};

/// Registers existing OpenFold demo output directories and returns the full current
/// artifact list for the run. Repeated calls do not add duplicate type/URI entries.
pub async fn register_known_openfold_artifacts(
    db: &DatabaseConnection,
    run_id: i32,
) -> Result<Vec<artifacts::Model>, DbErr> {
    let run = runs::Entity::find_by_id(run_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("run {run_id} does not exist")))?;
    let profile = model_invocation_profiles::Entity::find_by_id(run.invocation_profile_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("model invocation profile does not exist".into()))?;
    let workspace = resolve_output_location(&profile, &run)?;

    register_directory_if_present(db, run_id, "run_output_directory", &workspace).await?;
    register_directory_if_present(
        db,
        run_id,
        "attention_output_directory",
        &workspace.join("attention"),
    )
    .await?;

    artifact_service::list_artifacts_for_run(db, run_id).await
}

async fn register_directory_if_present(
    db: &DatabaseConnection,
    run_id: i32,
    artifact_type_slug: &str,
    path: &std::path::Path,
) -> Result<(), DbErr> {
    if !path.is_dir() {
        return Ok(());
    }

    let artifact_type = artifact_types::Entity::find()
        .filter(artifact_types::Column::Slug.eq(artifact_type_slug))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("artifact type '{artifact_type_slug}' is missing")))?;
    let storage_uri = path.display().to_string();
    let already_registered = artifacts::Entity::find()
        .filter(artifacts::Column::RunId.eq(run_id))
        .filter(artifacts::Column::ArtifactTypeId.eq(artifact_type.id))
        .filter(artifacts::Column::StorageUri.eq(&storage_uri))
        .one(db)
        .await?
        .is_some();

    if !already_registered {
        artifact_service::record_artifact_manifest_entry_by_type_slug(
            db,
            RecordArtifactByTypeSlugInput {
                run_id,
                artifact_type_slug: artifact_type_slug.into(),
                format: "directory".into(),
                storage_uri,
                metadata_json: "{}".into(),
            },
        )
        .await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use sea_orm::{ConnectionTrait, Database, DatabaseConnection, DbErr, Statement};
    use serde_json::json;

    use crate::core::{
        db, seed,
        services::{
            execution_targets::{self, RegisterExecutionTargetInput},
            model_backends::{self, RegisterModelBackendInput},
            model_invocation_profiles::{self, RegisterModelInvocationProfileInput},
            runs::{self, SubmitRunInput},
        },
    };

    use super::register_known_openfold_artifacts;

    async fn test_db() -> Result<DatabaseConnection, DbErr> {
        let db = Database::connect("sqlite::memory:").await?;
        db.execute(Statement::from_string(
            db.get_database_backend(),
            "PRAGMA foreign_keys = ON".to_owned(),
        ))
        .await?;
        db::migrate_database(&db).await?;
        seed::seed_defaults(&db).await?;
        Ok(db)
    }

    async fn run_with_output_root(
        db: &DatabaseConnection,
        output_root: &PathBuf,
    ) -> Result<crate::core::entities::runs::Model, DbErr> {
        let backend = model_backends::register_model_backend(
            db,
            RegisterModelBackendInput {
                slug: "openfold-artifact-test".into(),
                label: "OpenFold".into(),
                version: None,
                description: None,
                artifact_capabilities_json: "{}".into(),
                parameter_schema_json: json!({"type":"object","properties":{}}).to_string(),
            },
        )
        .await?;
        let target = execution_targets::register_execution_target(
            db,
            RegisterExecutionTargetInput {
                slug: "local-artifact-test".into(),
                target_type: "local".into(),
                description: None,
                available_resources_json: json!({"type":"object","properties":{}}).to_string(),
            },
        )
        .await?;
        let profile = model_invocation_profiles::register_model_invocation_profile(
            db,
            RegisterModelInvocationProfileInput {
                model_backend_id: backend.id,
                execution_target_id: target.id,
                invocation_kind: "local_subprocess".into(),
                config_json: json!({"output_location": output_root}).to_string(),
            },
        )
        .await?;
        runs::submit_run(
            db,
            SubmitRunInput {
                model_backend_id: backend.id,
                execution_target_id: target.id,
                invocation_profile_id: profile.id,
                status: "completed".into(),
                input_id: "test_input".into(),
                input_sequence: "MSTNPKPQRITF".into(),
                model_parameters_json: "{}".into(),
                execution_parameters_json: "{}".into(),
                provenance_json: None,
            },
        )
        .await
    }

    fn temp_root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "executor-openfold-artifacts-{}-{}",
            std::process::id(),
            chrono::Utc::now()
                .timestamp_nanos_opt()
                .expect("timestamp is representable")
        ))
    }

    #[tokio::test]
    async fn missing_run_returns_clear_error() -> Result<(), DbErr> {
        let db = test_db().await?;
        let error = register_known_openfold_artifacts(&db, 999)
            .await
            .expect_err("missing run should fail");
        assert!(error.to_string().contains("run 999 does not exist"));
        Ok(())
    }

    #[tokio::test]
    async fn registers_existing_workspace_and_attention_directories_idempotently()
    -> Result<(), DbErr> {
        let db = test_db().await?;
        let root = temp_root();
        let run = run_with_output_root(&db, &root).await?;
        let workspace = root.join(run.id.to_string());
        fs::create_dir_all(workspace.join("attention")).expect("output directories should exist");

        let first = register_known_openfold_artifacts(&db, run.id).await?;
        let second = register_known_openfold_artifacts(&db, run.id).await?;

        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 2);
        assert!(
            first
                .iter()
                .any(|artifact| artifact.storage_uri == workspace.display().to_string())
        );
        assert!(
            first.iter().any(|artifact| artifact.storage_uri
                == workspace.join("attention").display().to_string())
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn skips_missing_output_directories() -> Result<(), DbErr> {
        let db = test_db().await?;
        let root = temp_root();
        let run = run_with_output_root(&db, &root).await?;

        let artifacts = register_known_openfold_artifacts(&db, run.id).await?;

        assert!(artifacts.is_empty());
        Ok(())
    }
}
