#![cfg(test)]

use sea_orm::{ConnectionTrait, Database, DatabaseConnection, DbErr, Statement};
use serde_json::json;

use crate::core::{
    db,
    services::{
        artifacts::{self, RecordArtifactInput},
        execution_targets::{self, RegisterExecutionTargetInput},
        model_backends::{self, RegisterModelBackendInput},
        model_invocation_profiles::{self, RegisterModelInvocationProfileInput},
        runs::{self, SubmitRunInput, UpdateRunStatusInput},
    },
};

async fn test_db() -> Result<DatabaseConnection, DbErr> {
    let db = Database::connect("sqlite::memory:").await?;
    db.execute(Statement::from_string(
        db.get_database_backend(),
        "PRAGMA foreign_keys = ON".to_owned(),
    ))
    .await?;
    db::migrate_database(&db).await?;
    Ok(db)
}

fn sample_model_backend_input() -> RegisterModelBackendInput {
    RegisterModelBackendInput {
        slug: "openfold".into(),
        label: "OpenFold".into(),
        version: Some("1.0".into()),
        description: Some("Reference folding backend".into()),
        artifact_capabilities_json: json!({
            "structure": { "formats": ["pdb", "cif"], "required": true },
            "confidence_metrics": { "formats": ["json"], "required": false }
        })
        .to_string(),
        parameter_schema_json: json!({
            "type": "object",
            "properties": {
                "num_recycles": { "type": "integer", "minimum": 0, "default": 3 }
            }
        })
        .to_string(),
    }
}

fn sample_execution_target_input() -> RegisterExecutionTargetInput {
    RegisterExecutionTargetInput {
        slug: "local-mock".into(),
        target_type: "local".into(),
        description: Some("Test execution target".into()),
        parameter_schema_json: json!({
            "type": "object",
            "properties": {
                "gpu_count": { "type": "integer", "minimum": 0, "default": 0 },
                "walltime": { "type": "string", "default": "00:05:00" }
            }
        })
        .to_string(),
    }
}

fn sample_invocation_profile_input(
    model_backend_id: i32,
    execution_target_id: i32,
) -> RegisterModelInvocationProfileInput {
    RegisterModelInvocationProfileInput {
        model_backend_id,
        execution_target_id,
        invocation_kind: "mock".into(),
        config_json: json!({"mode": "test"}).to_string(),
        parameter_schema_json: json!({"type": "object", "properties": {}}).to_string(),
    }
}

#[tokio::test]
async fn creates_model_backend_without_capabilities_json() -> Result<(), DbErr> {
    let db = test_db().await?;

    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;

    assert_eq!(backend.slug, "openfold");
    assert!(
        serde_json::from_str::<serde_json::Value>(&backend.artifact_capabilities_json)
            .expect("artifact_capabilities_json should parse")
            .is_object()
    );
    Ok(())
}

#[tokio::test]
async fn creates_execution_target() -> Result<(), DbErr> {
    let db = test_db().await?;

    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;

    assert_eq!(target.slug, "local-mock");
    assert_eq!(target.target_type, "local");
    Ok(())
}

#[tokio::test]
async fn creates_model_invocation_profile() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;

    let profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;

    assert_eq!(profile.model_backend_id, backend.id);
    assert_eq!(profile.execution_target_id, target.id);
    assert_eq!(profile.invocation_kind, "mock");
    Ok(())
}

#[tokio::test]
async fn lists_model_invocation_profiles() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let _profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;

    let profiles = model_invocation_profiles::list_model_invocation_profiles(&db).await?;

    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].model_backend_id, backend.id);
    Ok(())
}

#[tokio::test]
async fn creates_run_with_separate_parameter_sets() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;

    let run = runs::submit_run(
        &db,
        SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "submitted".into(),
            input_id: "1UBQ_1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: json!({"num_recycles": 5}).to_string(),
            execution_parameters_json: json!({"gpu_count": 1, "walltime": "02:00:00"}).to_string(),
        },
    )
    .await?;

    assert_eq!(run.model_backend_id, backend.id);
    assert_eq!(run.execution_target_id, target.id);
    assert_eq!(run.invocation_profile_id, profile.id);
    assert_eq!(run.input_id, "1UBQ_1");
    assert!(run.started_at.is_none());
    Ok(())
}

#[tokio::test]
async fn rejects_run_with_empty_input_id() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;

    let error = runs::submit_run(
        &db,
        SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "submitted".into(),
            input_id: "   ".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: json!({"num_recycles": 2}).to_string(),
            execution_parameters_json: json!({"gpu_count": 0}).to_string(),
        },
    )
    .await
    .expect_err("empty input_id should fail");

    assert!(error.to_string().contains("input_id must be non-empty"));
    Ok(())
}

#[tokio::test]
async fn rejects_run_with_empty_input_sequence() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;

    let error = runs::submit_run(
        &db,
        SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "submitted".into(),
            input_id: "1UBQ_1".into(),
            input_sequence: "   ".into(),
            model_parameters_json: json!({"num_recycles": 2}).to_string(),
            execution_parameters_json: json!({"gpu_count": 0}).to_string(),
        },
    )
    .await
    .expect_err("empty input_sequence should fail");

    assert!(
        error
            .to_string()
            .contains("input_sequence must be non-empty")
    );
    Ok(())
}

#[tokio::test]
async fn rejects_run_with_mismatched_invocation_profile() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let other_target = execution_targets::register_execution_target(
        &db,
        RegisterExecutionTargetInput {
            slug: "docker-local".into(),
            target_type: "docker".into(),
            description: Some("Other target".into()),
            parameter_schema_json: json!({"type": "object", "properties": {}}).to_string(),
        },
    )
    .await?;
    let mismatched_profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, other_target.id),
    )
    .await?;

    let error = runs::submit_run(
        &db,
        SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: mismatched_profile.id,
            status: "submitted".into(),
            input_id: "1UBQ_1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: json!({"num_recycles": 5}).to_string(),
            execution_parameters_json: json!({"gpu_count": 1}).to_string(),
        },
    )
    .await
    .expect_err("mismatched invocation profile should fail");

    assert!(
        error
            .to_string()
            .contains("model invocation profile does not match")
    );
    Ok(())
}

#[tokio::test]
async fn rejects_non_object_json_parameters() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;

    let error = runs::submit_run(
        &db,
        SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "submitted".into(),
            input_id: "1UBQ_1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: "[]".into(),
            execution_parameters_json: json!({"gpu_count": 1}).to_string(),
        },
    )
    .await
    .expect_err("non-object model parameters should fail");

    assert!(
        error
            .to_string()
            .contains("model_parameters must be a JSON object")
    );
    Ok(())
}

#[tokio::test]
async fn records_artifact_manifest_entry() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;
    let run = runs::submit_run(
        &db,
        SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "submitted".into(),
            input_id: "1UBQ_1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: json!({"num_recycles": 2}).to_string(),
            execution_parameters_json: json!({"gpu_count": 0}).to_string(),
        },
    )
    .await?;

    let artifact = artifacts::record_artifact_manifest_entry(
        &db,
        RecordArtifactInput {
            run_id: run.id,
            artifact_type: "structure".into(),
            format: "pdb".into(),
            storage_uri: "file:///tmp/run-1/model.pdb".into(),
            metadata_json: json!({"bytes": 1280, "sha256": "abc123"}).to_string(),
        },
    )
    .await?;

    assert_eq!(artifact.storage_uri, "file:///tmp/run-1/model.pdb");
    assert!(artifact.metadata_json.contains("sha256"));
    Ok(())
}

#[tokio::test]
async fn retrieves_run_with_artifacts() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;
    let run = runs::submit_run(
        &db,
        SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "submitted".into(),
            input_id: "1UBQ_1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: json!({"num_recycles": 2}).to_string(),
            execution_parameters_json: json!({"gpu_count": 0}).to_string(),
        },
    )
    .await?;

    let _artifact = artifacts::record_artifact_manifest_entry(
        &db,
        RecordArtifactInput {
            run_id: run.id,
            artifact_type: "logs".into(),
            format: "txt".into(),
            storage_uri: "file:///tmp/run-1/stdout.log".into(),
            metadata_json: json!({"line_count": 42}).to_string(),
        },
    )
    .await?;

    let hydrated = runs::get_run_with_artifacts(&db, run.id)
        .await?
        .expect("run should exist");

    assert_eq!(hydrated.run.id, run.id);
    assert_eq!(hydrated.artifacts.len(), 1);
    Ok(())
}

#[tokio::test]
async fn artifact_manifest_stores_uri_and_metadata_only() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;
    let run = runs::submit_run(
        &db,
        SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "submitted".into(),
            input_id: "1UBQ_1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: json!({"num_recycles": 2}).to_string(),
            execution_parameters_json: json!({"gpu_count": 0}).to_string(),
        },
    )
    .await?;

    let artifact = artifacts::record_artifact_manifest_entry(
        &db,
        RecordArtifactInput {
            run_id: run.id,
            artifact_type: "confidence_metrics".into(),
            format: "json".into(),
            storage_uri: "s3://vizfold/runs/1/confidence.json".into(),
            metadata_json: json!({"bytes": 256, "content_type": "application/json"}).to_string(),
        },
    )
    .await?;

    assert_eq!(artifact.format, "json");
    assert!(!artifact.metadata_json.contains("ATOM"));
    assert!(!artifact.storage_uri.starts_with("{"));
    Ok(())
}

#[tokio::test]
async fn updates_run_status() -> Result<(), DbErr> {
    let db = test_db().await?;
    let backend = model_backends::register_model_backend(&db, sample_model_backend_input()).await?;
    let target =
        execution_targets::register_execution_target(&db, sample_execution_target_input()).await?;
    let profile = model_invocation_profiles::register_model_invocation_profile(
        &db,
        sample_invocation_profile_input(backend.id, target.id),
    )
    .await?;
    let run = runs::submit_run(
        &db,
        SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "submitted".into(),
            input_id: "1UBQ_1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: json!({"num_recycles": 2}).to_string(),
            execution_parameters_json: json!({"gpu_count": 0}).to_string(),
        },
    )
    .await?;

    let updated = runs::update_run_status(
        &db,
        run.id,
        UpdateRunStatusInput {
            status: "completed".into(),
            completed_at: Some(Some(chrono::Utc::now())),
            ..Default::default()
        },
    )
    .await?;

    assert_eq!(updated.status, "completed");
    assert!(updated.completed_at.is_some());
    Ok(())
}
