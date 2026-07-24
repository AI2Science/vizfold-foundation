#![cfg(test)]

use sea_orm::{ConnectionTrait, Database, DatabaseBackend, DatabaseConnection, DbErr, Statement};
use serde_json::json;

use crate::core::{
    config, db, seed,
    services::{
        artifact_types,
        execution_targets::{self, RegisterExecutionTargetInput},
        model_backends::{self, RegisterModelBackendInput},
        model_invocation_profiles::{self, RegisterModelInvocationProfileInput},
        runs::{self, SubmitRunInput},
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

#[tokio::test]
async fn seeds_artifact_type_catalog() -> Result<(), DbErr> {
    let db = test_db().await?;

    seed::seed_defaults(&db).await?;
    seed::seed_defaults(&db).await?;

    let artifact_types = artifact_types::list_artifact_types(&db).await?;
    assert_eq!(artifact_types.len(), 13);
    let protein_structure = artifact_types::get_artifact_type_by_slug(&db, "protein_structure")
        .await?
        .expect("protein structure type should be seeded");
    assert_eq!(protein_structure.default_format, "pdb");
    assert_eq!(protein_structure.viewer_kind, "ngl_viewer");
    Ok(())
}

#[tokio::test]
async fn seeds_local_openfold_target_and_profile() -> Result<(), DbErr> {
    let db = test_db().await?;

    seed::seed_defaults(&db).await?;
    seed::seed_defaults(&db).await?;

    let targets = execution_targets::list_execution_targets(&db).await?;
    let openfold_target = targets
        .iter()
        .find(|target| target.slug == "local-openfold")
        .expect("local OpenFold target should be seeded");
    assert_eq!(openfold_target.target_type, "local");
    assert_eq!(
        openfold_target.description.as_deref(),
        Some("Local OpenFold subprocess execution target for demo/development.")
    );

    let backend = model_backends::list_model_backends(&db)
        .await?
        .into_iter()
        .find(|backend| backend.slug == "openfold")
        .expect("OpenFold backend should be seeded");
    let profile = model_invocation_profiles::list_model_invocation_profiles(&db)
        .await?
        .into_iter()
        .find(|profile| {
            profile.model_backend_id == backend.id
                && profile.execution_target_id == openfold_target.id
        })
        .expect("local OpenFold profile should be seeded");
    assert_eq!(profile.invocation_kind, "local_subprocess");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&profile.config_json)
            .map_err(|error| DbErr::Custom(error.to_string()))?,
        json!({
            "program": "python3",
            "script": "run_pretrained_openfold.py",
            "working_dir": config::openfold_home(),
            "output_location": config::prefix().join("runs"),
        })
    );
    Ok(())
}

#[tokio::test]
async fn seeds_local_esmfold_target_and_profile() -> Result<(), DbErr> {
    let db = test_db().await?;

    seed::seed_defaults(&db).await?;
    seed::seed_defaults(&db).await?;

    let backend = model_backends::list_model_backends(&db)
        .await?
        .into_iter()
        .find(|backend| backend.slug == "esmfold")
        .expect("ESMFold backend should be seeded");
    let target = execution_targets::list_execution_targets(&db)
        .await?
        .into_iter()
        .find(|target| target.slug == "local-esmfold")
        .expect("local ESMFold target should be seeded");
    let profile = model_invocation_profiles::list_model_invocation_profiles(&db)
        .await?
        .into_iter()
        .find(|profile| {
            profile.model_backend_id == backend.id && profile.execution_target_id == target.id
        })
        .expect("local ESMFold profile should be seeded");

    assert_eq!(profile.invocation_kind, "local_subprocess");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&profile.config_json)
            .map_err(|error| DbErr::Custom(error.to_string()))?,
        json!({
            "program": "python3",
            "script": "run_pretrained_esmf.py",
            "working_dir": config::openfold_home(),
            "output_location": config::prefix().join("runs"),
        })
    );
    Ok(())
}

fn sample_execution_target_input() -> RegisterExecutionTargetInput {
    RegisterExecutionTargetInput {
        slug: "test-target".into(),
        target_type: "local".into(),
        description: Some("Test execution target".into()),
        available_resources_json: json!({
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
        invocation_kind: "test".into(),
        config_json: json!({"mode": "test"}).to_string(),
    }
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
            provenance_json: None,
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
            provenance_json: None,
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
            available_resources_json: json!({"type": "object", "properties": {}}).to_string(),
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
            provenance_json: None,
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
            provenance_json: None,
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
async fn baseline_schema_creates_every_table_including_provenance() -> Result<(), DbErr> {
    let db = test_db().await?;

    let tables: Vec<String> = db
        .query_all(Statement::from_string(
            DatabaseBackend::Sqlite,
            "select name from sqlite_master where type='table' order by name".to_owned(),
        ))
        .await?
        .iter()
        .map(|row| row.try_get::<String>("", "name").expect("name"))
        .collect();

    for expected in [
        "artifact_types",
        "artifacts",
        "execution_targets",
        "model_backends",
        "model_invocation_profiles",
        "runs",
    ] {
        assert!(
            tables.iter().any(|t| t == expected),
            "missing table {expected}"
        );
    }

    let columns: Vec<String> = db
        .query_all(Statement::from_string(
            DatabaseBackend::Sqlite,
            "select name from pragma_table_info('runs')".to_owned(),
        ))
        .await?
        .iter()
        .map(|row| row.try_get::<String>("", "name").expect("name"))
        .collect();

    assert!(columns.iter().any(|c| c == "provenance_json"));
    Ok(())
}
