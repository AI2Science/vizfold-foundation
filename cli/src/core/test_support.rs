#![cfg(test)]

use std::{
    env, fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use serde_json::json;

use sea_orm::{ConnectionTrait, Database, DatabaseConnection, DbErr, Statement};

use crate::core::{
    commands::CommandSpec,
    db,
    entities::{execution_targets, model_backends},
    preflight::{PreflightReport, PreflightStatus},
    seed,
};

static NEXT_TEST_DIR: AtomicUsize = AtomicUsize::new(0);

/// Shared OpenFold fixture: a script, a one-record FASTA dir, and empty data/output/alignment dirs.
pub(crate) struct TestLayout {
    pub root: PathBuf,
    pub working_dir: PathBuf,
    pub fasta_dir: PathBuf,
    pub data_dir: PathBuf,
    pub alignment_dir: PathBuf,
    pub output_location: PathBuf,
}

impl TestLayout {
    /// Written verbatim after `>`, so a caller controls both the tag and any trailing header text.
    pub fn new(fasta_header: &str) -> Self {
        let root = env::temp_dir().join(format!(
            "executor-test-layout-{}-{}",
            std::process::id(),
            NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let working_dir = root.join("workspace");
        let fasta_dir = root.join("fasta");
        let data_dir = root.join("data");
        let alignment_dir = root.join("alignments");
        let output_location = root.join("outputs");

        fs::create_dir_all(&working_dir).expect("working directory should be created");
        fs::create_dir_all(&fasta_dir).expect("fasta directory should be created");
        fs::create_dir_all(&data_dir).expect("data directory should be created");
        fs::create_dir_all(&output_location).expect("output location should be created");
        fs::write(working_dir.join("run_openfold.py"), "# test script")
            .expect("script should be created");
        fs::write(
            fasta_dir.join("input.fasta"),
            format!(">{fasta_header}\nMSTNPKPQRITF\n"),
        )
        .expect("matching FASTA should be created");

        Self {
            root,
            working_dir,
            fasta_dir,
            data_dir,
            alignment_dir,
            output_location,
        }
    }

    pub fn command(&self) -> CommandSpec {
        CommandSpec {
            program: "python3".into(),
            args: vec!["-u".into(), "run_openfold.py".into()],
            current_dir: Some(self.working_dir.clone()),
            ..Default::default()
        }
    }

    pub fn execution_parameters(&self) -> serde_json::Value {
        json!({
            "fasta_dir": self.fasta_dir,
            "data_dir": self.data_dir,
        })
    }
}

impl Drop for TestLayout {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// A migrated in-memory database. Foreign keys are off by default in SQLite, and every table the
/// executor writes is behind one, so a test without this passes on rows that could never exist.
pub(crate) async fn test_db() -> Result<DatabaseConnection, DbErr> {
    let db = Database::connect("sqlite::memory:").await?;
    db.execute(Statement::from_string(
        db.get_database_backend(),
        "PRAGMA foreign_keys = ON".to_owned(),
    ))
    .await?;
    db::migrate_database(&db).await?;
    Ok(db)
}

/// `test_db` plus the default catalog -- what a test needs when it submits or executes a run.
pub(crate) async fn seeded_db() -> Result<DatabaseConnection, DbErr> {
    let db = test_db().await?;
    seed::seed_defaults(&db).await?;
    Ok(db)
}

/// The status a named preflight check reported. Absent is a test bug, not a failure to assert on.
pub(crate) fn check_status(report: &PreflightReport, name: &str) -> PreflightStatus {
    report
        .checks
        .iter()
        .find(|check| check.name == name)
        .unwrap_or_else(|| panic!("{name} check should be present"))
        .status
}

/// The message a named preflight check reported, for asserting on what it says went wrong.
pub(crate) fn check_message<'a>(report: &'a PreflightReport, name: &str) -> &'a str {
    report
        .checks
        .iter()
        .find(|check| check.name == name)
        .unwrap_or_else(|| panic!("{name} check should be present"))
        .message
        .as_deref()
        .unwrap_or_else(|| panic!("{name} check should have a message"))
}

/// Env vars are process-wide, but cargo runs tests as threads of one process: a test that sets
/// `OPENFOLD_ENV_PREFIX` or `VIZFOLD_CONFIG` and a test that reads a config-derived path cannot be
/// allowed to overlap. Both sides take this, so they queue instead of racing.
pub(crate) fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The registered pair every run fixture needs: a backend with empty schemas and a local target.
/// The slugs are the caller's so two fixtures in one database cannot collide.
pub(crate) async fn local_backend_and_target(
    db: &DatabaseConnection,
    backend_slug: &str,
    target_slug: &str,
) -> Result<(model_backends::Model, execution_targets::Model), DbErr> {
    let backend = crate::core::services::model_backends::register_model_backend(
        db,
        crate::core::services::model_backends::RegisterModelBackendInput {
            slug: backend_slug.into(),
            label: backend_slug.into(),
            version: None,
            description: None,
            artifact_capabilities_json: "{}".into(),
            parameter_schema_json: json!({"type":"object","properties":{}}).to_string(),
        },
    )
    .await?;
    let target = crate::core::services::execution_targets::register_execution_target(
        db,
        crate::core::services::execution_targets::RegisterExecutionTargetInput {
            slug: target_slug.into(),
            target_type: "local".into(),
            description: None,
            available_resources_json: json!({"type":"object","properties":{}}).to_string(),
        },
    )
    .await?;
    Ok((backend, target))
}
