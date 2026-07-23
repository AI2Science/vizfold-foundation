use chrono::Utc;
use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{
    commands::CommandRunner,
    execution::{ExecutionWorkflowResult, execute_command_workflow},
    model_runners::openfold::{OpenFoldPreflightRunner, plan_openfold_command},
    repositories,
};

use super::runs::{self, UpdateRunStatusInput};

/// Plans and executes an OpenFold run stored in the executor database.
pub async fn execute_openfold_run(
    db: &DatabaseConnection,
    run_id: i32,
    runner: &dyn CommandRunner,
) -> Result<ExecutionWorkflowResult, DbErr> {
    let run = repositories::runs::find_by_id(db, run_id)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("run {run_id} does not exist")))?;

    let execution = async {
        let model_backend = repositories::model_backends::find_by_id(db, run.model_backend_id)
            .await?
            .ok_or_else(|| DbErr::Custom("model backend does not exist".into()))?;
        let execution_target =
            repositories::execution_targets::find_by_id(db, run.execution_target_id)
                .await?
                .ok_or_else(|| DbErr::Custom("execution target does not exist".into()))?;
        let invocation_profile =
            repositories::model_invocation_profiles::find_by_id(db, run.invocation_profile_id)
                .await?
                .ok_or_else(|| DbErr::Custom("model invocation profile does not exist".into()))?;

        let command =
            plan_openfold_command(&model_backend, &execution_target, &invocation_profile, &run)?;
        let preflight_runner = OpenFoldPreflightRunner {
            command: &command,
            invocation_profile: &invocation_profile,
            run: &run,
        };

        execute_command_workflow(&command, runner, Some(&preflight_runner)).await
    }
    .await;

    match execution {
        Ok(result) if result.command_output.is_none() => {
            mark_failed(db, run_id, preflight_failure_message(&result)).await?;
            Ok(result)
        }
        Ok(result) => {
            let output = result
                .command_output
                .as_ref()
                .expect("command output is present when execution was not skipped");
            if output.exit_code == 0 {
                runs::update_run_status(
                    db,
                    run_id,
                    UpdateRunStatusInput {
                        status: "completed".into(),
                        completed_at: Some(Some(Utc::now())),
                        error_message: Some(None),
                        ..Default::default()
                    },
                )
                .await?;
            } else {
                let message = if output.stderr.trim().is_empty() {
                    format!("OpenFold command exited with code {}", output.exit_code)
                } else {
                    output.stderr.trim().to_owned()
                };
                mark_failed(db, run_id, message).await?;
            }
            Ok(result)
        }
        Err(error) => {
            mark_failed(db, run_id, error.to_string()).await?;
            Err(error)
        }
    }
}

async fn mark_failed(
    db: &DatabaseConnection,
    run_id: i32,
    error_message: impl Into<String>,
) -> Result<(), DbErr> {
    runs::update_run_status(
        db,
        run_id,
        UpdateRunStatusInput {
            status: "failed".into(),
            completed_at: Some(Some(Utc::now())),
            error_message: Some(Some(error_message.into())),
            ..Default::default()
        },
    )
    .await?;
    Ok(())
}

fn preflight_failure_message(result: &ExecutionWorkflowResult) -> String {
    let failures = result
        .preflight_report
        .as_ref()
        .map(|report| {
            report
                .failures()
                .into_iter()
                .filter_map(|check| check.message.as_deref())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if failures.is_empty() {
        result
            .skipped_execution_reason
            .clone()
            .unwrap_or_else(|| "OpenFold execution was skipped".into())
    } else {
        format!("OpenFold preflight failed: {}", failures.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use sea_orm::{ConnectionTrait, Database, DbErr, Statement};
    use serde_json::json;

    use crate::core::{
        commands::{CommandOutput, CommandRunner, CommandSpec},
        db, repositories,
        services::{
            execution_targets::{self, RegisterExecutionTargetInput},
            model_backends::{self, RegisterModelBackendInput},
            model_invocation_profiles::{self, RegisterModelInvocationProfileInput},
            runs::{self, SubmitRunInput},
        },
    };

    use super::execute_openfold_run;

    struct TestRunner {
        output: CommandOutput,
        called: Arc<AtomicBool>,
        command: Arc<Mutex<Option<CommandSpec>>>,
    }

    #[async_trait::async_trait]
    impl CommandRunner for TestRunner {
        async fn run(&self, command: CommandSpec) -> Result<CommandOutput, DbErr> {
            self.called.store(true, Ordering::SeqCst);
            *self
                .command
                .lock()
                .expect("command lock should not be poisoned") = Some(command);
            Ok(self.output.clone())
        }
    }

    struct TestLayout {
        root: PathBuf,
        working_dir: PathBuf,
        fasta_dir: PathBuf,
        data_dir: PathBuf,
        output_location: PathBuf,
    }

    impl TestLayout {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "executor-openfold-execution-{}-{}",
                std::process::id(),
                chrono::Utc::now()
                    .timestamp_nanos_opt()
                    .expect("timestamp is representable")
            ));
            let working_dir = root.join("workspace");
            let fasta_dir = root.join("fasta");
            let data_dir = root.join("data");
            let output_location = root.join("outputs");
            fs::create_dir_all(&working_dir).expect("working directory should be created");
            fs::create_dir_all(&fasta_dir).expect("FASTA directory should be created");
            fs::create_dir_all(&data_dir).expect("data directory should be created");
            fs::create_dir_all(&output_location).expect("output location should be created");
            fs::write(working_dir.join("run_openfold.py"), "# test script")
                .expect("script should be created");
            fs::write(fasta_dir.join("input.fasta"), ">test_input\nMSTNPKPQRITF\n")
                .expect("FASTA should be created");
            Self {
                root,
                working_dir,
                fasta_dir,
                data_dir,
                output_location,
            }
        }
    }

    impl Drop for TestLayout {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    async fn test_db() -> Result<sea_orm::DatabaseConnection, DbErr> {
        let db = Database::connect("sqlite::memory:").await?;
        db.execute(Statement::from_string(
            db.get_database_backend(),
            "PRAGMA foreign_keys = ON".to_owned(),
        ))
        .await?;
        db::migrate_database(&db).await?;
        Ok(db)
    }

    async fn create_run(
        db: &sea_orm::DatabaseConnection,
        layout: &TestLayout,
        invalid_working_dir: bool,
    ) -> Result<crate::core::entities::runs::Model, DbErr> {
        let backend = model_backends::register_model_backend(
            db,
            RegisterModelBackendInput {
                slug: "openfold-test".into(),
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
                slug: "local-test".into(),
                target_type: "local".into(),
                description: None,
                available_resources_json: json!({"type":"object","properties":{}}).to_string(),
            },
        )
        .await?;
        let working_dir = if invalid_working_dir {
            layout.root.join("missing")
        } else {
            layout.working_dir.clone()
        };
        let profile = model_invocation_profiles::register_model_invocation_profile(db, RegisterModelInvocationProfileInput {
            model_backend_id: backend.id, execution_target_id: target.id, invocation_kind: "local_subprocess".into(),
            config_json: json!({"program":"python3","script":"run_openfold.py","working_dir":working_dir,"output_location":layout.output_location}).to_string(),
        }).await?;
        runs::submit_run(
            db,
            SubmitRunInput {
                model_backend_id: backend.id,
                execution_target_id: target.id,
                invocation_profile_id: profile.id,
                status: "submitted".into(),
                input_id: "test_input".into(),
                input_sequence: "MSTNPKPQRITF".into(),
                model_parameters_json: "{}".into(),
                execution_parameters_json:
                    json!({"fasta_dir":layout.fasta_dir,"data_dir":layout.data_dir}).to_string(),
            },
        )
        .await
    }

    fn runner(
        exit_code: i32,
        stderr: &str,
    ) -> (TestRunner, Arc<AtomicBool>, Arc<Mutex<Option<CommandSpec>>>) {
        let called = Arc::new(AtomicBool::new(false));
        let command = Arc::new(Mutex::new(None));
        (
            TestRunner {
                output: CommandOutput {
                    exit_code,
                    stdout: String::new(),
                    stderr: stderr.into(),
                },
                called: Arc::clone(&called),
                command: Arc::clone(&command),
            },
            called,
            command,
        )
    }

    #[tokio::test]
    async fn missing_run_returns_clear_error() -> Result<(), DbErr> {
        let db = test_db().await?;
        let (runner, _, _) = runner(0, "");
        let error = execute_openfold_run(&db, 999, &runner)
            .await
            .expect_err("missing run should error");
        assert!(error.to_string().contains("run 999 does not exist"));
        Ok(())
    }

    #[tokio::test]
    async fn successful_command_completes_run_and_uses_openfold_plan() -> Result<(), DbErr> {
        let db = test_db().await?;
        let layout = TestLayout::new();
        let run = create_run(&db, &layout, false).await?;
        let (runner, called, command) = runner(0, "");
        let result = execute_openfold_run(&db, run.id, &runner).await?;
        assert!(called.load(Ordering::SeqCst));
        assert_eq!(result.command_output.expect("output").exit_code, 0);
        let command = command
            .lock()
            .expect("command lock")
            .clone()
            .expect("planned command");
        assert_eq!(command.program, "python3");
        assert_eq!(command.args, vec!["-u", "run_openfold.py"]);
        let updated = repositories::runs::find_by_id(&db, run.id)
            .await?
            .expect("run exists");
        assert_eq!(updated.status, "completed");
        assert!(updated.completed_at.is_some());
        assert_eq!(updated.error_message, None);
        Ok(())
    }

    #[tokio::test]
    async fn non_zero_command_fails_run() -> Result<(), DbErr> {
        let db = test_db().await?;
        let layout = TestLayout::new();
        let run = create_run(&db, &layout, false).await?;
        let (runner, _, _) = runner(7, "OpenFold failed");
        execute_openfold_run(&db, run.id, &runner).await?;
        let updated = repositories::runs::find_by_id(&db, run.id)
            .await?
            .expect("run exists");
        assert_eq!(updated.status, "failed");
        assert_eq!(updated.error_message.as_deref(), Some("OpenFold failed"));
        Ok(())
    }

    #[tokio::test]
    async fn failing_preflight_skips_runner_and_fails_run() -> Result<(), DbErr> {
        let db = test_db().await?;
        let layout = TestLayout::new();
        let run = create_run(&db, &layout, true).await?;
        let (runner, called, _) = runner(0, "");
        let result = execute_openfold_run(&db, run.id, &runner).await?;
        assert!(!called.load(Ordering::SeqCst));
        assert_eq!(
            result.skipped_execution_reason.as_deref(),
            Some("preflight failed")
        );
        let updated = repositories::runs::find_by_id(&db, run.id)
            .await?
            .expect("run exists");
        assert_eq!(updated.status, "failed");
        assert!(
            updated
                .error_message
                .expect("error message")
                .contains("working directory")
        );
        Ok(())
    }
}
