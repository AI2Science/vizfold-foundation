use std::path::Path;

use chrono::Utc;
use sea_orm::{DatabaseConnection, DbErr, EntityTrait};

use crate::core::{
    commands::{CommandOutput, CommandRunner, CommandSpec},
    config,
    entities::{execution_targets, model_backends, model_invocation_profiles, runs as run_entity},
    model_runners::{esmfold::preflight_esmfold, openfold::preflight_openfold, plan::plan_command},
    output_locations::resolve_output_location,
    preflight::PreflightReport,
};

use super::runs::{self, UpdateRunStatusInput};

/// Selects the preflight and the env wrapping. Unknown slugs fall through to OpenFold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendKind {
    Openfold,
    Esmfold,
}

impl BackendKind {
    fn from_slug(slug: &str) -> Self {
        match slug {
            "esmfold" => Self::Esmfold,
            _ => Self::Openfold,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Openfold => "OpenFold",
            Self::Esmfold => "ESMFold",
        }
    }
}

/// `output` is None when preflight failed, so nothing ran.
#[derive(Debug)]
pub struct ExecutionOutcome {
    pub report: PreflightReport,
    pub output: Option<CommandOutput>,
}

pub async fn execute_run(
    db: &DatabaseConnection,
    run_id: i32,
    runner: &dyn CommandRunner,
) -> Result<ExecutionOutcome, DbErr> {
    let run = run_entity::Entity::find_by_id(run_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("run {run_id} does not exist")))?;

    let started_at = Utc::now();
    let mut kind = BackendKind::Openfold;
    let execution: Result<ExecutionOutcome, DbErr> = async {
        let model_backend = model_backends::Entity::find_by_id(run.model_backend_id)
            .one(db)
            .await?
            .ok_or_else(|| DbErr::Custom("model backend does not exist".into()))?;
        kind = BackendKind::from_slug(&model_backend.slug);
        let execution_target = execution_targets::Entity::find_by_id(run.execution_target_id)
            .one(db)
            .await?
            .ok_or_else(|| DbErr::Custom("execution target does not exist".into()))?;
        let invocation_profile =
            model_invocation_profiles::Entity::find_by_id(run.invocation_profile_id)
                .one(db)
                .await?
                .ok_or_else(|| DbErr::Custom("model invocation profile does not exist".into()))?;

        // Created up front, so a fresh install needs no mkdir; OpenFold also seeds its attention/ dir.
        let workspace = resolve_output_location(&invocation_profile, &run)?;
        let to_create = match kind {
            BackendKind::Openfold => workspace.join("attention"),
            BackendKind::Esmfold => workspace.clone(),
        };
        std::fs::create_dir_all(&to_create).map_err(|error| {
            DbErr::Custom(format!(
                "failed to create run output workspace '{}': {error}",
                workspace.display()
            ))
        })?;

        let command = plan_command(&model_backend, &execution_target, &invocation_profile, &run)?;

        // Preflight sees the bare command, the runner the env-wrapped one; the CLI's prereq gate
        // already refused an uninstalled backend, so there is no unwrapped fallback to pick.
        let exec_command = match kind {
            BackendKind::Openfold => compose_exec_command(
                &command,
                &config::openfold_env_prefix(),
                &config::gpu_launch_args(),
            ),
            BackendKind::Esmfold => compose_esmfold_command(
                &command,
                &config::esmfold_env_prefix(),
                &config::gpu_launch_args(),
            ),
        };

        let report = match kind {
            BackendKind::Openfold => preflight_openfold(&command, &invocation_profile, &run)?,
            BackendKind::Esmfold => preflight_esmfold(&command, &invocation_profile, &run)?,
        };
        if report.has_failures() {
            return Ok(ExecutionOutcome {
                report,
                output: None,
            });
        }
        // Mark running before the fold blocks, so nothing reads a stale `submitted`.
        runs::update_run_status(
            db,
            run_id,
            UpdateRunStatusInput {
                status: "running".into(),
                started_at: Some(Some(started_at)),
                completed_at: None,
                error_message: None,
            },
        )
        .await?;
        let output = runner.run(exec_command).await?;
        Ok(ExecutionOutcome {
            report,
            output: Some(output),
        })
    }
    .await;

    match execution {
        Ok(outcome) if outcome.output.is_none() => {
            let failures = outcome
                .report
                .failures()
                .into_iter()
                .filter_map(|check| check.message.as_deref())
                .collect::<Vec<_>>();
            let message = if failures.is_empty() {
                format!("{} preflight failed", kind.label())
            } else {
                format!("{} preflight failed: {}", kind.label(), failures.join("; "))
            };
            mark_failed(db, run_id, started_at, message).await?;
            Ok(outcome)
        }
        Ok(outcome) => {
            let output = outcome
                .output
                .as_ref()
                .expect("command output is present when execution was not skipped");
            if output.exit_code == 0 {
                runs::update_run_status(
                    db,
                    run_id,
                    UpdateRunStatusInput {
                        status: "completed".into(),
                        started_at: Some(Some(started_at)),
                        completed_at: Some(Some(Utc::now())),
                        error_message: Some(None),
                    },
                )
                .await?;
                // Inline and idempotent: a completed run has its artifacts without a second command.
                super::run_artifacts::register_known_run_artifacts(db, run_id).await?;
            } else {
                let message = if output.stderr.trim().is_empty() {
                    format!(
                        "{} command exited with code {}",
                        kind.label(),
                        output.exit_code
                    )
                } else {
                    output.stderr.trim().to_owned()
                };
                mark_failed(db, run_id, started_at, message).await?;
            }
            Ok(outcome)
        }
        Err(error) => {
            // Don't `?`-propagate the DB write: it would mask the real execution error.
            let _ = mark_failed(db, run_id, started_at, error.to_string()).await;
            Err(error)
        }
    }
}

async fn mark_failed(
    db: &DatabaseConnection,
    run_id: i32,
    started_at: chrono::DateTime<Utc>,
    error_message: impl Into<String>,
) -> Result<(), DbErr> {
    runs::update_run_status(
        db,
        run_id,
        UpdateRunStatusInput {
            status: "failed".into(),
            started_at: Some(Some(started_at)),
            completed_at: Some(Some(Utc::now())),
            error_message: Some(Some(error_message.into())),
        },
    )
    .await?;
    Ok(())
}

/// `micromamba run -p` applies the env's activate.d hook, where every runtime variable a fold needs
/// is set -- the same command `setup::ready` hands the user.
fn activate_env_command(command: &CommandSpec, env_prefix: &Path) -> CommandSpec {
    let mut args = vec![
        "run".to_owned(),
        "-p".to_owned(),
        env_prefix.display().to_string(),
        command.program.clone(),
    ];
    args.extend(command.args.iter().cloned());
    let mut wrapped = CommandSpec {
        // Off PATH: install.sh puts it in ~/.local/bin, which srun's default --export=ALL carries over.
        program: "micromamba".to_owned(),
        args,
        ..command.clone()
    };
    // Carried, not left to activate.d: an older installer's hook sets neither. Triton defaults to NFS $HOME.
    let user = std::env::var("USER").unwrap_or_else(|_| "vizfold".to_owned());
    for (key, value) in [
        (
            "OPENFOLD_DATA_DIR",
            config::data_dir().display().to_string(),
        ),
        ("TRITON_CACHE_DIR", format!("/tmp/vizfold-triton-{user}")),
    ] {
        if std::env::var_os(key).is_none() {
            wrapped.env.entry(key.to_owned()).or_insert(value);
        }
    }
    wrapped
}

/// srun stays outermost, or the env is entered on the submit host. Streaming always on.
fn compose_exec_command(
    command: &CommandSpec,
    env_prefix: &Path,
    launch: &[String],
) -> CommandSpec {
    CommandSpec {
        stream: true,
        ..srun_command(activate_env_command(command, env_prefix), launch)
    }
}

/// ESMFold needs no activate.d hook, so its own `<env>/bin/python` is the whole activation.
fn compose_esmfold_command(
    command: &CommandSpec,
    env_prefix: &Path,
    launch: &[String],
) -> CommandSpec {
    let mut command = CommandSpec {
        program: env_prefix.join("bin/python").display().to_string(),
        ..command.clone()
    };
    // ~2.6 GB on the first fold; HuggingFace's default is the quota'd $HOME nothing ever cleans.
    if std::env::var_os("HF_HOME").is_none() {
        let cache = env_prefix.join("hf");
        command
            .env
            .insert("HF_HOME".to_owned(), cache.display().to_string());
    }
    CommandSpec {
        stream: true,
        ..srun_command(command, launch)
    }
}

fn srun_command(command: CommandSpec, launch: &[String]) -> CommandSpec {
    let Some((program, prefix)) = launch.split_first() else {
        return command;
    };
    let mut args = prefix.to_vec();
    args.push(command.program);
    args.extend(command.args);
    CommandSpec {
        program: program.clone(),
        args,
        current_dir: command.current_dir,
        env: command.env,
        stream: command.stream,
    }
}

#[cfg(test)]
mod tests {
    /// The var has to be on the command that folds, not merely recorded.
    #[test]
    fn an_installed_esmfold_command_caches_its_weights_under_the_env() {
        let env = PathBuf::from("/scratch/me/vizfold/envs/vizfold-esmfold");
        let planned = CommandSpec {
            program: "python3".to_owned(),
            ..Default::default()
        };

        let composed = compose_esmfold_command(&planned, &env, &[]);
        assert_eq!(
            composed.env.get("HF_HOME").map(String::as_str),
            Some("/scratch/me/vizfold/envs/vizfold-esmfold/hf"),
            "the fold must not cache weights in $HOME"
        );
    }

    use std::{
        path::PathBuf,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use sea_orm::{ConnectionTrait, Database, DbErr, EntityTrait, Statement};
    use serde_json::json;

    use crate::core::{
        commands::{CommandOutput, CommandRunner, CommandSpec, FakeCommandRunner},
        config, db,
        entities::runs as run_entity,
        services::{
            execution_targets::{self, RegisterExecutionTargetInput},
            model_backends::{self, RegisterModelBackendInput},
            model_invocation_profiles::{self, RegisterModelInvocationProfileInput},
            runs::{self, SubmitRunInput},
        },
        test_support::TestLayout,
    };

    use super::{activate_env_command, compose_esmfold_command, compose_exec_command, execute_run};

    #[test]
    fn srun_command_wraps_the_whole_activated_command() {
        let inner = CommandSpec {
            program: "bash".into(),
            args: vec!["-c".into(), "script".into()],
            current_dir: Some(PathBuf::from("/repo")),
            ..Default::default()
        };
        let wrapped = super::srun_command(
            inner,
            &["srun".to_owned(), "-p".to_owned(), "gpu".to_owned()],
        );

        assert_eq!(wrapped.program, "srun");
        assert_eq!(wrapped.args, vec!["-p", "gpu", "bash", "-c", "script"]);
        assert_eq!(wrapped.current_dir, Some(PathBuf::from("/repo")));
    }

    #[test]
    fn srun_command_is_a_no_op_without_a_launch_prefix() {
        let inner = CommandSpec {
            program: "python3".into(),
            ..Default::default()
        };
        assert_eq!(super::srun_command(inner.clone(), &[]), inner);
    }

    #[test]
    fn activate_env_command_runs_the_plan_through_micromamba_run() {
        let command = CommandSpec {
            program: "python3".into(),
            args: vec!["-u".into(), "run_openfold.py".into(), "6KWC_1".into()],
            current_dir: Some(PathBuf::from("/repo")),
            ..Default::default()
        };
        let wrapped =
            activate_env_command(&command, &PathBuf::from("/work/of/envs/vizfold-openfold"));

        // No shell, so nothing is re-quoted.
        assert_eq!(wrapped.program, "micromamba");
        assert_eq!(
            wrapped.args,
            [
                "run",
                "-p",
                "/work/of/envs/vizfold-openfold",
                "python3",
                "-u",
                "run_openfold.py",
                "6KWC_1"
            ]
        );
        assert_eq!(wrapped.current_dir, Some(PathBuf::from("/repo")));
    }

    /// Through an env built by an older installer, neither would be set.
    #[test]
    fn the_fold_carries_the_data_root_and_a_node_local_cache() {
        let wrapped = activate_env_command(&CommandSpec::default(), &PathBuf::from("/env"));

        assert_eq!(
            wrapped.env.get("OPENFOLD_DATA_DIR").map(String::as_str),
            Some(config::data_dir().display().to_string().as_str())
        );
        assert!(wrapped.env["TRITON_CACHE_DIR"].starts_with("/tmp/"));
    }

    #[test]
    fn compose_exec_command_wraps_srun_outside_the_activation() {
        let command = CommandSpec {
            program: "python3".into(),
            args: vec!["-u".into(), "run_openfold.py".into()],
            ..Default::default()
        };

        let composed = compose_exec_command(
            &command,
            &PathBuf::from("/work/of/envs/vizfold-openfold"),
            &["srun".to_owned(), "-p".to_owned(), "gpu".to_owned()],
        );

        assert_eq!(composed.program, "srun");
        assert_eq!(
            composed.args,
            [
                "-p",
                "gpu",
                "micromamba",
                "run",
                "-p",
                "/work/of/envs/vizfold-openfold",
                "python3",
                "-u",
                "run_openfold.py"
            ]
        );
        assert!(composed.stream);
    }

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

    async fn test_db() -> Result<sea_orm::DatabaseConnection, DbErr> {
        let db = Database::connect("sqlite::memory:").await?;
        db.execute(Statement::from_string(
            db.get_database_backend(),
            "PRAGMA foreign_keys = ON".to_owned(),
        ))
        .await?;
        db::migrate_database(&db).await?;
        crate::core::seed::seed_defaults(&db).await?;
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
                provenance_json: None,
            },
        )
        .await
    }

    /// The `esmfold` slug routes the run through the ESMFold path: one `--fasta`, no data_dir.
    async fn create_esmfold_run(
        db: &sea_orm::DatabaseConnection,
        layout: &TestLayout,
    ) -> Result<crate::core::entities::runs::Model, DbErr> {
        // Reuse the seeded esmfold backend; re-registering the slug violates its unique constraint.
        let backend = model_backends::list_model_backends(db)
            .await?
            .into_iter()
            .find(|backend| backend.slug == "esmfold")
            .expect("seeded esmfold backend");
        let target = execution_targets::register_execution_target(
            db,
            RegisterExecutionTargetInput {
                slug: "local-esmfold-test".into(),
                target_type: "local".into(),
                description: None,
                available_resources_json: json!({"type":"object","properties":{
                    "model_device":{"type":"string","enum":["cpu","cuda:0"],"default":"cuda:0","cli_flag":"--device"}
                }}).to_string(),
            },
        )
        .await?;
        let profile = model_invocation_profiles::register_model_invocation_profile(db, RegisterModelInvocationProfileInput {
            model_backend_id: backend.id, execution_target_id: target.id, invocation_kind: "local_subprocess".into(),
            config_json: json!({"program":"python3","script":"run_openfold.py","working_dir":layout.working_dir,"output_location":layout.output_location}).to_string(),
        }).await?;
        let fasta = layout.fasta_dir.join("input.fasta");
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
                execution_parameters_json: json!({"fasta": fasta, "model_device": "cpu"})
                    .to_string(),
                provenance_json: None,
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
        let error = execute_run(&db, 999, &runner)
            .await
            .expect_err("missing run should error");
        assert!(error.to_string().contains("run 999 does not exist"));
        Ok(())
    }

    #[tokio::test]
    async fn successful_command_completes_run_and_uses_openfold_plan() -> Result<(), DbErr> {
        let db = test_db().await?;
        let layout = TestLayout::new("test_input");
        let run = create_run(&db, &layout, false).await?;
        let (runner, called, command) = runner(0, "");
        execute_run(&db, run.id, &runner).await?;
        assert!(called.load(Ordering::SeqCst));
        let command = command
            .lock()
            .expect("command lock")
            .clone()
            .expect("planned command");
        assert_eq!(command.program, "micromamba");
        assert_eq!(
            command.args,
            vec![
                "run",
                "-p",
                &config::openfold_env_prefix().display().to_string(),
                "python3",
                "-u",
                "run_openfold.py"
            ]
        );
        assert!(command.stream, "long-running folds must stream output");
        let updated = run_entity::Entity::find_by_id(run.id)
            .one(&db)
            .await?
            .expect("run exists");
        assert_eq!(updated.status, "completed");
        assert!(updated.started_at.is_some());
        assert!(updated.completed_at.is_some());
        assert_eq!(updated.error_message, None);
        let workspace = layout.output_location.join(run.id.to_string());
        assert!(workspace.is_dir());
        assert!(workspace.join("attention").is_dir());
        // Output directories register inline on completion.
        let artifacts =
            crate::core::services::artifacts::list_artifacts_for_run(&db, run.id).await?;
        assert_eq!(artifacts.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn esmfold_run_completes_via_the_esmfold_path() -> Result<(), DbErr> {
        let db = test_db().await?;
        let layout = TestLayout::new("test_input");
        let run = create_esmfold_run(&db, &layout).await?;
        let (runner, called, command) = runner(0, "");

        execute_run(&db, run.id, &runner).await?;

        assert!(called.load(Ordering::SeqCst));
        let command = command
            .lock()
            .expect("command lock")
            .clone()
            .expect("planned command");
        assert_eq!(
            command.program,
            config::esmfold_env_prefix()
                .join("bin/python")
                .display()
                .to_string()
        );
        assert!(command.stream);

        let updated = run_entity::Entity::find_by_id(run.id)
            .one(&db)
            .await?
            .expect("run exists");
        assert_eq!(updated.status, "completed");
        // ESMFold seeds no attention/ dir, so only the output directory registers.
        let workspace = layout.output_location.join(run.id.to_string());
        assert!(workspace.is_dir());
        assert!(!workspace.join("attention").exists());
        let artifacts =
            crate::core::services::artifacts::list_artifacts_for_run(&db, run.id).await?;
        assert_eq!(artifacts.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn non_zero_command_fails_run() -> Result<(), DbErr> {
        let db = test_db().await?;
        let layout = TestLayout::new("test_input");
        let run = create_run(&db, &layout, false).await?;
        let (runner, _, _) = runner(7, "OpenFold failed");
        execute_run(&db, run.id, &runner).await?;
        let updated = run_entity::Entity::find_by_id(run.id)
            .one(&db)
            .await?
            .expect("run exists");
        assert_eq!(updated.status, "failed");
        assert_eq!(updated.error_message.as_deref(), Some("OpenFold failed"));
        Ok(())
    }

    #[tokio::test]
    async fn failing_preflight_skips_runner_and_fails_run() -> Result<(), DbErr> {
        let db = test_db().await?;
        let layout = TestLayout::new("test_input");
        let run = create_run(&db, &layout, true).await?;
        let (runner, called, _) = runner(0, "");
        let result = execute_run(&db, run.id, &runner).await?;
        assert!(!called.load(Ordering::SeqCst));
        assert!(result.output.is_none());
        assert!(result.report.has_failures());
        let updated = run_entity::Entity::find_by_id(run.id)
            .one(&db)
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

    #[tokio::test]
    async fn runner_error_after_preflight_passes_fails_run_and_propagates() -> Result<(), DbErr> {
        let db = test_db().await?;
        let layout = TestLayout::new("test_input");
        let run = create_run(&db, &layout, false).await?;
        let runner = FakeCommandRunner::fails("boom");

        let error = execute_run(&db, run.id, &runner)
            .await
            .expect_err("runner error should propagate");
        assert!(error.to_string().contains("boom"));

        let updated = run_entity::Entity::find_by_id(run.id)
            .one(&db)
            .await?
            .expect("run exists");
        assert_eq!(updated.status, "failed");
        assert!(
            updated
                .error_message
                .expect("error message")
                .contains("boom")
        );
        Ok(())
    }
}
