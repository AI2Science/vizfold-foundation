use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{
    commands::{CommandOutput, CommandRunner, CommandSpec},
    db,
    preflight::{PreflightReport, PreflightRunner},
    services,
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutionWorkflowResult {
    pub preflight_report: Option<PreflightReport>,
    pub command_output: Option<CommandOutput>,
    pub skipped_execution_reason: Option<String>,
}

pub async fn execute_command_workflow<R>(
    command: &CommandSpec,
    command_runner: &R,
    preflight_runner: Option<&dyn PreflightRunner>,
) -> Result<ExecutionWorkflowResult, DbErr>
where
    R: CommandRunner + ?Sized,
{
    let preflight_report = match preflight_runner {
        Some(preflight_runner) => {
            let report = preflight_runner.run_preflight()?;
            if report.has_failures() {
                return Ok(ExecutionWorkflowResult {
                    preflight_report: Some(report),
                    command_output: None,
                    skipped_execution_reason: Some("preflight failed".into()),
                });
            }
            Some(report)
        }
        None => None,
    };

    let command_output = command_runner.run(command.clone()).await?;

    Ok(ExecutionWorkflowResult {
        preflight_report,
        command_output: Some(command_output),
        skipped_execution_reason: None,
    })
}

pub struct ExecutionCore {
    db: DatabaseConnection,
}

impl ExecutionCore {
    pub async fn bootstrap() -> Result<Self, DbErr> {
        let db = db::connect_and_migrate().await?;
        crate::core::seed::seed_defaults(&db).await?;
        Ok(Self { db })
    }

    pub async fn check_readiness(&self) -> Result<(), DbErr> {
        let _ = services::model_backends::list_model_backends(&self.db).await?;
        Ok(())
    }

    pub fn db(&self) -> &DatabaseConnection {
        &self.db
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use sea_orm::DbErr;

    use crate::core::{
        commands::{CommandOutput, CommandRunner, CommandSpec},
        preflight::{PreflightCheck, PreflightReport, PreflightRunner},
    };

    use super::execute_command_workflow;

    struct TestCommandRunner {
        called: Arc<AtomicBool>,
        result: Result<CommandOutput, String>,
    }

    #[async_trait::async_trait]
    impl CommandRunner for TestCommandRunner {
        async fn run(&self, _spec: CommandSpec) -> Result<CommandOutput, DbErr> {
            self.called.store(true, Ordering::SeqCst);
            self.result.clone().map_err(DbErr::Custom)
        }
    }

    struct TestPreflightRunner {
        called: Arc<AtomicBool>,
        result: Result<PreflightReport, String>,
    }

    impl PreflightRunner for TestPreflightRunner {
        fn run_preflight(&self) -> Result<PreflightReport, DbErr> {
            self.called.store(true, Ordering::SeqCst);
            self.result.clone().map_err(DbErr::Custom)
        }
    }

    fn command() -> CommandSpec {
        CommandSpec {
            program: "ignored".into(),
            ..Default::default()
        }
    }

    fn successful_command_runner() -> (TestCommandRunner, Arc<AtomicBool>) {
        let called = Arc::new(AtomicBool::new(false));
        (
            TestCommandRunner {
                called: Arc::clone(&called),
                result: Ok(CommandOutput {
                    exit_code: 0,
                    stdout: "done".into(),
                    stderr: String::new(),
                }),
            },
            called,
        )
    }

    fn preflight_runner(
        result: Result<PreflightReport, String>,
    ) -> (TestPreflightRunner, Arc<AtomicBool>) {
        let called = Arc::new(AtomicBool::new(false));
        (
            TestPreflightRunner {
                called: Arc::clone(&called),
                result,
            },
            called,
        )
    }

    #[tokio::test]
    async fn runs_command_without_a_preflight_runner() {
        let (command_runner, command_called) = successful_command_runner();

        let result = execute_command_workflow(&command(), &command_runner, None)
            .await
            .expect("workflow should run command");

        assert!(command_called.load(Ordering::SeqCst));
        assert!(result.preflight_report.is_none());
        assert!(result.command_output.is_some());
        assert!(result.skipped_execution_reason.is_none());
    }

    #[tokio::test]
    async fn runs_command_after_a_passing_preflight() {
        let (command_runner, command_called) = successful_command_runner();
        let (preflight_runner, preflight_called) = preflight_runner(Ok(PreflightReport::new(
            vec![PreflightCheck::passed("workspace", "ready")],
        )));

        let result = execute_command_workflow(&command(), &command_runner, Some(&preflight_runner))
            .await
            .expect("workflow should run command");

        assert!(preflight_called.load(Ordering::SeqCst));
        assert!(command_called.load(Ordering::SeqCst));
        assert!(result.preflight_report.is_some());
        assert!(result.command_output.is_some());
    }

    #[tokio::test]
    async fn warning_only_preflight_does_not_block_execution() {
        let (command_runner, command_called) = successful_command_runner();
        let (preflight_runner, _) =
            preflight_runner(Ok(PreflightReport::new(vec![PreflightCheck::warning(
                "cuda",
                "not available",
            )])));

        let result = execute_command_workflow(&command(), &command_runner, Some(&preflight_runner))
            .await
            .expect("workflow should run command");

        assert!(command_called.load(Ordering::SeqCst));
        assert!(
            !result
                .preflight_report
                .expect("preflight report should be returned")
                .has_failures()
        );
        assert!(result.command_output.is_some());
    }

    #[tokio::test]
    async fn failing_preflight_skips_command_execution() {
        let (command_runner, command_called) = successful_command_runner();
        let (preflight_runner, preflight_called) = preflight_runner(Ok(PreflightReport::new(
            vec![PreflightCheck::failed("workspace", "missing")],
        )));

        let result = execute_command_workflow(&command(), &command_runner, Some(&preflight_runner))
            .await
            .expect("workflow should return skipped result");

        assert!(preflight_called.load(Ordering::SeqCst));
        assert!(!command_called.load(Ordering::SeqCst));
        assert!(result.command_output.is_none());
        assert_eq!(
            result.skipped_execution_reason.as_deref(),
            Some("preflight failed")
        );
        assert!(
            result
                .preflight_report
                .expect("preflight report should be returned")
                .has_failures()
        );
    }

    #[tokio::test]
    async fn command_runner_errors_propagate_after_preflight_passes() {
        let command_called = Arc::new(AtomicBool::new(false));
        let command_runner = TestCommandRunner {
            called: Arc::clone(&command_called),
            result: Err("command failed".into()),
        };
        let (preflight_runner, _) = preflight_runner(Ok(PreflightReport::default()));

        let error = execute_command_workflow(&command(), &command_runner, Some(&preflight_runner))
            .await
            .expect_err("command runner error should propagate");

        assert!(command_called.load(Ordering::SeqCst));
        assert!(error.to_string().contains("command failed"));
    }

    #[tokio::test]
    async fn preflight_runner_errors_propagate_before_command_execution() {
        let (command_runner, command_called) = successful_command_runner();
        let (preflight_runner, preflight_called) = preflight_runner(Err("preflight failed".into()));

        let error = execute_command_workflow(&command(), &command_runner, Some(&preflight_runner))
            .await
            .expect_err("preflight error should propagate");

        assert!(preflight_called.load(Ordering::SeqCst));
        assert!(!command_called.load(Ordering::SeqCst));
        assert!(error.to_string().contains("preflight failed"));
    }
}
