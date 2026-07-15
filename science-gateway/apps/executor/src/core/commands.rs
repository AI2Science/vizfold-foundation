use std::{collections::BTreeMap, path::PathBuf};

use sea_orm::DbErr;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommandSpec {
    pub program: String,
    pub args: Vec<String>,
    pub current_dir: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[async_trait::async_trait]
pub trait CommandRunner {
    async fn run(&self, spec: CommandSpec) -> Result<CommandOutput, DbErr>;
}

#[derive(Clone, Debug, Default)]
pub struct LocalCommandRunner;

#[async_trait::async_trait]
impl CommandRunner for LocalCommandRunner {
    async fn run(&self, spec: CommandSpec) -> Result<CommandOutput, DbErr> {
        let mut command = tokio::process::Command::new(&spec.program);
        command.args(&spec.args);
        command.envs(&spec.env);

        if let Some(current_dir) = &spec.current_dir {
            command.current_dir(current_dir);
        }

        let output = command.output().await.map_err(|error| {
            DbErr::Custom(format!(
                "failed to spawn command '{}': {error}",
                spec.program
            ))
        })?;

        Ok(CommandOutput {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct FakeCommandRunner {
    output: Result<CommandOutput, String>,
}

#[cfg(test)]
impl FakeCommandRunner {
    pub fn succeeds(output: CommandOutput) -> Self {
        Self { output: Ok(output) }
    }

    pub fn fails(message: impl Into<String>) -> Self {
        Self {
            output: Err(message.into()),
        }
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl CommandRunner for FakeCommandRunner {
    async fn run(&self, _spec: CommandSpec) -> Result<CommandOutput, DbErr> {
        self.output
            .clone()
            .map_err(|message| DbErr::Custom(message))
    }
}

#[cfg(test)]
mod tests {
    use std::{env, path::Path};

    use super::{CommandOutput, CommandRunner, CommandSpec, FakeCommandRunner, LocalCommandRunner};

    #[cfg(unix)]
    fn shell_command(command: &str) -> CommandSpec {
        CommandSpec {
            program: "sh".into(),
            args: vec!["-c".into(), command.into()],
            ..Default::default()
        }
    }

    #[cfg(windows)]
    fn shell_command(command: &str) -> CommandSpec {
        CommandSpec {
            program: "cmd".into(),
            args: vec!["/C".into(), command.into()],
            ..Default::default()
        }
    }

    #[test]
    fn command_spec_captures_program_args_dir_and_env() {
        let mut spec = CommandSpec {
            program: "openfold".into(),
            args: vec!["--fasta".into(), "input.fasta".into()],
            current_dir: Some("runs/run-1".into()),
            ..Default::default()
        };
        spec.env.insert("CUDA_VISIBLE_DEVICES".into(), "0".into());

        assert_eq!(spec.program, "openfold");
        assert_eq!(spec.args, vec!["--fasta", "input.fasta"]);
        assert_eq!(spec.current_dir, Some("runs/run-1".into()));
        assert_eq!(spec.env["CUDA_VISIBLE_DEVICES"], "0");
    }

    #[tokio::test]
    async fn fake_command_runner_returns_configured_success() {
        let runner = FakeCommandRunner::succeeds(CommandOutput {
            exit_code: 0,
            stdout: "done".into(),
            stderr: String::new(),
        });

        let output = runner
            .run(CommandSpec {
                program: "ignored".into(),
                ..Default::default()
            })
            .await
            .expect("fake runner should succeed");

        assert_eq!(output.exit_code, 0);
        assert_eq!(output.stdout, "done");
        assert_eq!(output.stderr, "");
    }

    #[tokio::test]
    async fn fake_command_runner_returns_configured_failure() {
        let runner = FakeCommandRunner::fails("command failed");

        let error = runner
            .run(CommandSpec {
                program: "ignored".into(),
                ..Default::default()
            })
            .await
            .expect_err("fake runner should fail");

        assert!(error.to_string().contains("command failed"));
    }

    #[tokio::test]
    async fn local_command_runner_captures_successful_command_output() {
        let runner = LocalCommandRunner;
        #[cfg(unix)]
        let spec = shell_command("printf stdout; printf stderr >&2");
        #[cfg(windows)]
        let spec = shell_command("echo stdout & echo stderr 1>&2");

        let output = runner.run(spec).await.expect("command should run");

        assert_eq!(output.exit_code, 0);
        assert_eq!(output.stdout.trim(), "stdout");
        assert_eq!(output.stderr.trim(), "stderr");
    }

    #[tokio::test]
    async fn local_command_runner_returns_non_zero_exit_codes() {
        let runner = LocalCommandRunner;
        #[cfg(unix)]
        let spec = shell_command("exit 7");
        #[cfg(windows)]
        let spec = shell_command("exit /B 7");

        let output = runner.run(spec).await.expect("command should run");

        assert_eq!(output.exit_code, 7);
    }

    #[tokio::test]
    async fn local_command_runner_passes_environment_variables() {
        let runner = LocalCommandRunner;
        #[cfg(unix)]
        let mut spec = shell_command("printf '%s' \"$EXECUTOR_TEST_VALUE\"");
        #[cfg(windows)]
        let mut spec = shell_command("echo %EXECUTOR_TEST_VALUE%");
        spec.env
            .insert("EXECUTOR_TEST_VALUE".into(), "configured-value".into());

        let output = runner.run(spec).await.expect("command should run");

        assert_eq!(output.stdout.trim(), "configured-value");
    }

    #[tokio::test]
    async fn local_command_runner_applies_current_directory() {
        let runner = LocalCommandRunner;
        let current_dir = env::temp_dir()
            .canonicalize()
            .expect("temp directory exists");
        #[cfg(unix)]
        let mut spec = shell_command("pwd");
        #[cfg(windows)]
        let mut spec = shell_command("cd");
        spec.current_dir = Some(current_dir.clone());

        let output = runner.run(spec).await.expect("command should run");
        let reported_dir = Path::new(output.stdout.trim())
            .canonicalize()
            .expect("command should report a valid directory");

        assert_eq!(reported_dir, current_dir);
    }

    #[tokio::test]
    async fn local_command_runner_reports_spawn_failures_clearly() {
        let runner = LocalCommandRunner;
        let spec = CommandSpec {
            program: "executor-command-that-does-not-exist".into(),
            ..Default::default()
        };

        let error = runner
            .run(spec)
            .await
            .expect_err("missing command should fail to spawn");

        assert!(error.to_string().contains("failed to spawn command"));
        assert!(
            error
                .to_string()
                .contains("executor-command-that-does-not-exist")
        );
    }
}
