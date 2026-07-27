use std::{collections::BTreeMap, path::PathBuf};

use sea_orm::DbErr;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommandSpec {
    pub program: String,
    pub args: Vec<String>,
    pub current_dir: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    /// Relay the child's output to the parent instead of capturing it, so a long run reports progress live.
    pub stream: bool,
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

        let spawn_error = |error: std::io::Error| {
            DbErr::Custom(format!(
                "failed to spawn command '{}': {}",
                spec.program, error
            ))
        };

        if spec.stream {
            // Progress, not the result: stdout stays clean so `fold --json` parses.
            command.stdout(std::process::Stdio::piped());
            let mut child = command.spawn().map_err(spawn_error)?;
            let relay = child.stdout.take().map(|mut out| {
                tokio::spawn(
                    async move { tokio::io::copy(&mut out, &mut tokio::io::stderr()).await },
                )
            });
            let status = child.wait().await.map_err(spawn_error)?;
            if let Some(relay) = relay {
                let _ = relay.await;
            }
            return Ok(CommandOutput {
                exit_code: status.code().unwrap_or(-1),
                stdout: String::new(),
                stderr: String::new(),
            });
        }

        let output = command.output().await.map_err(spawn_error)?;

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
        self.output.clone().map_err(DbErr::Custom)
    }
}

#[cfg(test)]
mod tests {
    use super::{CommandRunner, CommandSpec, LocalCommandRunner};

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

    #[tokio::test]
    async fn local_command_runner_captures_successful_command_output() {
        let runner = LocalCommandRunner;
        #[cfg(unix)]
        let spec = shell_command("printf stdout; printf stderr >&2");
        #[cfg(windows)]
        let spec = shell_command("echo stdout & echo stderr 1>&2");

        let output = runner.run(spec).await.expect("command should run");

        assert_eq!(output.stdout.trim(), "stdout");
        assert_eq!(output.stderr.trim(), "stderr");
    }

    /// A signal-killed child has no exit code; -1 is ours, so the run cannot look successful.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_signalled_child_reports_the_minus_one_sentinel() {
        let runner = LocalCommandRunner;

        let output = runner
            .run(shell_command("kill -9 $$"))
            .await
            .expect("command should run");

        assert_eq!(output.exit_code, -1);
    }

    #[tokio::test]
    async fn streaming_returns_the_exit_code_without_capturing_output() {
        let runner = LocalCommandRunner;
        #[cfg(unix)]
        let mut spec = shell_command("printf visible; exit 3");
        #[cfg(windows)]
        let mut spec = shell_command("echo visible & exit /B 3");
        spec.stream = true;

        let output = runner.run(spec).await.expect("command should run");

        assert_eq!(output.exit_code, 3);
        assert!(output.stdout.is_empty());
        assert!(output.stderr.is_empty());
    }

    #[tokio::test]
    async fn local_command_runner_applies_current_dir_and_env() {
        let runner = LocalCommandRunner;
        let dir = std::env::temp_dir();
        #[cfg(unix)]
        let mut spec = shell_command("pwd; printf \"$VIZFOLD_RUNNER_TEST_VAR\"");
        #[cfg(windows)]
        let mut spec = shell_command("cd & echo %VIZFOLD_RUNNER_TEST_VAR%");
        spec.current_dir = Some(dir.clone());
        spec.env
            .insert("VIZFOLD_RUNNER_TEST_VAR".into(), "applied".into());

        let output = runner.run(spec).await.expect("command should run");

        let printed_dir = output.stdout.lines().next().expect("pwd line");
        assert_eq!(
            std::fs::canonicalize(printed_dir).expect("printed dir should exist"),
            std::fs::canonicalize(&dir).expect("temp dir should exist")
        );
        assert!(output.stdout.contains("applied"));
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
