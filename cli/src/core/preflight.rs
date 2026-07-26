#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreflightStatus {
    Passed,
    Warning,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreflightCheck {
    pub name: String,
    pub status: PreflightStatus,
    pub message: Option<String>,
}

impl PreflightCheck {
    pub fn passed(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::with_message(name, PreflightStatus::Passed, message)
    }

    pub fn warning(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::with_message(name, PreflightStatus::Warning, message)
    }

    pub fn failed(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::with_message(name, PreflightStatus::Failed, message)
    }

    fn with_message(
        name: impl Into<String>,
        status: PreflightStatus,
        message: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            status,
            message: Some(message.into()),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PreflightReport {
    pub checks: Vec<PreflightCheck>,
}

impl PreflightReport {
    pub fn new(checks: Vec<PreflightCheck>) -> Self {
        Self { checks }
    }

    pub fn has_failures(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.status == PreflightStatus::Failed)
    }

    pub fn failures(&self) -> Vec<&PreflightCheck> {
        self.checks
            .iter()
            .filter(|check| check.status == PreflightStatus::Failed)
            .collect()
    }
}

use std::path::Path;

use crate::core::commands::CommandSpec;

/// Mirrors the fold's own `nvidia-smi --query-gpu=name --format=csv,noheader` probe.
pub fn detect_gpu() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    let name = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_owned();
    (output.status.success() && !name.is_empty()).then_some(name)
}

pub fn gpu_check(detected: Option<&str>) -> PreflightCheck {
    match detected {
        Some(name) => PreflightCheck::passed("gpu", format!("GPU visible: {name}")),
        None => PreflightCheck::warning("gpu", "no GPU visible; the run will fall back to CPU"),
    }
}

/// Callers push these command checks after `gpu_check` and before their own input checks.
pub fn base_command_checks(command: &CommandSpec) -> Vec<PreflightCheck> {
    let program = if command.program.trim().is_empty() {
        PreflightCheck::failed("program configured", "command program is empty")
    } else {
        PreflightCheck::passed(
            "program configured",
            format!("program '{}' is configured", command.program),
        )
    };

    let script = script_argument(command);
    let script_arg = match script {
        Some(script) => PreflightCheck::passed(
            "script argument configured",
            format!("script argument '{script}' follows -u"),
        ),
        None => PreflightCheck::failed(
            "script argument configured",
            "command args must include a script argument after -u",
        ),
    };

    let working_dir = match &command.current_dir {
        Some(current_dir) if current_dir.is_dir() => PreflightCheck::passed(
            "working directory",
            format!("working directory '{}' exists", current_dir.display()),
        ),
        Some(current_dir) => PreflightCheck::failed(
            "working directory",
            format!(
                "working directory '{}' does not exist or is not a directory",
                current_dir.display()
            ),
        ),
        None => PreflightCheck::warning(
            "working directory",
            "no working directory is configured; script resolution may depend on the caller",
        ),
    };

    let script_file = match (script, &command.current_dir) {
        (Some(script), _) if Path::new(script).is_absolute() => {
            path_exists_check("script file", Path::new(script))
        }
        (Some(script), Some(current_dir)) => {
            path_exists_check("script file", &current_dir.join(script))
        }
        (Some(script), None) => PreflightCheck::warning(
            "script file",
            format!("relative script '{script}' cannot be resolved without a working directory"),
        ),
        (None, _) => PreflightCheck::failed(
            "script file",
            "script path is unavailable because the -u script argument is missing",
        ),
    };

    vec![program, script_arg, working_dir, script_file]
}

fn script_argument(command: &CommandSpec) -> Option<&str> {
    command
        .args
        .iter()
        .position(|arg| arg == "-u")
        .and_then(|index| command.args.get(index + 1))
        .map(String::as_str)
        .filter(|script| !script.is_empty())
}

pub fn path_exists_check(name: &str, path: &Path) -> PreflightCheck {
    if path.exists() {
        PreflightCheck::passed(name, format!("'{}' exists", path.display()))
    } else {
        PreflightCheck::failed(name, format!("'{}' does not exist", path.display()))
    }
}

pub fn input_id_check(input_id: &str) -> PreflightCheck {
    if input_id.trim().is_empty() {
        PreflightCheck::failed("input_id", "run input_id is missing or empty")
    } else {
        PreflightCheck::passed(
            "input_id",
            format!("run input_id '{input_id}' is configured"),
        )
    }
}

pub fn output_dir_check(output_path: &Path) -> PreflightCheck {
    if output_path.exists() {
        return if output_path.is_dir() {
            PreflightCheck::passed(
                "output_dir parent",
                format!("'{}' already exists", output_path.display()),
            )
        } else {
            PreflightCheck::failed(
                "output_dir parent",
                format!("'{}' exists but is not a directory", output_path.display()),
            )
        };
    }

    let parent = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if parent.is_dir() {
        PreflightCheck::passed(
            "output_dir parent",
            format!("parent '{}' exists", parent.display()),
        )
    } else {
        PreflightCheck::failed(
            "output_dir parent",
            format!(
                "parent '{}' does not exist or is not a directory",
                parent.display()
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{PreflightCheck, PreflightReport, PreflightStatus};

    #[test]
    fn all_passing_checks_have_no_failures() {
        let report = PreflightReport::new(vec![PreflightCheck::passed("workspace", "ready")]);

        assert!(!report.has_failures());
    }

    #[test]
    fn failed_check_marks_report_as_failed() {
        let report = PreflightReport::new(vec![PreflightCheck::failed("python", "not found")]);

        assert!(report.has_failures());
    }

    #[test]
    fn warnings_do_not_count_as_failures() {
        let report = PreflightReport::new(vec![PreflightCheck::warning(
            "cuda",
            "GPU support is unavailable",
        )]);

        assert!(!report.has_failures());
    }

    #[test]
    fn helpers_return_checks_matching_their_status() {
        let report = PreflightReport::new(vec![
            PreflightCheck::passed("workspace", "ready"),
            PreflightCheck::warning("cuda", "unavailable"),
            PreflightCheck::failed("python", "not found"),
        ]);

        assert_eq!(report.failures().len(), 1);
        assert_eq!(report.failures()[0].status, PreflightStatus::Failed);
        assert!(report.has_failures());
    }

    #[test]
    fn empty_report_has_no_failures_or_checks() {
        let report = PreflightReport::default();

        assert!(!report.has_failures());
        assert!(report.failures().is_empty());
        assert!(report.checks.is_empty());
    }
}
