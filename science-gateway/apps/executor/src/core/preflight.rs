use sea_orm::DbErr;

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

pub trait PreflightRunner {
    fn run_preflight(&self) -> Result<PreflightReport, DbErr>;
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
