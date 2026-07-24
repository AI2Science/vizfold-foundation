use std::path::Path;

use sea_orm::DbErr;
use serde_json::Value;

use crate::core::{
    commands::CommandSpec,
    entities::{model_invocation_profiles, runs},
    output_locations::resolve_output_location,
    preflight::{PreflightCheck, PreflightReport},
};

use super::openfold::{
    base_command_checks, detect_gpu, gpu_check, input_id_check, output_dir_check,
};

/// Preflight for an ESMFold `run_pretrained_esmf.py` command. ESMFold is single-sequence
/// (HuggingFace `EsmForProteinFolding`, weights fetched at run time) so it needs none of
/// OpenFold's MSA/template checks: just the command itself, a readable `--fasta` file, and a
/// writable output workspace. The planned command is built by the shared schema-driven planner.
pub fn preflight_esmfold(
    command: &CommandSpec,
    invocation_profile: &model_invocation_profiles::Model,
    run: &runs::Model,
) -> Result<PreflightReport, DbErr> {
    let execution_parameters: Value = serde_json::from_str(&run.execution_parameters_json)
        .map_err(|error| {
            DbErr::Custom(format!(
                "run execution_parameters_json must be valid JSON: {error}"
            ))
        })?;

    let mut checks = vec![gpu_check(detect_gpu().as_deref())];
    checks.extend(base_command_checks(command));
    checks.push(input_id_check(&run.input_id));
    checks.push(fasta_file_check(&execution_parameters));
    checks.push(output_dir_check(&resolve_output_location(
        invocation_profile,
        run,
    )?));

    Ok(PreflightReport::new(checks))
}

/// ESMFold folds a single FASTA *file* (`--fasta`), unlike OpenFold's `fasta_dir`.
fn fasta_file_check(execution_parameters: &Value) -> PreflightCheck {
    let Some(fasta) = execution_parameters
        .get("fasta")
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
    else {
        return PreflightCheck::failed("fasta", "fasta is missing");
    };

    if Path::new(fasta).is_file() {
        PreflightCheck::passed("fasta", format!("'{fasta}' exists"))
    } else {
        PreflightCheck::failed(
            "fasta",
            format!("'{fasta}' does not exist or is not a file"),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use chrono::Utc;
    use serde_json::json;

    use crate::core::{
        commands::CommandSpec,
        entities::{model_invocation_profiles, runs},
        preflight::PreflightStatus,
        test_support::TestLayout,
    };

    use super::preflight_esmfold;

    fn invocation_profile(output_location: &std::path::Path) -> model_invocation_profiles::Model {
        let now = Utc::now();
        model_invocation_profiles::Model {
            id: 3,
            model_backend_id: 1,
            execution_target_id: 2,
            invocation_kind: "local_subprocess".into(),
            config_json: json!({ "output_location": output_location }).to_string(),
            created_at: now,
            updated_at: now,
        }
    }

    fn run(execution_parameters: serde_json::Value) -> runs::Model {
        let now = Utc::now();
        runs::Model {
            id: 4,
            model_backend_id: 1,
            execution_target_id: 2,
            invocation_profile_id: 3,
            status: "submitted".into(),
            input_id: "6KWC_1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json: "{}".into(),
            execution_parameters_json: execution_parameters.to_string(),
            provenance_json: None,
            submitted_at: now,
            started_at: None,
            completed_at: None,
            error_message: None,
        }
    }

    fn command(layout: &TestLayout) -> CommandSpec {
        CommandSpec {
            program: "python3".into(),
            args: vec!["-u".into(), "run_openfold.py".into()],
            current_dir: Some(layout.working_dir.clone()),
            ..Default::default()
        }
    }

    fn status(report: &crate::core::preflight::PreflightReport, name: &str) -> PreflightStatus {
        report
            .checks
            .iter()
            .find(|check| check.name == name)
            .unwrap_or_else(|| panic!("{name} check should be present"))
            .status
    }

    #[test]
    fn passes_when_fasta_file_and_workspace_parent_exist() {
        let layout = TestLayout::new("6KWC_1");
        let fasta = layout.fasta_dir.join("6KWC.fasta");
        fs::write(&fasta, ">6KWC_1\nMSTNPKPQRITF\n").expect("fasta should be written");

        let report = preflight_esmfold(
            &command(&layout),
            &invocation_profile(&layout.output_location),
            &run(json!({ "fasta": fasta, "model_device": "cpu" })),
        )
        .expect("preflight should inspect local paths");

        assert!(!report.has_failures());
        assert_eq!(status(&report, "fasta"), PreflightStatus::Passed);
        assert_eq!(
            status(&report, "output_dir parent"),
            PreflightStatus::Passed
        );
        // No MSA/template checks exist for ESMFold.
        assert!(!report.checks.iter().any(|check| check.name == "data_dir"));
    }

    #[test]
    fn fails_when_fasta_file_is_missing() {
        let layout = TestLayout::new("6KWC_1");
        let report = preflight_esmfold(
            &command(&layout),
            &invocation_profile(&layout.output_location),
            &run(json!({ "fasta": layout.fasta_dir.join("absent.fasta") })),
        )
        .expect("preflight should inspect the fasta path");

        assert!(report.has_failures());
        assert_eq!(status(&report, "fasta"), PreflightStatus::Failed);
    }
}
