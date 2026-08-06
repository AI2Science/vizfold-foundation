use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sea_orm::DbErr;
use serde_json::Value;

use super::plan::{optional_bool, optional_string};
use crate::core::services::validation::require_json_object;
use crate::core::{
    commands::CommandSpec,
    entities::{model_invocation_profiles, runs},
    output_locations::resolve_output_location,
    preflight::{
        PreflightCheck, PreflightReport, base_command_checks, detect_gpu, gpu_check,
        input_id_check, output_dir_check,
    },
};

pub fn preflight_openfold(
    command: &CommandSpec,
    invocation_profile: &model_invocation_profiles::Model,
    run: &runs::Model,
) -> Result<PreflightReport, DbErr> {
    let execution_parameters = require_json_object(
        "run execution_parameters_json",
        &run.execution_parameters_json,
    )?;
    let mut checks = vec![gpu_check(detect_gpu().as_deref())];
    checks.extend(base_command_checks(command));

    checks.push(input_id_check(&run.input_id));
    checks.push(fasta_input_check(&execution_parameters, &run.input_id));
    checks.push(required_directory_check(&execution_parameters, "data_dir"));
    let output_dir = resolve_output_location(invocation_profile, run)?;
    checks.push(output_dir_check(&output_dir));

    if optional_bool(&execution_parameters, "use_precomputed_alignments").unwrap_or(false) {
        checks.push(required_directory_check(
            &execution_parameters,
            "alignment_dir",
        ));
        checks.push(precomputed_alignment_key_check(
            &execution_parameters,
            &run.input_id,
        ));
    }

    Ok(PreflightReport::new(checks))
}

fn required_directory_check(parameters: &Value, field_name: &str) -> PreflightCheck {
    let Some(path) = optional_string(parameters, field_name).filter(|path| !path.is_empty()) else {
        return PreflightCheck::failed(field_name, format!("{field_name} is missing"));
    };

    if Path::new(&path).is_dir() {
        PreflightCheck::passed(field_name, format!("'{path}' exists and is a directory"))
    } else {
        PreflightCheck::failed(
            field_name,
            format!("'{path}' does not exist or is not a directory"),
        )
    }
}

fn fasta_input_check(parameters: &Value, input_id: &str) -> PreflightCheck {
    let Some(fasta_dir) = optional_string(parameters, "fasta_dir").filter(|path| !path.is_empty())
    else {
        return PreflightCheck::failed("fasta_dir", "fasta_dir is missing");
    };

    let fasta_dir = Path::new(&fasta_dir);
    // A one-target run passes the FASTA file itself, so a file counts as much as a directory.
    let fasta_files = if fasta_dir.is_file() {
        vec![fasta_dir.to_path_buf()]
    } else if fasta_dir.is_dir() {
        match fasta_files_in_directory(fasta_dir) {
            Ok(files) => files,
            Err(error) => {
                return PreflightCheck::failed(
                    "fasta_dir",
                    format!("could not inspect '{}': {error}", fasta_dir.display()),
                );
            }
        }
    } else {
        return PreflightCheck::failed(
            "fasta_dir",
            format!("'{}' does not exist", fasta_dir.display()),
        );
    };

    if fasta_files.is_empty() {
        return PreflightCheck::failed(
            "fasta_dir",
            format!("'{}' contains no .fasta or .fa files", fasta_dir.display()),
        );
    }

    let mut found: BTreeSet<String> = BTreeSet::new();
    for fasta_path in &fasta_files {
        // Monomer mode skips a multi-record file with only a print, so the target would vanish.
        match parse_single_fasta_tag(fasta_path) {
            Ok(tag) => {
                found.insert(tag);
            }
            Err(error) => {
                return PreflightCheck::failed(
                    "fasta_dir",
                    format!(
                        "'{}' is not a valid single-record FASTA: {error}",
                        fasta_path.display()
                    ),
                );
            }
        }
    }

    if found.len() != fasta_files.len() {
        return PreflightCheck::failed(
            "fasta_dir",
            format!(
                "'{}' holds {} FASTA files but only {} distinct tags; every output is keyed by tag",
                fasta_dir.display(),
                fasta_files.len(),
                found.len()
            ),
        );
    }

    if input_id.trim().is_empty() {
        return PreflightCheck::failed(
            "fasta_dir",
            "cannot validate FASTA identity because run input_id is missing or empty",
        );
    }

    // input_id names the whole batch: `+`-joined tags.
    let expected: BTreeSet<&str> = input_id.split('+').collect();
    let found: BTreeSet<&str> = found.iter().map(String::as_str).collect();
    if found != expected {
        let difference = |from: &BTreeSet<&str>, to: &BTreeSet<&str>| {
            from.difference(to).copied().collect::<Vec<_>>().join(", ")
        };
        return PreflightCheck::failed(
            "fasta_dir",
            format!(
                "FASTA tag set '{}' does not match run input_id '{input_id}' (missing: '{}')",
                difference(&found, &expected),
                difference(&expected, &found)
            ),
        );
    }

    PreflightCheck::passed(
        "fasta_dir",
        format!(
            "'{}' holds {} FASTA file(s), tagged '{input_id}' as run input_id says",
            fasta_dir.display(),
            fasta_files.len()
        ),
    )
}

fn fasta_files_in_directory(fasta_dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut fasta_files = std::fs::read_dir(fasta_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            matches!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("fasta" | "fa")
            )
        })
        .collect::<Vec<_>>();
    fasta_files.sort();
    Ok(fasta_files)
}

fn parse_single_fasta_tag(fasta_path: &Path) -> Result<String, String> {
    let contents = std::fs::read_to_string(fasta_path).map_err(|error| error.to_string())?;
    let mut lines = contents.lines();
    let Some(header) = lines.next().map(str::trim) else {
        return Err("file is empty".into());
    };
    let Some(header_text) = header.strip_prefix('>') else {
        return Err("first line is not a FASTA header".into());
    };
    if lines.any(|line| line.trim_start().starts_with('>')) {
        return Err("multiple FASTA records are not supported".into());
    }

    let tag = header_text
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect::<String>();
    if tag.is_empty() {
        return Err("header does not contain an OpenFold tag".into());
    }

    Ok(tag)
}

fn precomputed_alignment_key_check(parameters: &Value, input_id: &str) -> PreflightCheck {
    let Some(alignment_dir) =
        optional_string(parameters, "alignment_dir").filter(|path| !path.is_empty())
    else {
        return PreflightCheck::failed("precomputed alignment key", "alignment_dir is missing");
    };

    let alignment_dir = Path::new(&alignment_dir);
    if !alignment_dir.is_dir() {
        return PreflightCheck::failed(
            "precomputed alignment key",
            format!("alignment_dir '{}' is unavailable", alignment_dir.display()),
        );
    }
    if input_id.trim().is_empty() {
        return PreflightCheck::failed(
            "precomputed alignment key",
            "cannot validate alignment key because run input_id is missing or empty",
        );
    }

    // One key per tag, so a target with no alignments fails before the GPU is touched.
    let missing: Vec<String> = input_id
        .split('+')
        .map(|tag| alignment_dir.join(tag))
        .filter(|key_directory| !key_directory.is_dir())
        .map(|key_directory| format!("'{}'", key_directory.display()))
        .collect();
    if missing.is_empty() {
        PreflightCheck::passed(
            "precomputed alignment key",
            format!(
                "'{}' holds every tag in '{input_id}'",
                alignment_dir.display()
            ),
        )
    } else {
        PreflightCheck::failed(
            "precomputed alignment key",
            format!(
                "missing alignment director{}: {}",
                if missing.len() == 1 { "y" } else { "ies" },
                missing.join(", ")
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{env, fs, path::PathBuf};

    use chrono::Utc;
    use sea_orm::DbErr;
    use serde_json::json;

    use crate::core::{
        commands::CommandSpec,
        entities::{execution_targets, model_backends, model_invocation_profiles, runs},
        preflight::{PreflightReport, PreflightStatus},
        test_support::{TestLayout, check_message, check_status},
    };

    use super::preflight_openfold as preflight_openfold_impl;
    use crate::core::model_runners::plan::{plan_command, resolve_declared_value};

    fn preflight_run(execution_parameters: serde_json::Value) -> runs::Model {
        preflight_run_with_input_id("1UBQ_1", execution_parameters)
    }

    fn preflight_run_with_input_id(
        input_id: &str,
        execution_parameters: serde_json::Value,
    ) -> runs::Model {
        let mut run = run(json!({}).to_string(), execution_parameters.to_string());
        run.input_id = input_id.into();
        run
    }

    fn preflight_invocation_profile() -> model_invocation_profiles::Model {
        invocation_profile(json!({"output_location": env::temp_dir()}).to_string())
    }

    fn preflight_openfold(
        command: &CommandSpec,
        run: &runs::Model,
    ) -> Result<PreflightReport, DbErr> {
        let invocation_profile = preflight_invocation_profile();
        preflight_openfold_impl(command, &invocation_profile, run)
    }

    fn model_backend() -> model_backends::Model {
        model_backends::Model {
            id: 1,
            slug: "openfold".into(),
            label: "OpenFold".into(),
            version: Some("test".into()),
            description: None,
            artifact_capabilities_json: "{}".into(),
            parameter_schema_json: openfold_parameter_schema().to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn execution_target() -> execution_targets::Model {
        execution_targets::Model {
            id: 2,
            slug: "local".into(),
            target_type: "local".into(),
            description: None,
            available_resources_json: available_resources_schema().to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn invocation_profile(config_json: String) -> model_invocation_profiles::Model {
        model_invocation_profiles::Model {
            id: 3,
            model_backend_id: 1,
            execution_target_id: 2,
            invocation_kind: "local_subprocess".into(),
            config_json,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn run(model_parameters_json: String, execution_parameters_json: String) -> runs::Model {
        runs::Model {
            id: 4,
            model_backend_id: 1,
            execution_target_id: 2,
            invocation_profile_id: 3,
            status: "submitted".into(),
            input_id: "1UBQ_1".into(),
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json,
            execution_parameters_json,
            provenance_json: None,
            submitted_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error_message: None,
        }
    }

    fn config() -> String {
        json!({
            "program": "python3",
            "script": "run_pretrained_openfold.py",
            "output_location": "/tmp/outputs",
            "working_dir": "/repo",
            "env": {
                "PYTHONPATH": "/repo"
            }
        })
        .to_string()
    }

    fn execution_parameters() -> serde_json::Value {
        json!({
            "fasta_dir": "/tmp/fasta",
            "data_dir": "/data",
            "model_device": "cuda:0"
        })
    }

    fn openfold_parameter_schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "config_preset": {
                    "type": "string",
                    "default": "model_1_ptm",
                    "cli_flag": "--config_preset"
                },
                "fasta_dir": {
                    "type": "path",
                    "source": "execution_parameters",
                    "parameter": "fasta_dir",
                    "positional": true,
                    "position": 1
                },
                "template_mmcif_dir": {
                    "type": "path",
                    "source": "data_dir",
                    "relative_path": "pdb_mmcif/mmcif_files",
                    "positional": true,
                    "position": 2
                },
                "uniref90_database_path": {
                    "type": "path",
                    "source": "data_dir",
                    "relative_path": "uniref90/uniref90.fasta",
                    "cli_flag": "--uniref90_database_path"
                },
                "mgnify_database_path": {
                    "type": "path",
                    "source": "data_dir",
                    "relative_path": "mgnify/mgy_clusters_2022_05.fa",
                    "cli_flag": "--mgnify_database_path"
                },
                "pdb70_database_path": {
                    "type": "path",
                    "source": "data_dir",
                    "relative_path": "pdb70/pdb70",
                    "cli_flag": "--pdb70_database_path"
                },
                "uniclust30_database_path": {
                    "type": "path",
                    "source": "data_dir",
                    "relative_path": "uniclust30/uniclust30_2018_08/uniclust30_2018_08",
                    "cli_flag": "--uniclust30_database_path"
                },
                "bfd_database_path": {
                    "type": "path",
                    "source": "data_dir",
                    "relative_path": "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
                    "cli_flag": "--bfd_database_path"
                },
                "output_dir": {
                    "type": "path",
                    "source": "run_output_workspace",
                    "cli_flag": "--output_dir"
                },
                "attn_map_dir": {
                    "type": "path",
                    "source": "run_output_workspace",
                    "relative_path": "attention",
                    "cli_flag": "--attn_map_dir"
                },
                "save_outputs": {
                    "type": "boolean",
                    "cli_flag": "--save_outputs"
                },
                "demo_attn": {
                    "type": "boolean",
                    "cli_flag": "--demo_attn"
                },
                "num_recycles_save": {
                    "type": "integer",
                    "cli_flag": "--num_recycles_save"
                }
            }
        })
    }

    fn available_resources_schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "model_device": {
                    "type": "string",
                    "enum": ["cpu", "cuda:0"],
                    "default": "cuda:0",
                    "cli_flag": "--model_device"
                },
                "cpus": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 14,
                    "cli_flag": "--cpus"
                }
            }
        })
    }

    #[test]
    fn builds_basic_openfold_command_spec() {
        let command = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution_parameters().to_string()),
        )
        .expect("command should plan");

        assert_eq!(command.program, "python3");
        assert_eq!(command.current_dir, Some("/repo".into()));
        assert_eq!(command.env["PYTHONPATH"], "/repo");
        assert_eq!(command.args[0], "-u");
        assert_eq!(command.args[1], "run_pretrained_openfold.py");
        assert_eq!(command.args[2], "/tmp/fasta");
        assert_eq!(command.args[3], "/data/pdb_mmcif/mmcif_files");
        let output_dir = PathBuf::from("/tmp/outputs").join("4");
        assert_pair(
            &command.args,
            "--attn_map_dir",
            &output_dir.join("attention").to_string_lossy(),
        );
        assert!(command.args.contains(&"--config_preset".into()));
        assert!(command.args.contains(&"model_1_ptm".into()));
        assert!(command.args.contains(&"--model_device".into()));
        assert!(command.args.contains(&"cuda:0".into()));
    }

    #[test]
    fn schema_declared_output_paths_ignore_execution_parameter_values() {
        let mut execution = execution_parameters();
        execution["output_dir"] = json!("/stale/output");
        execution["attn_map_dir"] = json!("/stale/attention");

        let command = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect("command should plan");

        let output_dir = PathBuf::from("/tmp/outputs").join("4");
        assert_pair(&command.args, "--output_dir", &output_dir.to_string_lossy());
        assert_pair(
            &command.args,
            "--attn_map_dir",
            &output_dir.join("attention").to_string_lossy(),
        );
        assert!(!command.args.contains(&"/stale/output".into()));
        assert!(!command.args.contains(&"/stale/attention".into()));
    }

    #[test]
    fn schema_declared_output_paths_require_profile_output_location() {
        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(
                json!({
                    "program": "python3",
                    "script": "run_pretrained_openfold.py"
                })
                .to_string(),
            ),
            &run(json!({}).to_string(), execution_parameters().to_string()),
        )
        .expect_err("schema-declared output paths should require output_location");

        assert!(error.to_string().contains("output_location is required"));
    }

    #[test]
    fn resolves_invocation_profile_config_source_with_relative_path() {
        let invocation_config = json!({"profile_data_dir": "/profile/data"});
        let declaration = json!({
            "source": "invocation_profile_config",
            "parameter": "profile_data_dir",
            "relative_path": "datasets/openfold"
        });

        let value = resolve_declared_value(
            &declaration,
            &json!({}),
            &json!({}),
            &invocation_config,
            &invocation_profile(invocation_config.to_string()),
            &run(json!({}).to_string(), json!({}).to_string()),
        )
        .expect("invocation profile config source should resolve");

        assert_eq!(value, "/profile/data/datasets/openfold");
    }

    #[test]
    fn invocation_profile_config_source_requires_the_declared_key() {
        let declaration = json!({
            "source": "invocation_profile_config",
            "parameter": "profile_data_dir"
        });

        let error = resolve_declared_value(
            &declaration,
            &json!({}),
            &json!({}),
            &json!({}),
            &invocation_profile("{}".into()),
            &run(json!({}).to_string(), json!({}).to_string()),
        )
        .expect_err("missing invocation profile config should fail");

        assert!(
            error
                .to_string()
                .contains("invocation profile config 'profile_data_dir' is required")
        );
    }

    #[test]
    fn includes_optional_model_parameters_when_present() {
        let command = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(
                json!({
                    "config_preset": "model_2_ptm",
                    "save_outputs": true,
                    "num_recycles_save": 1,
                    "model_preset": "monomer"
                })
                .to_string(),
                execution_parameters().to_string(),
            ),
        )
        .expect("command should plan");

        assert!(command.args.contains(&"model_2_ptm".into()));
        assert!(command.args.contains(&"--save_outputs".into()));
        assert!(command.args.contains(&"--num_recycles_save".into()));
        assert!(command.args.contains(&"1".into()));
        assert!(!command.args.contains(&"--model_preset".into()));
    }

    #[test]
    fn available_resources_flags_come_from_execution_parameters() {
        let cases = [
            (
                "model_device",
                json!("cpu"),
                json!({"model_device": "cuda:0"}), // same-named model parameter must not win
                "--model_device",
                "cpu",
            ),
            ("cpus", json!(12), json!({}), "--cpus", "12"),
        ];

        for (field, value, model_parameters, expected_flag, expected_value) in cases {
            let mut execution = execution_parameters();
            execution[field] = value;

            let command = plan_command(
                &model_backend(),
                &execution_target(),
                &invocation_profile(config()),
                &run(model_parameters.to_string(), execution.to_string()),
            )
            .unwrap_or_else(|error| panic!("command should plan for {field}: {error}"));

            assert_pair(&command.args, expected_flag, expected_value);
        }
    }

    #[test]
    fn rejects_invalid_model_device_from_available_resources_enum() {
        let mut execution = execution_parameters();
        execution["model_device"] = json!("cuda:1");

        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect_err("unsupported model device should fail");

        assert!(
            error
                .to_string()
                .contains("model_device 'cuda:1' is not allowed")
        );
    }

    #[test]
    fn rejects_cpus_above_available_resources_maximum() {
        let mut execution = execution_parameters();
        execution["cpus"] = json!(15);

        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect_err("too many cpus should fail");

        assert!(error.to_string().contains("cpus 15 exceeds"));
    }

    #[test]
    fn rejects_wrong_typed_cpus_that_would_bypass_range_guard() {
        let mut execution = execution_parameters();
        execution["cpus"] = json!("999");

        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect_err("string cpus must not bypass the integer range guard");

        assert!(error.to_string().contains("cpus must be an integer"));
    }

    #[test]
    fn a_trailing_separator_on_data_dir_does_not_double_up() {
        let mut execution = execution_parameters();
        execution["data_dir"] = json!("/data/");

        let command = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect("command should plan");

        assert_pair(
            &command.args,
            "--uniref90_database_path",
            "/data/uniref90/uniref90.fasta",
        );
    }

    #[test]
    fn residue_idx_is_emitted_as_triangle_residue_idx() {
        let mut execution = execution_parameters();
        execution["residue_idx"] = json!(7);

        let command = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(
                json!({"demo_attn": true}).to_string(),
                execution.to_string(),
            ),
        )
        .expect("command should plan");

        assert_pair(&command.args, "--triangle_residue_idx", "7");
        assert!(command.args.contains(&"--demo_attn".into()));
    }

    #[test]
    fn includes_precomputed_alignment_flags_when_requested() {
        let mut execution = execution_parameters();
        execution["use_precomputed_alignments"] = json!(true);
        execution["alignment_dir"] = json!("/tmp/alignments");

        let command = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect("command should plan");

        assert_pair(
            &command.args,
            "--use_precomputed_alignments",
            "/tmp/alignments",
        );
    }

    #[test]
    fn rejects_missing_alignment_dir_when_precomputed_alignments_requested() {
        let mut execution = execution_parameters();
        execution["use_precomputed_alignments"] = json!(true);

        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect_err("missing alignment_dir should fail");

        assert!(error.to_string().contains("alignment_dir is required"));
    }

    #[test]
    fn returns_clear_error_when_required_config_field_is_missing() {
        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(json!({"script": "run_pretrained_openfold.py"}).to_string()),
            &run(json!({}).to_string(), execution_parameters().to_string()),
        )
        .expect_err("missing program should fail");

        assert!(error.to_string().contains("program is required"));
    }

    #[test]
    fn returns_clear_error_when_required_execution_parameter_is_missing() {
        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(
                json!({}).to_string(),
                json!({
                    "fasta_dir": "/tmp/fasta"
                })
                .to_string(),
            ),
        )
        .expect_err("missing data_dir should fail");

        assert!(error.to_string().contains("data_dir is required"));
    }

    #[test]
    fn returns_clear_error_when_schema_declared_fasta_dir_is_missing() {
        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(
                json!({}).to_string(),
                json!({
                    "data_dir": "/data",
                    "model_device": "cuda:0"
                })
                .to_string(),
            ),
        )
        .expect_err("missing fasta_dir should fail");

        assert!(error.to_string().contains("fasta_dir is required"));
    }

    #[test]
    fn validates_env_is_string_to_string_object() {
        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(
                json!({
                    "program": "python3",
                    "script": "run_pretrained_openfold.py",
                    "env": {"PYTHONPATH": 123}
                })
                .to_string(),
            ),
            &run(json!({}).to_string(), execution_parameters().to_string()),
        )
        .expect_err("non-string env values should fail");

        assert!(
            error
                .to_string()
                .contains("config env value for 'PYTHONPATH' must be a string")
        );
    }

    #[test]
    fn rejects_non_local_subprocess_invocation_profile() {
        let mut profile = invocation_profile(config());
        profile.invocation_kind = "docker".into();

        let error = plan_command(
            &model_backend(),
            &execution_target(),
            &profile,
            &run(json!({}).to_string(), execution_parameters().to_string()),
        )
        .expect_err("unsupported invocation kind should fail");

        assert!(
            error
                .to_string()
                .contains("only supports local_subprocess invocation profiles")
        );
    }

    #[test]
    fn preflight_passes_when_local_configuration_is_ready() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let report = preflight_openfold(
            &layout.command(),
            &preflight_run(layout.execution_parameters()),
        )
        .expect("preflight should inspect valid local paths");

        assert!(!report.has_failures());
        assert_eq!(
            check_status(&report, "program configured"),
            PreflightStatus::Passed
        );
        assert_eq!(
            check_status(&report, "script file"),
            PreflightStatus::Passed
        );
        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Passed);
        assert_eq!(check_status(&report, "data_dir"), PreflightStatus::Passed);
        assert_eq!(
            check_status(&report, "output_dir parent"),
            PreflightStatus::Passed
        );
    }

    #[test]
    fn preflight_warns_when_relative_script_has_no_working_directory() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let mut command = layout.command();
        command.current_dir = None;

        let report = preflight_openfold(&command, &preflight_run(layout.execution_parameters()))
            .expect("preflight should inspect configured values");

        assert!(!report.has_failures());
        assert_eq!(
            check_status(&report, "working directory"),
            PreflightStatus::Warning
        );
        assert_eq!(
            check_status(&report, "script file"),
            PreflightStatus::Warning
        );
    }

    #[test]
    fn preflight_fails_when_script_is_missing() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let mut command = layout.command();
        command.args[1] = "missing_script.py".into();

        let report = preflight_openfold(&command, &preflight_run(layout.execution_parameters()))
            .expect("preflight should inspect configured values");

        assert!(report.has_failures());
        assert_eq!(
            check_status(&report, "script file"),
            PreflightStatus::Failed
        );
    }

    #[test]
    fn preflight_fails_when_fasta_dir_is_missing() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let mut execution = layout.execution_parameters();
        execution["fasta_dir"] = json!(layout.root.join("missing-fasta"));

        let report = preflight_openfold(&layout.command(), &preflight_run(execution))
            .expect("preflight should inspect configured values");

        assert!(report.has_failures());
        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Failed);
    }

    #[test]
    fn preflight_fails_when_fasta_tag_does_not_match_input_id() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        fs::write(
            layout.fasta_dir.join("input.fasta"),
            ">1UBQ\nMSTNPKPQRITF\n",
        )
        .expect("mismatched FASTA should be written");

        let report = preflight_openfold(
            &layout.command(),
            &preflight_run(layout.execution_parameters()),
        )
        .expect("preflight should inspect the FASTA tag");

        assert!(report.has_failures());
        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Failed);
        assert!(check_message(&report, "fasta_dir").contains("does not match run input_id"));
    }

    #[test]
    fn preflight_fails_when_fasta_dir_contains_no_fasta_files() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        fs::remove_file(layout.fasta_dir.join("input.fasta"))
            .expect("default FASTA should be removed");

        let report = preflight_openfold(
            &layout.command(),
            &preflight_run(layout.execution_parameters()),
        )
        .expect("preflight should inspect the FASTA directory");

        assert!(report.has_failures());
        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Failed);
        assert!(check_message(&report, "fasta_dir").contains("contains no .fasta or .fa files"));
    }

    #[test]
    fn preflight_passes_when_every_fasta_in_the_directory_is_named_by_input_id() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        fs::write(
            layout.fasta_dir.join("second.fa"),
            ">2OMF_1\nMSTNPKPQRITF\n",
        )
        .expect("second FASTA should be written");

        let report = preflight_openfold(
            &layout.command(),
            &preflight_run_with_input_id("1UBQ_1+2OMF_1", layout.execution_parameters()),
        )
        .expect("preflight should inspect the FASTA directory");

        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Passed);

        let report = preflight_openfold(
            &layout.command(),
            &preflight_run_with_input_id("1UBQ_1+2OMF_1+6KWC_1", layout.execution_parameters()),
        )
        .expect("preflight should inspect the FASTA directory");

        assert!(report.has_failures());
        assert!(check_message(&report, "fasta_dir").contains("missing: '6KWC_1'"));
    }

    /// Every file is checked, not just the first, or the batch folds one target short.
    #[test]
    fn preflight_fails_when_a_later_fasta_holds_multiple_records() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        fs::write(
            layout.fasta_dir.join("second.fa"),
            ">2OMF_1\nMSTNPKPQRITF\n>6KWC_1\nMSTNPKPQRITF\n",
        )
        .expect("second FASTA should be written");

        let report = preflight_openfold(
            &layout.command(),
            &preflight_run_with_input_id("1UBQ_1+2OMF_1", layout.execution_parameters()),
        )
        .expect("preflight should inspect every FASTA in the directory");

        assert!(report.has_failures());
        assert!(check_message(&report, "fasta_dir").contains("multiple FASTA records"));
    }

    /// The staged directory names its links by tag, but a user's own directory need not.
    #[test]
    fn preflight_fails_when_two_fastas_share_a_tag() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        fs::write(
            layout.fasta_dir.join("second.fa"),
            ">1UBQ_1\nMSTNPKPQRITF\n",
        )
        .expect("second FASTA should be written");

        let report = preflight_openfold(
            &layout.command(),
            &preflight_run(layout.execution_parameters()),
        )
        .expect("preflight should inspect the FASTA directory");

        assert!(report.has_failures());
        assert!(check_message(&report, "fasta_dir").contains("distinct tags"));
    }

    #[test]
    fn preflight_passes_when_fasta_dir_is_a_single_file() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let mut execution = layout.execution_parameters();
        execution["fasta_dir"] = json!(layout.fasta_dir.join("input.fasta"));

        let report = preflight_openfold(&layout.command(), &preflight_run(execution))
            .expect("preflight should inspect the FASTA file");

        assert!(!report.has_failures());
        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Passed);
    }

    #[test]
    fn preflight_fails_when_fasta_file_has_no_header() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        fs::write(layout.fasta_dir.join("input.fasta"), "MSTNPKPQRITF\n")
            .expect("headerless FASTA should be written");

        let report = preflight_openfold(
            &layout.command(),
            &preflight_run(layout.execution_parameters()),
        )
        .expect("preflight should inspect the FASTA header");

        assert!(report.has_failures());
        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Failed);
        assert!(check_message(&report, "fasta_dir").contains("not a FASTA header"));
    }

    #[test]
    fn preflight_fails_when_fasta_file_contains_multiple_records() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        fs::write(
            layout.fasta_dir.join("input.fasta"),
            ">1UBQ_1\nMSTNPKPQRITF\n>2OMF_1\nMSTNPKPQRITF\n",
        )
        .expect("multi-record FASTA should be written");

        let report = preflight_openfold(
            &layout.command(),
            &preflight_run(layout.execution_parameters()),
        )
        .expect("preflight should inspect FASTA record count");

        assert!(report.has_failures());
        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Failed);
        assert!(check_message(&report, "fasta_dir").contains("multiple FASTA records"));
    }

    #[test]
    fn preflight_fails_when_data_dir_is_missing() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let mut execution = layout.execution_parameters();
        execution["data_dir"] = json!(layout.root.join("missing-data"));

        let report = preflight_openfold(&layout.command(), &preflight_run(execution))
            .expect("preflight should inspect configured values");

        assert!(report.has_failures());
        assert_eq!(check_status(&report, "data_dir"), PreflightStatus::Failed);
    }

    /// The profile decides the output directory, so a stale execution parameter is not what is checked.
    #[test]
    fn preflight_ignores_an_output_dir_in_execution_parameters() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let mut execution = layout.execution_parameters();
        execution["output_dir"] = json!("/stale/output");

        let report = preflight_openfold(&layout.command(), &preflight_run(execution))
            .expect("preflight should inspect configured values");

        assert_eq!(
            check_status(&report, "output_dir parent"),
            PreflightStatus::Passed
        );
    }

    #[test]
    fn preflight_fails_when_resolved_output_dir_parent_is_missing() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let profile = invocation_profile(
            json!({"output_location": layout.root.join("missing-parent")}).to_string(),
        );

        let report = preflight_openfold_impl(
            &layout.command(),
            &profile,
            &preflight_run(layout.execution_parameters()),
        )
        .expect("preflight should inspect configured values");

        assert!(report.has_failures());
        assert_eq!(
            check_status(&report, "output_dir parent"),
            PreflightStatus::Failed
        );
    }

    /// An unresolvable output location aborts preflight rather than landing as one failed check.
    #[test]
    fn preflight_returns_clear_error_for_missing_output_location() {
        let layout = TestLayout::new("1UBQ_1|Chain A");

        preflight_openfold_impl(
            &layout.command(),
            &invocation_profile("{}".into()),
            &preflight_run(layout.execution_parameters()),
        )
        .expect_err("missing output location should fail preflight");
    }

    #[test]
    fn preflight_fails_when_requested_alignment_dir_is_missing() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let mut execution = layout.execution_parameters();
        execution["use_precomputed_alignments"] = json!(true);

        let report = preflight_openfold(&layout.command(), &preflight_run(execution))
            .expect("preflight should inspect configured values");

        assert!(report.has_failures());
        assert_eq!(
            check_status(&report, "alignment_dir"),
            PreflightStatus::Failed
        );
    }

    #[test]
    fn preflight_passes_when_requested_alignment_dir_exists() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        fs::create_dir_all(layout.alignment_dir.join("1UBQ_1"))
            .expect("alignment key directory should be created");
        let mut execution = layout.execution_parameters();
        execution["use_precomputed_alignments"] = json!(true);
        execution["alignment_dir"] = json!(layout.alignment_dir);

        let report = preflight_openfold(&layout.command(), &preflight_run(execution))
            .expect("preflight should inspect configured values");

        assert!(!report.has_failures());
        assert_eq!(
            check_status(&report, "alignment_dir"),
            PreflightStatus::Passed
        );
        assert_eq!(
            check_status(&report, "precomputed alignment key"),
            PreflightStatus::Passed
        );
    }

    #[test]
    fn preflight_fails_when_precomputed_alignment_key_is_missing() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        fs::create_dir_all(&layout.alignment_dir).expect("alignment directory should be created");
        let mut execution = layout.execution_parameters();
        execution["use_precomputed_alignments"] = json!(true);
        execution["alignment_dir"] = json!(layout.alignment_dir);

        let report = preflight_openfold(&layout.command(), &preflight_run(execution))
            .expect("preflight should inspect the alignment key directory");

        assert!(report.has_failures());
        assert_eq!(
            check_status(&report, "alignment_dir"),
            PreflightStatus::Passed
        );
        assert_eq!(
            check_status(&report, "precomputed alignment key"),
            PreflightStatus::Failed
        );
        assert!(check_message(&report, "precomputed alignment key").contains("1UBQ_1"));
    }

    #[test]
    fn preflight_fails_when_input_id_is_empty() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let report = preflight_openfold(
            &layout.command(),
            &preflight_run_with_input_id("  ", layout.execution_parameters()),
        )
        .expect("preflight should inspect input_id");

        assert!(report.has_failures());
        assert_eq!(check_status(&report, "input_id"), PreflightStatus::Failed);
        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Failed);
    }

    #[test]
    fn preflight_fails_when_program_is_empty() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let mut command = layout.command();
        command.program.clear();

        let report = preflight_openfold(&command, &preflight_run(layout.execution_parameters()))
            .expect("preflight should inspect configured values");

        assert_eq!(
            check_status(&report, "program configured"),
            PreflightStatus::Failed
        );
    }

    fn assert_pair(args: &[String], flag: &str, value: &str) {
        let index = args
            .iter()
            .position(|arg| arg == flag)
            .unwrap_or_else(|| panic!("{flag} should be present"));

        assert_eq!(args[index + 1], value);
    }

    #[test]
    fn gpu_check_passes_when_a_gpu_is_visible() {
        let check = super::gpu_check(Some("NVIDIA A100-SXM4-40GB"));
        assert_eq!(check.status, PreflightStatus::Passed);
        assert!(check.message.unwrap().contains("A100"));
    }

    #[test]
    fn gpu_check_warns_when_no_gpu_is_visible() {
        let check = super::gpu_check(None);
        assert_eq!(check.status, PreflightStatus::Warning);
        assert!(check.message.unwrap().contains("no GPU visible"));
    }
}
