use std::{collections::BTreeMap, path::PathBuf};

use sea_orm::DbErr;
use serde_json::Value;

use crate::core::{
    commands::CommandSpec,
    entities::{execution_targets, model_backends, model_invocation_profiles, runs},
};

pub fn plan_openfold_command(
    model_backend: &model_backends::Model,
    execution_target: &execution_targets::Model,
    invocation_profile: &model_invocation_profiles::Model,
    run: &runs::Model,
) -> Result<CommandSpec, DbErr> {
    validate_entity_consistency(model_backend, execution_target, invocation_profile, run)?;

    let config = parse_object(
        "model invocation profile config_json",
        &invocation_profile.config_json,
    )?;
    let model_schema = parse_object(
        "model backend parameter_schema_json",
        &model_backend.parameter_schema_json,
    )?;
    let execution_schema = parse_object(
        "execution target parameter_schema_json",
        &execution_target.parameter_schema_json,
    )?;
    let model_parameters = parse_object("run model_parameters_json", &run.model_parameters_json)?;
    let execution_parameters = parse_object(
        "run execution_parameters_json",
        &run.execution_parameters_json,
    )?;
    validate_execution_parameters(&execution_schema, &execution_parameters)?;

    let program = required_string(&config, "program")?;
    let script = required_string(&config, "script")?;
    let current_dir = optional_string(&config, "working_dir").map(PathBuf::from);
    let env = parse_env(&config)?;

    let fasta_dir = required_string(&execution_parameters, "fasta_dir")?;
    let output_dir = required_string(&execution_parameters, "output_dir")?;

    let mut args = vec!["-u".into(), script, fasta_dir];

    append_model_schema_args(
        &mut args,
        &model_schema,
        &model_parameters,
        &execution_parameters,
    )?;

    args.extend(["--output_dir".into(), output_dir]);
    append_execution_schema_args(&mut args, &execution_schema, &execution_parameters);

    if let Some(attn_map_dir) = optional_string(&execution_parameters, "attn_map_dir") {
        args.extend(["--attn_map_dir".into(), attn_map_dir]);
    }

    if let Some(residue_idx) = optional_i64(&execution_parameters, "residue_idx") {
        args.extend(["--triangle_residue_idx".into(), residue_idx.to_string()]);
    }

    if optional_bool(&execution_parameters, "use_precomputed_alignments").unwrap_or(false) {
        // TODO: move OpenFold precomputed-alignment handling into a later
        // preflight/flow step that can validate layout before execution.
        let alignment_dir = required_string(&execution_parameters, "alignment_dir")?;
        args.extend(["--use_precomputed_alignments".into(), alignment_dir]);
    }

    // The OpenFold script in this repository exposes --config_preset, but not
    // --model_preset. Keep model_preset out of the resolved command for now.

    Ok(CommandSpec {
        program,
        args,
        current_dir,
        env,
    })
}

fn validate_entity_consistency(
    model_backend: &model_backends::Model,
    execution_target: &execution_targets::Model,
    invocation_profile: &model_invocation_profiles::Model,
    run: &runs::Model,
) -> Result<(), DbErr> {
    if run.model_backend_id != model_backend.id {
        return Err(DbErr::Custom(
            "run model_backend_id does not match loaded model backend".into(),
        ));
    }

    if run.execution_target_id != execution_target.id {
        return Err(DbErr::Custom(
            "run execution_target_id does not match loaded execution target".into(),
        ));
    }

    if run.invocation_profile_id != invocation_profile.id {
        return Err(DbErr::Custom(
            "run invocation_profile_id does not match loaded invocation profile".into(),
        ));
    }

    if invocation_profile.model_backend_id != model_backend.id
        || invocation_profile.execution_target_id != execution_target.id
    {
        return Err(DbErr::Custom(
            "model invocation profile does not match loaded model backend and execution target"
                .into(),
        ));
    }

    if invocation_profile.invocation_kind != "local_subprocess" {
        return Err(DbErr::Custom(format!(
            "OpenFold planner only supports local_subprocess invocation profiles, got '{}'",
            invocation_profile.invocation_kind
        )));
    }

    Ok(())
}

fn append_model_schema_args(
    args: &mut Vec<String>,
    model_schema: &Value,
    model_parameters: &Value,
    execution_parameters: &Value,
) -> Result<(), DbErr> {
    for (_name, declaration) in sorted_schema_declarations(model_schema, true) {
        if optional_bool(declaration, "positional").unwrap_or(false) {
            args.push(resolve_declared_value(
                declaration,
                model_parameters,
                execution_parameters,
            )?);
        }
    }

    for (name, declaration) in sorted_schema_declarations(model_schema, false) {
        if optional_bool(declaration, "positional").unwrap_or(false) {
            continue;
        }

        let Some(cli_flag) = optional_string(declaration, "cli_flag") else {
            continue;
        };

        if optional_string(declaration, "type").as_deref() == Some("boolean") {
            if optional_bool(model_parameters, name).unwrap_or(false) {
                args.push(cli_flag);
            }
            continue;
        }

        if declaration.get("source").is_some() {
            let value =
                resolve_declared_value(declaration, model_parameters, execution_parameters)?;
            args.extend([cli_flag, value]);
            continue;
        }

        if let Some(value) = selected_or_default_string(model_parameters, declaration, name) {
            args.extend([cli_flag, value]);
        }
    }

    Ok(())
}

fn append_execution_schema_args(
    args: &mut Vec<String>,
    execution_schema: &Value,
    execution_parameters: &Value,
) {
    for (name, declaration) in sorted_schema_declarations(execution_schema, false) {
        let Some(cli_flag) = optional_string(declaration, "cli_flag") else {
            continue;
        };

        if let Some(value) = selected_or_default_string(execution_parameters, declaration, name) {
            args.extend([cli_flag, value]);
        }
    }
}

fn sorted_schema_declarations(schema: &Value, position_first: bool) -> Vec<(&str, &Value)> {
    let mut declarations = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .iter()
                .map(|(name, declaration)| (name.as_str(), declaration))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    declarations.sort_by(|(left_name, left), (right_name, right)| {
        let left_position = optional_i64(left, "position").unwrap_or(i64::MAX);
        let right_position = optional_i64(right, "position").unwrap_or(i64::MAX);
        if position_first {
            left_position
                .cmp(&right_position)
                .then_with(|| left_name.cmp(right_name))
        } else {
            left_name.cmp(right_name)
        }
    });

    declarations
}

fn resolve_declared_value(
    declaration: &Value,
    model_parameters: &Value,
    execution_parameters: &Value,
) -> Result<String, DbErr> {
    if optional_string(declaration, "source").as_deref() == Some("data_dir") {
        let data_dir = required_string(execution_parameters, "data_dir")?;
        let relative_path = required_string(declaration, "relative_path")?;
        return Ok(data_path(&data_dir, &relative_path));
    }

    let name = required_string(declaration, "name")?;
    selected_or_default_string(model_parameters, declaration, &name)
        .ok_or_else(|| DbErr::Custom(format!("{name} is required")))
}

fn selected_or_default_string(
    parameters: &Value,
    declaration: &Value,
    field_name: &str,
) -> Option<String> {
    parameters
        .get(field_name)
        .or_else(|| declaration.get("default"))
        .and_then(json_value_to_string)
}

fn json_value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn validate_execution_parameters(
    execution_schema: &Value,
    execution_parameters: &Value,
) -> Result<(), DbErr> {
    let Some(properties) = execution_schema
        .get("properties")
        .and_then(Value::as_object)
    else {
        return Ok(());
    };

    if let Some(declaration) = properties.get("model_device") {
        if let Some(model_device) = optional_string(execution_parameters, "model_device") {
            if let Some(allowed_values) = declaration.get("enum").and_then(Value::as_array) {
                let allowed = allowed_values
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>();

                if !allowed.contains(&model_device.as_str()) {
                    return Err(DbErr::Custom(format!(
                        "model_device '{model_device}' is not allowed by execution target schema"
                    )));
                }
            }
        }
    }

    if let Some(cpus) = optional_i64(execution_parameters, "cpus") {
        if let Some(declaration) = properties.get("cpus") {
            if let Some(minimum) = optional_i64(declaration, "minimum") {
                if cpus < minimum {
                    return Err(DbErr::Custom(format!(
                        "cpus {cpus} is below execution target minimum {minimum}"
                    )));
                }
            }

            if let Some(maximum) = optional_i64(declaration, "maximum") {
                if cpus > maximum {
                    return Err(DbErr::Custom(format!(
                        "cpus {cpus} exceeds execution target maximum {maximum}"
                    )));
                }
            }
        }
    }

    Ok(())
}

fn parse_object(field_name: &str, raw: &str) -> Result<Value, DbErr> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|error| DbErr::Custom(format!("{field_name} must be valid JSON: {error}")))?;

    if !value.is_object() {
        return Err(DbErr::Custom(format!("{field_name} must be a JSON object")));
    }

    Ok(value)
}

fn parse_env(config: &Value) -> Result<BTreeMap<String, String>, DbErr> {
    let Some(env) = config.get("env") else {
        return Ok(BTreeMap::new());
    };

    let Some(env_object) = env.as_object() else {
        return Err(DbErr::Custom("config env must be a JSON object".into()));
    };

    let mut parsed = BTreeMap::new();
    for (key, value) in env_object {
        let Some(value) = value.as_str() else {
            return Err(DbErr::Custom(format!(
                "config env value for '{key}' must be a string"
            )));
        };
        parsed.insert(key.clone(), value.to_owned());
    }

    Ok(parsed)
}

fn required_string(object: &Value, field_name: &str) -> Result<String, DbErr> {
    optional_string(object, field_name)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| DbErr::Custom(format!("{field_name} is required")))
}

fn optional_string(object: &Value, field_name: &str) -> Option<String> {
    object
        .get(field_name)
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn optional_bool(object: &Value, field_name: &str) -> Option<bool> {
    object.get(field_name).and_then(Value::as_bool)
}

fn optional_i64(object: &Value, field_name: &str) -> Option<i64> {
    object.get(field_name).and_then(Value::as_i64)
}

fn data_path(data_dir: &str, suffix: &str) -> String {
    format!("{}/{}", data_dir.trim_end_matches(['/', '\\']), suffix)
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use serde_json::json;

    use crate::core::entities::{
        execution_targets, model_backends, model_invocation_profiles, runs,
    };

    use super::plan_openfold_command;

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
            parameter_schema_json: execution_parameter_schema().to_string(),
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
            parameter_schema_json: "{}".into(),
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
            input_sequence: "MSTNPKPQRITF".into(),
            model_parameters_json,
            execution_parameters_json,
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
            "output_dir": "/tmp/output",
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

    fn execution_parameter_schema() -> serde_json::Value {
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
        let command = plan_openfold_command(
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
        assert!(command.args.contains(&"/tmp/fasta".into()));
        assert!(command.args.contains(&"/data/pdb_mmcif/mmcif_files".into()));
        assert!(command.args.contains(&"--output_dir".into()));
        assert!(command.args.contains(&"/tmp/output".into()));
        assert!(command.args.contains(&"--config_preset".into()));
        assert!(command.args.contains(&"model_1_ptm".into()));
        assert!(command.args.contains(&"--model_device".into()));
        assert!(command.args.contains(&"cuda:0".into()));
    }

    #[test]
    fn includes_optional_model_parameters_when_present() {
        let mut execution = execution_parameters();
        execution["model_device"] = json!("cpu");

        let command = plan_openfold_command(
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
                execution.to_string(),
            ),
        )
        .expect("command should plan");

        assert!(command.args.contains(&"model_2_ptm".into()));
        assert!(command.args.contains(&"cpu".into()));
        assert!(command.args.contains(&"--save_outputs".into()));
        assert!(command.args.contains(&"--num_recycles_save".into()));
        assert!(command.args.contains(&"1".into()));
        assert!(!command.args.contains(&"--model_preset".into()));
    }

    #[test]
    fn model_device_comes_from_execution_parameters() {
        let mut execution = execution_parameters();
        execution["model_device"] = json!("cpu");

        let command = plan_openfold_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(
                json!({"model_device": "cuda:0"}).to_string(),
                execution.to_string(),
            ),
        )
        .expect("command should plan");

        let model_device_index = command
            .args
            .iter()
            .position(|arg| arg == "--model_device")
            .expect("model device flag should be present");

        assert_eq!(command.args[model_device_index + 1], "cpu");
    }

    #[test]
    fn cpus_comes_from_execution_parameters() {
        let mut execution = execution_parameters();
        execution["cpus"] = json!(12);

        let command = plan_openfold_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect("command should plan");

        let cpus_index = command
            .args
            .iter()
            .position(|arg| arg == "--cpus")
            .expect("cpus flag should be present");

        assert_eq!(command.args[cpus_index + 1], "12");
    }

    #[test]
    fn rejects_invalid_model_device_from_execution_schema_enum() {
        let mut execution = execution_parameters();
        execution["model_device"] = json!("cuda:1");

        let error = plan_openfold_command(
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
    fn rejects_cpus_above_execution_schema_maximum() {
        let mut execution = execution_parameters();
        execution["cpus"] = json!(15);

        let error = plan_openfold_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect_err("too many cpus should fail");

        assert!(error.to_string().contains("cpus 15 exceeds"));
    }

    #[test]
    fn database_paths_are_generated_from_model_schema_declarations() {
        let command = plan_openfold_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution_parameters().to_string()),
        )
        .expect("command should plan");

        assert_pair(
            &command.args,
            "--uniref90_database_path",
            "/data/uniref90/uniref90.fasta",
        );
        assert_pair(
            &command.args,
            "--bfd_database_path",
            "/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
        );
    }

    #[test]
    fn includes_attention_demo_flags_when_enabled() {
        let mut execution = execution_parameters();
        execution["attn_map_dir"] = json!("/tmp/attn");
        execution["residue_idx"] = json!(7);

        let command = plan_openfold_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(
                json!({"demo_attn": true}).to_string(),
                execution.to_string(),
            ),
        )
        .expect("command should plan");

        assert!(command.args.contains(&"--attn_map_dir".into()));
        assert!(command.args.contains(&"/tmp/attn".into()));
        assert!(command.args.contains(&"--triangle_residue_idx".into()));
        assert!(command.args.contains(&"7".into()));
        assert!(command.args.contains(&"--demo_attn".into()));
    }

    #[test]
    fn includes_precomputed_alignment_flags_when_requested() {
        let mut execution = execution_parameters();
        execution["use_precomputed_alignments"] = json!(true);
        execution["alignment_dir"] = json!("/tmp/alignments");

        let command = plan_openfold_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect("command should plan");

        assert!(
            command
                .args
                .contains(&"--use_precomputed_alignments".into())
        );
        assert!(command.args.contains(&"/tmp/alignments".into()));
    }

    #[test]
    fn rejects_missing_alignment_dir_when_precomputed_alignments_requested() {
        let mut execution = execution_parameters();
        execution["use_precomputed_alignments"] = json!(true);

        let error = plan_openfold_command(
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
        let error = plan_openfold_command(
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
        let error = plan_openfold_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(
                json!({}).to_string(),
                json!({
                    "fasta_dir": "/tmp/fasta",
                    "output_dir": "/tmp/output"
                })
                .to_string(),
            ),
        )
        .expect_err("missing data_dir should fail");

        assert!(error.to_string().contains("data_dir is required"));
    }

    #[test]
    fn validates_env_is_string_to_string_object() {
        let error = plan_openfold_command(
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

        let error = plan_openfold_command(
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

    fn assert_pair(args: &[String], flag: &str, value: &str) {
        let index = args
            .iter()
            .position(|arg| arg == flag)
            .unwrap_or_else(|| panic!("{flag} should be present"));

        assert_eq!(args[index + 1], value);
    }
}
