use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use sea_orm::DbErr;
use serde_json::Value;

use crate::core::{
    commands::CommandSpec,
    entities::{execution_targets, model_backends, model_invocation_profiles, runs},
    output_locations::resolve_output_location,
    preflight::{PreflightCheck, PreflightReport},
};

pub fn plan_openfold_command(
    model_backend: &model_backends::Model,
    execution_target: &execution_targets::Model,
    invocation_profile: &model_invocation_profiles::Model,
    run: &runs::Model,
) -> Result<CommandSpec, DbErr> {
    validate_entity_consistency(invocation_profile)?;

    let config = parse_object(
        "model invocation profile config_json",
        &invocation_profile.config_json,
    )?;
    let model_schema = parse_object(
        "model backend parameter_schema_json",
        &model_backend.parameter_schema_json,
    )?;
    let available_resources = parse_object(
        "execution target available_resources_json",
        &execution_target.available_resources_json,
    )?;
    let model_parameters = parse_object("run model_parameters_json", &run.model_parameters_json)?;
    let execution_parameters = parse_object(
        "run execution_parameters_json",
        &run.execution_parameters_json,
    )?;
    validate_execution_parameters_against_available_resources(
        &available_resources,
        &execution_parameters,
    )?;

    let program = required_string(&config, "program")?;
    let script = required_string(&config, "script")?;
    let current_dir = optional_string(&config, "working_dir").map(PathBuf::from);
    let env = parse_env(&config)?;
    let mut args = vec!["-u".into(), script];

    append_model_schema_args(
        &mut args,
        &model_schema,
        &model_parameters,
        &execution_parameters,
        &config,
        invocation_profile,
        run,
    )?;

    append_available_resources_args(&mut args, &available_resources, &execution_parameters);

    if let Some(residue_idx) = optional_i64(&execution_parameters, "residue_idx") {
        args.extend(["--triangle_residue_idx".into(), residue_idx.to_string()]);
    }

    if optional_bool(&execution_parameters, "use_precomputed_alignments").unwrap_or(false) {
        // TODO: move OpenFold precomputed-alignment handling into a later
        // preflight/flow step that can validate layout before execution.
        let alignment_dir = required_string(&execution_parameters, "alignment_dir")?;
        args.extend(["--use_precomputed_alignments".into(), alignment_dir]);
    }

    // Intentionally do not emit model_preset. The OpenFold script used by this
    // repository currently exposes --config_preset, and model_preset is not part
    // of the MVP OpenFold parameter schema.

    Ok(CommandSpec {
        program,
        args,
        current_dir,
        env,
        stream: false,
    })
}

pub fn preflight_openfold(
    command: &CommandSpec,
    invocation_profile: &model_invocation_profiles::Model,
    run: &runs::Model,
) -> Result<PreflightReport, DbErr> {
    let execution_parameters = parse_object(
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

/// Mirrors backends/openfold/fold.sh's `nvidia-smi --query-gpu=name --format=csv,noheader` probe.
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

/// Program/script-argument/working-directory/script-file checks shared by every local_subprocess
/// backend's preflight (OpenFold, ESMFold): they validate the planned command itself, independent
/// of a backend's inputs. Callers push these after `gpu_check` and before their own input checks.
pub(super) fn base_command_checks(command: &CommandSpec) -> Vec<PreflightCheck> {
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

pub(super) fn path_exists_check(name: &str, path: &Path) -> PreflightCheck {
    if path.exists() {
        PreflightCheck::passed(name, format!("'{}' exists", path.display()))
    } else {
        PreflightCheck::failed(name, format!("'{}' does not exist", path.display()))
    }
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

pub(super) fn input_id_check(input_id: &str) -> PreflightCheck {
    if input_id.trim().is_empty() {
        PreflightCheck::failed("input_id", "run input_id is missing or empty")
    } else {
        PreflightCheck::passed(
            "input_id",
            format!("run input_id '{input_id}' is configured"),
        )
    }
}

fn fasta_input_check(parameters: &Value, input_id: &str) -> PreflightCheck {
    let Some(fasta_dir) = optional_string(parameters, "fasta_dir").filter(|path| !path.is_empty())
    else {
        return PreflightCheck::failed("fasta_dir", "fasta_dir is missing");
    };

    let fasta_dir = Path::new(&fasta_dir);
    if !fasta_dir.is_dir() {
        return PreflightCheck::failed(
            "fasta_dir",
            format!(
                "'{}' does not exist or is not a directory",
                fasta_dir.display()
            ),
        );
    }

    let fasta_files = match fasta_files_in_directory(fasta_dir) {
        Ok(files) => files,
        Err(error) => {
            return PreflightCheck::failed(
                "fasta_dir",
                format!("could not inspect '{}': {error}", fasta_dir.display()),
            );
        }
    };

    if fasta_files.is_empty() {
        return PreflightCheck::failed(
            "fasta_dir",
            format!("'{}' contains no .fasta or .fa files", fasta_dir.display()),
        );
    }

    if fasta_files.len() != 1 {
        return PreflightCheck::failed(
            "fasta_dir",
            format!(
                "'{}' must contain exactly one .fasta or .fa file, found {}",
                fasta_dir.display(),
                fasta_files.len()
            ),
        );
    }

    let fasta_path = &fasta_files[0];
    let tag = match parse_single_fasta_tag(fasta_path) {
        Ok(tag) => tag,
        Err(error) => {
            return PreflightCheck::failed(
                "fasta_dir",
                format!(
                    "'{}' is not a valid single-record FASTA: {error}",
                    fasta_path.display()
                ),
            );
        }
    };

    if input_id.trim().is_empty() {
        return PreflightCheck::failed(
            "fasta_dir",
            "cannot validate FASTA identity because run input_id is missing or empty",
        );
    }

    if tag != input_id {
        return PreflightCheck::failed(
            "fasta_dir",
            format!("FASTA tag '{tag}' does not match run input_id '{input_id}'"),
        );
    }

    PreflightCheck::passed(
        "fasta_dir",
        format!(
            "'{}' contains one FASTA file with tag '{tag}' matching run input_id",
            fasta_dir.display()
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

    let key_directory = alignment_dir.join(input_id);
    if key_directory.is_dir() {
        PreflightCheck::passed(
            "precomputed alignment key",
            format!("'{}' exists", key_directory.display()),
        )
    } else {
        PreflightCheck::failed(
            "precomputed alignment key",
            format!(
                "expected alignment directory '{}' is missing",
                key_directory.display()
            ),
        )
    }
}

pub(super) fn output_dir_check(output_path: &Path) -> PreflightCheck {
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

fn validate_entity_consistency(
    invocation_profile: &model_invocation_profiles::Model,
) -> Result<(), DbErr> {
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
    invocation_config: &Value,
    invocation_profile: &model_invocation_profiles::Model,
    run: &runs::Model,
) -> Result<(), DbErr> {
    for (_name, declaration) in sorted_schema_declarations(model_schema, true) {
        if optional_bool(declaration, "positional").unwrap_or(false) {
            args.push(resolve_declared_value(
                declaration,
                model_parameters,
                execution_parameters,
                invocation_config,
                invocation_profile,
                run,
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
            let value = resolve_declared_value(
                declaration,
                model_parameters,
                execution_parameters,
                invocation_config,
                invocation_profile,
                run,
            )?;
            args.extend([cli_flag, value]);
            continue;
        }

        if let Some(value) = selected_or_default_string(model_parameters, declaration, name) {
            args.extend([cli_flag, value]);
        }
    }

    Ok(())
}

fn append_available_resources_args(
    args: &mut Vec<String>,
    available_resources: &Value,
    execution_parameters: &Value,
) {
    for (name, declaration) in sorted_schema_declarations(available_resources, false) {
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
    _model_parameters: &Value,
    execution_parameters: &Value,
    invocation_config: &Value,
    invocation_profile: &model_invocation_profiles::Model,
    run: &runs::Model,
) -> Result<String, DbErr> {
    let source = required_string(declaration, "source")?;
    match source.as_str() {
        "data_dir" => {
            let data_dir = required_string(execution_parameters, "data_dir")?;
            let relative_path = required_string(declaration, "relative_path")?;
            Ok(data_path(&data_dir, &relative_path))
        }
        "execution_parameters" => {
            let parameter_name = required_string(declaration, "parameter")?;
            required_string(execution_parameters, &parameter_name)
        }
        "invocation_profile_config" => {
            let parameter_name = required_string(declaration, "parameter")?;
            let value =
                required_invocation_profile_config_string(invocation_config, &parameter_name)?;
            let mut path = PathBuf::from(value);
            if let Some(relative_path) = optional_string(declaration, "relative_path") {
                path.push(relative_path);
            }
            Ok(path.to_string_lossy().into_owned())
        }
        "run_output_workspace" => {
            let workspace = resolve_output_location(invocation_profile, run)?;
            let path = optional_string(declaration, "relative_path")
                .map(|relative_path| workspace.join(relative_path))
                .unwrap_or(workspace);
            Ok(path.to_string_lossy().into_owned())
        }
        _ => Err(DbErr::Custom(format!(
            "unsupported model parameter source '{source}'"
        ))),
    }
}

fn required_invocation_profile_config_string(
    config: &Value,
    parameter_name: &str,
) -> Result<String, DbErr> {
    let Some(value) = config.get(parameter_name) else {
        return Err(DbErr::Custom(format!(
            "invocation profile config '{parameter_name}' is required"
        )));
    };
    let Some(value) = value.as_str().filter(|value| !value.trim().is_empty()) else {
        return Err(DbErr::Custom(format!(
            "invocation profile config '{parameter_name}' must be a non-empty string"
        )));
    };

    Ok(value.to_owned())
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

fn validate_execution_parameters_against_available_resources(
    available_resources: &Value,
    execution_parameters: &Value,
) -> Result<(), DbErr> {
    let Some(properties) = available_resources
        .get("properties")
        .and_then(Value::as_object)
    else {
        return Ok(());
    };

    // A present field is validated against its raw JSON type, not skipped when the
    // type is wrong: emission (json_value_to_string) stringifies numbers/bools, so a
    // wrong-typed value would otherwise bypass the enum/range guard yet still be emitted.
    if let Some(declaration) = properties.get("model_device")
        && let Some(value) = execution_parameters.get("model_device")
    {
        let model_device = value
            .as_str()
            .ok_or_else(|| DbErr::Custom("model_device must be a string".into()))?;
        if let Some(allowed_values) = declaration.get("enum").and_then(Value::as_array) {
            let allowed = allowed_values
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();

            if !allowed.contains(&model_device) {
                return Err(DbErr::Custom(format!(
                    "model_device '{model_device}' is not allowed by execution target available resources"
                )));
            }
        }
    }

    if let Some(declaration) = properties.get("cpus")
        && let Some(value) = execution_parameters.get("cpus")
    {
        let cpus = value
            .as_i64()
            .ok_or_else(|| DbErr::Custom("cpus must be an integer".into()))?;
        if let Some(minimum) = optional_i64(declaration, "minimum")
            && cpus < minimum
        {
            return Err(DbErr::Custom(format!(
                "cpus {cpus} is below execution target resource minimum {minimum}"
            )));
        }

        if let Some(maximum) = optional_i64(declaration, "maximum")
            && cpus > maximum
        {
            return Err(DbErr::Custom(format!(
                "cpus {cpus} exceeds execution target resource maximum {maximum}"
            )));
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
    use std::{env, fs, path::PathBuf};

    use chrono::Utc;
    use sea_orm::DbErr;
    use serde_json::json;

    use crate::core::{
        commands::CommandSpec,
        entities::{execution_targets, model_backends, model_invocation_profiles, runs},
        preflight::{PreflightReport, PreflightStatus},
        test_support::TestLayout,
    };

    use super::{
        plan_openfold_command, preflight_openfold as preflight_openfold_impl,
        resolve_declared_value,
    };

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

    fn check_status(report: &PreflightReport, name: &str) -> PreflightStatus {
        report
            .checks
            .iter()
            .find(|check| check.name == name)
            .unwrap_or_else(|| panic!("{name} check should be present"))
            .status
    }

    fn check_message<'a>(report: &'a PreflightReport, name: &str) -> &'a str {
        report
            .checks
            .iter()
            .find(|check| check.name == name)
            .unwrap_or_else(|| panic!("{name} check should be present"))
            .message
            .as_deref()
            .unwrap_or_else(|| panic!("{name} check should have a message"))
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
        assert_eq!(command.args[2], "/tmp/fasta");
        assert_eq!(command.args[3], "/data/pdb_mmcif/mmcif_files");
        assert!(command.args.contains(&"--output_dir".into()));
        let output_dir = PathBuf::from("/tmp/outputs").join("4");
        assert!(
            command
                .args
                .contains(&output_dir.to_string_lossy().into_owned())
        );
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

        let command = plan_openfold_command(
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
        let error = plan_openfold_command(
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

        assert_eq!(
            value,
            PathBuf::from("/profile/data")
                .join("datasets/openfold")
                .to_string_lossy()
        );
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
    fn available_resources_flags_come_from_execution_parameters() {
        // (execution_parameters field, override value, model_parameters, expected flag, expected value)
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

            let command = plan_openfold_command(
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
    fn rejects_cpus_above_available_resources_maximum() {
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
    fn rejects_wrong_typed_cpus_that_would_bypass_range_guard() {
        let mut execution = execution_parameters();
        execution["cpus"] = json!("999");

        let error = plan_openfold_command(
            &model_backend(),
            &execution_target(),
            &invocation_profile(config()),
            &run(json!({}).to_string(), execution.to_string()),
        )
        .expect_err("string cpus must not bypass the integer range guard");

        assert!(error.to_string().contains("cpus must be an integer"));
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
    fn derives_attention_map_directory_from_resolved_output_location() {
        let mut execution = execution_parameters();
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

        assert_pair(
            &command.args,
            "--attn_map_dir",
            &PathBuf::from("/tmp/outputs")
                .join("4")
                .join("attention")
                .to_string_lossy(),
        );
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
        let error = plan_openfold_command(
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
    fn preflight_fails_when_fasta_dir_contains_multiple_fasta_files() {
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
        assert_eq!(check_status(&report, "fasta_dir"), PreflightStatus::Failed);
        assert!(check_message(&report, "fasta_dir").contains("exactly one .fasta or .fa file"));
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

    #[test]
    fn preflight_does_not_require_output_dir_in_execution_parameters() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let report = preflight_openfold(
            &layout.command(),
            &preflight_run(layout.execution_parameters()),
        )
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

    #[test]
    fn preflight_returns_clear_error_for_missing_output_location() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let error = preflight_openfold_impl(
            &layout.command(),
            &invocation_profile("{}".into()),
            &preflight_run(layout.execution_parameters()),
        )
        .expect_err("missing output location should fail preflight");

        assert!(error.to_string().contains("output_location is required"));
    }

    #[test]
    fn preflight_returns_clear_error_for_invalid_output_location() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let error = preflight_openfold_impl(
            &layout.command(),
            &invocation_profile(json!({"output_location": 42}).to_string()),
            &preflight_run(layout.execution_parameters()),
        )
        .expect_err("non-string output location should fail preflight");

        assert!(
            error
                .to_string()
                .contains("output_location must be a string")
        );
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
    fn preflight_report_has_failures_tracks_failed_checks() {
        let layout = TestLayout::new("1UBQ_1|Chain A");
        let mut command = layout.command();
        command.program.clear();

        let report = preflight_openfold(&command, &preflight_run(layout.execution_parameters()))
            .expect("preflight should inspect configured values");

        assert!(report.has_failures());
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
