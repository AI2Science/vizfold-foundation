//! The schema-driven command planner, shared by every backend.
//!
//! A backend's invocation profile declares its CLI in JSON; this turns that declaration plus a
//! run's parameters into a CommandSpec. `run_execution` calls it for OpenFold and ESMFold alike --
//! it lived in the OpenFold runner, which is most of why that file was ten times the size of the
//! ESMFold one.

use std::collections::BTreeMap;
use std::path::PathBuf;

use sea_orm::DbErr;
use serde_json::Value;

use crate::core::{
    commands::CommandSpec,
    entities::{execution_targets, model_backends, model_invocation_profiles, runs},
    output_locations::resolve_output_location,
};

pub(crate) fn plan_command(
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

pub(crate) fn validate_entity_consistency(
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

pub(crate) fn append_model_schema_args(
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

pub(crate) fn append_available_resources_args(
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

pub(crate) fn sorted_schema_declarations(
    schema: &Value,
    position_first: bool,
) -> Vec<(&str, &Value)> {
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

pub(crate) fn resolve_declared_value(
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

pub(crate) fn required_invocation_profile_config_string(
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

pub(crate) fn selected_or_default_string(
    parameters: &Value,
    declaration: &Value,
    field_name: &str,
) -> Option<String> {
    parameters
        .get(field_name)
        .or_else(|| declaration.get("default"))
        .and_then(json_value_to_string)
}

pub(crate) fn json_value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

pub(crate) fn validate_execution_parameters_against_available_resources(
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

pub(crate) fn parse_object(field_name: &str, raw: &str) -> Result<Value, DbErr> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|error| DbErr::Custom(format!("{field_name} must be valid JSON: {error}")))?;

    if !value.is_object() {
        return Err(DbErr::Custom(format!("{field_name} must be a JSON object")));
    }

    Ok(value)
}

pub(crate) fn parse_env(config: &Value) -> Result<BTreeMap<String, String>, DbErr> {
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

pub(crate) fn required_string(object: &Value, field_name: &str) -> Result<String, DbErr> {
    optional_string(object, field_name)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| DbErr::Custom(format!("{field_name} is required")))
}

pub(crate) fn optional_string(object: &Value, field_name: &str) -> Option<String> {
    object
        .get(field_name)
        .and_then(Value::as_str)
        .map(str::to_owned)
}

pub(crate) fn optional_bool(object: &Value, field_name: &str) -> Option<bool> {
    object.get(field_name).and_then(Value::as_bool)
}

pub(crate) fn optional_i64(object: &Value, field_name: &str) -> Option<i64> {
    object.get(field_name).and_then(Value::as_i64)
}

pub(crate) fn data_path(data_dir: &str, suffix: &str) -> String {
    format!("{}/{}", data_dir.trim_end_matches(['/', '\\']), suffix)
}
