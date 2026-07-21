use std::{env, path::PathBuf};

use chrono::Utc;
use executor::core::{
    commands::LocalCommandRunner,
    entities::{execution_targets, model_backends, model_invocation_profiles, runs},
    execution::execute_command_workflow,
    model_runners::openfold::{OpenFoldPreflightRunner, plan_openfold_command},
    preflight::PreflightStatus,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), sea_orm::DbErr> {
    let now = Utc::now();
    let paths = DemoPaths::from_environment();
    print_demo_configuration(&paths);

    let model_backend = model_backend(now);
    let execution_target = execution_target(now);
    let invocation_profile = invocation_profile(now, &paths);
    let run = run(now, &paths);

    let command =
        plan_openfold_command(&model_backend, &execution_target, &invocation_profile, &run)?;
    print_command(&command);

    let preflight_runner = OpenFoldPreflightRunner {
        command: &command,
        run: &run,
    };
    let workflow_result =
        execute_command_workflow(&command, &LocalCommandRunner, Some(&preflight_runner)).await?;

    print_workflow_result(&workflow_result);
    Ok(())
}

struct DemoPaths {
    demo_run_id: String,
    input_id: String,
    model_device: String,
    attn_map_dir: String,
    residue_idx: i64,
    demo_attn: bool,
    working_dir: String,
    data_dir: String,
    fasta_dir: String,
    output_dir: String,
    alignment_dir: String,
}

impl DemoPaths {
    fn from_environment() -> Self {
        let repository_root = repository_root();
        let input_id = env_or_demo_value("VIZFOLD_OPENFOLD_INPUT_ID", "6KWC_1");
        let residue_idx = env_or_demo_i64("VIZFOLD_OPENFOLD_RESIDUE_IDX", 1);
        let demo_run_id = env_or_demo_value("VIZFOLD_OPENFOLD_DEMO_RUN_ID", "openfold-demo-run");
        let output_dir = env_or_demo_path(
            "VIZFOLD_OPENFOLD_OUTPUT_DIR",
            repository_root
                .join("science-gateway")
                .join("openfold-demo-output")
                .join(&demo_run_id),
        );
        Self {
            attn_map_dir: env_or_demo_path(
                "VIZFOLD_OPENFOLD_ATTN_MAP_DIR",
                PathBuf::from(&output_dir)
                    .join(format!("attention_files_{input_id}_demo_tri_{residue_idx}")),
            ),
            demo_attn: env_or_demo_bool("VIZFOLD_OPENFOLD_DEMO_ATTN", true),
            demo_run_id,
            input_id,
            model_device: env_or_demo_value("VIZFOLD_OPENFOLD_MODEL_DEVICE", "cuda:0"),
            residue_idx,
            working_dir: env_or_demo_path(
                "VIZFOLD_OPENFOLD_WORKING_DIR",
                repository_root.as_path(),
            ),
            data_dir: env_or_demo_path(
                "VIZFOLD_OPENFOLD_DATA_DIR",
                PathBuf::from("/tmp/vizfold-demo/data"),
            ),
            fasta_dir: env_or_demo_path(
                "VIZFOLD_OPENFOLD_FASTA_DIR",
                repository_root
                    .join("examples")
                    .join("monomer")
                    .join("fasta_dir_6KWC"),
            ),
            output_dir,
            alignment_dir: env_or_demo_path(
                "VIZFOLD_OPENFOLD_ALIGNMENT_DIR",
                repository_root
                    .join("examples")
                    .join("monomer")
                    .join("alignments"),
            ),
        }
    }
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("executor manifest should be nested under the repository root")
        .to_path_buf()
}

fn env_or_demo_path(name: &str, default: impl Into<PathBuf>) -> String {
    env::var(name).unwrap_or_else(|_| default.into().display().to_string())
}

fn env_or_demo_value(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.into())
}

fn env_or_demo_i64(name: &str, default: i64) -> i64 {
    match env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer")),
        Err(_) => default,
    }
}

fn env_or_demo_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => panic!("{name} must be true, false, 1, or 0"),
        },
        Err(_) => default,
    }
}

fn model_backend(now: chrono::DateTime<Utc>) -> model_backends::Model {
    model_backends::Model {
        id: 1,
        slug: "openfold".into(),
        label: "OpenFold".into(),
        version: Some("example".into()),
        description: Some("Self-contained workflow example".into()),
        artifact_capabilities_json: "{}".into(),
        parameter_schema_json: json!({
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
                "save_outputs": { "type": "boolean", "cli_flag": "--save_outputs" },
                "demo_attn": { "type": "boolean", "cli_flag": "--demo_attn" },
                "num_recycles_save": { "type": "integer", "cli_flag": "--num_recycles_save" }
            }
        })
        .to_string(),
        created_at: now,
        updated_at: now,
    }
}

fn execution_target(now: chrono::DateTime<Utc>) -> execution_targets::Model {
    execution_targets::Model {
        id: 2,
        slug: "local".into(),
        target_type: "local".into(),
        description: Some("Local example target".into()),
        available_resources_json: json!({
            "type": "object",
            "properties": {
                "model_device": {
                    "type": "string",
                    "enum": ["cpu", "cuda:0"],
                    "default": "cpu",
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
        .to_string(),
        created_at: now,
        updated_at: now,
    }
}

fn invocation_profile(
    now: chrono::DateTime<Utc>,
    paths: &DemoPaths,
) -> model_invocation_profiles::Model {
    model_invocation_profiles::Model {
        id: 3,
        model_backend_id: 1,
        execution_target_id: 2,
        invocation_kind: "local_subprocess".into(),
        config_json: json!({
            "program": "python3",
            "script": "run_pretrained_openfold.py",
            "working_dir": paths.working_dir,
        })
        .to_string(),
        parameter_schema_json: "{}".into(),
        created_at: now,
        updated_at: now,
    }
}

fn run(now: chrono::DateTime<Utc>, paths: &DemoPaths) -> runs::Model {
    runs::Model {
        id: 4,
        model_backend_id: 1,
        execution_target_id: 2,
        invocation_profile_id: 3,
        status: "submitted".into(),
        input_id: paths.input_id.clone(),
        input_sequence: "GSTIQPGTGYNNGYFYSYWNDGHGGVTYTNGPGGQFSVNWSNSGEFVGGKGWQPGTKNKVINFSGSYNPNGNSYLSVYGWSRNPLIEYYIVENFGTYNPSTGATKLGEVTSDGSVYDIYRTQRVNQPSIIGTATFYQYWSVRRNHRSSGSVNTANHFNAWAQQGLTLGTMDYQIVAVQGYFSSGSASITVS".into(),
        model_parameters_json: json!({
            "config_preset": "model_1_ptm",
            "save_outputs": true,
            "num_recycles_save": 1,
            "demo_attn": paths.demo_attn,
        })
        .to_string(),
        // Reuse requires an existing `<alignment_dir>/<input_id>` directory.
        execution_parameters_json: json!({
            "fasta_dir": paths.fasta_dir,
            "data_dir": paths.data_dir,
            "output_dir": paths.output_dir,
            "alignment_dir": paths.alignment_dir,
            "attn_map_dir": paths.attn_map_dir,
            "residue_idx": paths.residue_idx,
            "use_precomputed_alignments": true,
            "model_device": paths.model_device,
            "cpus": 1,
        })
        .to_string(),
        submitted_at: now,
        started_at: None,
        completed_at: None,
        error_message: None,
    }
}

fn print_demo_configuration(paths: &DemoPaths) {
    println!("== Demo configuration ==");
    println!("input_id: {}", paths.input_id);
    println!("model_device: {}", paths.model_device);
    println!("demo_run_id: {}", paths.demo_run_id);
    println!("output_dir: {}", paths.output_dir);
    println!("attn_map_dir: {}", paths.attn_map_dir);
    println!("triangle_residue_idx: {}", paths.residue_idx);
    println!("demo_attn: {}", paths.demo_attn);
}

fn print_command(command: &executor::core::commands::CommandSpec) {
    println!("== Planned command ==");
    println!("program: {}", command.program);
    println!(
        "working directory: {}",
        display_path(command.current_dir.as_deref())
    );
    println!("args:");
    for argument in &command.args {
        println!("  {argument}");
    }
}

fn print_workflow_result(result: &executor::core::execution::ExecutionWorkflowResult) {
    println!("\n== Preflight results ==");
    match &result.preflight_report {
        Some(report) => {
            for check in &report.checks {
                let message = check.message.as_deref().unwrap_or("no details");
                println!(
                    "[{}] {}: {}",
                    status_label(check.status),
                    check.name,
                    message
                );
            }
        }
        None => println!("No preflight runner was provided."),
    }

    if let Some(reason) = &result.skipped_execution_reason {
        println!("\n== Execution skipped ==\n{reason}");
    }

    if let Some(output) = &result.command_output {
        println!("\n== Command output ==");
        println!("exit_code: {}", output.exit_code);
        println!("stdout:\n{}", output.stdout);
        println!("stderr:\n{}", output.stderr);
    }
}

fn display_path(path: Option<&std::path::Path>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| "(caller working directory)".into())
}

fn status_label(status: PreflightStatus) -> &'static str {
    match status {
        PreflightStatus::Passed => "passed",
        PreflightStatus::Warning => "warning",
        PreflightStatus::Failed => "failed",
    }
}
