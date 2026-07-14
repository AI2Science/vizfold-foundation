use chrono::Utc;
use executor::core::{
    entities::{execution_targets, model_backends, model_invocation_profiles, runs},
    model_runners::openfold::plan_openfold_command,
};
use serde_json::json;

fn main() -> Result<(), sea_orm::DbErr> {
    let now = Utc::now();

    let model_backend = model_backends::Model {
        id: 1,
        slug: "openfold".into(),
        label: "OpenFold".into(),
        version: Some("example".into()),
        description: None,
        artifact_capabilities_json: "{}".into(),
        parameter_schema_json: openfold_parameter_schema().to_string(),
        created_at: now,
        updated_at: now,
    };

    let execution_target = execution_targets::Model {
        id: 2,
        slug: "local-runtime".into(),
        target_type: "local".into(),
        description: None,
        parameter_schema_json: execution_parameter_schema().to_string(),
        created_at: now,
        updated_at: now,
    };

    let invocation_profile = model_invocation_profiles::Model {
        id: 3,
        model_backend_id: model_backend.id,
        execution_target_id: execution_target.id,
        invocation_kind: "local_subprocess".into(),
        config_json: json!({
            "program": "python3",
            "script": "run_pretrained_openfold.py",
            "working_dir": "/path/to/vizfold-foundation",
            "env": {
                "PYTHONPATH": "/path/to/vizfold-foundation"
            }
        })
        .to_string(),
        parameter_schema_json: "{}".into(),
        created_at: now,
        updated_at: now,
    };

    let run = runs::Model {
        id: 4,
        model_backend_id: model_backend.id,
        execution_target_id: execution_target.id,
        invocation_profile_id: invocation_profile.id,
        status: "submitted".into(),
        input_sequence: "MSTNPKPQRITF".into(),
        model_parameters_json: json!({
            "config_preset": "model_1_ptm",
            "save_outputs": true,
            "demo_attn": true,
            "num_recycles_save": 1
        })
        .to_string(),
        execution_parameters_json: json!({
            "fasta_dir": "/tmp/vizfold/fasta",
            "output_dir": "/tmp/vizfold/output",
            "data_dir": "/data/openfold",
            "attn_map_dir": "/tmp/vizfold/attention",
            "model_device": "cuda:0",
            "use_precomputed_alignments": true,
            "alignment_dir": "/tmp/vizfold/alignments",
            "cpus": 14,
            "residue_idx": 1
        })
        .to_string(),
        submitted_at: now,
        started_at: None,
        completed_at: None,
        error_message: None,
    };

    let command =
        plan_openfold_command(&model_backend, &execution_target, &invocation_profile, &run)?;

    println!("{command:#?}");

    Ok(())
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
