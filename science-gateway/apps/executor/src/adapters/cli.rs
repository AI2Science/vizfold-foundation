use clap::{ArgAction, Args, Parser, Subcommand};
use sea_orm::DbErr;
use serde_json::json;

use crate::core::{
    commands::LocalCommandRunner,
    db,
    preflight::PreflightStatus,
    repositories::{execution_targets, model_backends, model_invocation_profiles},
    seed::seed_defaults,
    services::{openfold_execution::execute_openfold_run, runs},
};

#[derive(Debug, Parser)]
#[command(name = "vizfold", about = "VizFold executor administration CLI")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Seed the default executor records.
    Seed,
    /// List executor records.
    List(ListArgs),
    /// Show one executor record.
    Show(ShowArgs),
    /// Queue a run for a supported model backend.
    QueueRun(QueueRunArgs),
    /// Execute a queued run for a supported model backend.
    ExecuteRun(ExecuteRunArgs),
}

#[derive(Debug, Args)]
struct ListArgs {
    #[command(subcommand)]
    resource: ListResource,
}

#[derive(Debug, Subcommand)]
enum ListResource {
    /// List model backends.
    Models,
    /// List execution targets.
    Targets,
    /// List model invocation profiles.
    Profiles,
    /// List runs.
    Runs {
        /// Restrict results to runs with this status.
        #[arg(long)]
        status: Option<String>,
    },
}

#[derive(Debug, Args)]
struct ShowArgs {
    #[command(subcommand)]
    resource: ShowResource,
}

#[derive(Debug, Subcommand)]
enum ShowResource {
    /// Show a run and its artifacts.
    Run { run_id: i32 },
}

#[derive(Clone, Debug, Args)]
struct QueueRunArgs {
    #[command(subcommand)]
    model: QueueRunModel,
}

#[derive(Clone, Debug, Subcommand)]
enum QueueRunModel {
    /// Queue an OpenFold run.
    Openfold(OpenfoldQueueArgs),
}

#[derive(Clone, Debug, Args)]
struct OpenfoldQueueArgs {
    #[arg(long)]
    input_id: String,
    #[arg(long)]
    input_sequence: String,
    #[arg(long)]
    fasta_dir: String,
    #[arg(long)]
    data_dir: String,
    #[arg(long)]
    alignment_dir: Option<String>,
    #[arg(long, default_value = "cpu")]
    model_device: String,
    #[arg(long, default_value_t = 1)]
    cpus: i64,
    #[arg(long, default_value_t = 1)]
    residue_idx: i64,
    #[arg(long)]
    demo_attn: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    save_outputs: bool,
    #[arg(long, default_value_t = 1)]
    num_recycles_save: i64,
    #[arg(long)]
    use_precomputed_alignments: bool,
}

#[derive(Debug, Args)]
struct ExecuteRunArgs {
    #[command(subcommand)]
    model: ExecuteRunModel,
}

#[derive(Debug, Subcommand)]
enum ExecuteRunModel {
    /// Execute an OpenFold run.
    Openfold { run_id: i32 },
}

pub async fn run() -> Result<(), DbErr> {
    let cli = Cli::parse();
    let database = db::connect_and_migrate().await?;

    match cli.command {
        Command::Seed => {
            seed_defaults(&database).await?;
            println!("Seeded default executor records.");
        }
        Command::List(list) => match list.resource {
            ListResource::Models => list_models(&database).await?,
            ListResource::Targets => list_targets(&database).await?,
            ListResource::Profiles => list_profiles(&database).await?,
            ListResource::Runs { status } => list_runs(&database, status.as_deref()).await?,
        },
        Command::Show(show) => match show.resource {
            ShowResource::Run { run_id } => show_run(&database, run_id).await?,
        },
        Command::QueueRun(queue) => match queue.model {
            QueueRunModel::Openfold(args) => queue_openfold_run(&database, args).await?,
        },
        Command::ExecuteRun(execute) => match execute.model {
            ExecuteRunModel::Openfold { run_id } => execute_openfold(&database, run_id).await?,
        },
    }

    Ok(())
}

async fn execute_openfold(
    database: &sea_orm::DatabaseConnection,
    run_id: i32,
) -> Result<(), DbErr> {
    println!("Executing OpenFold run {run_id}");
    let result = execute_openfold_run(database, run_id, &LocalCommandRunner).await?;

    if let Some(report) = result.preflight_report {
        let outcome = if report.has_failures() {
            "failed"
        } else {
            "passed"
        };
        println!("\nPreflight: {outcome}");
        for check in report.checks {
            let message = check.message.as_deref().unwrap_or("no details");
            println!(
                "[{}] {}: {}",
                preflight_status_label(check.status),
                check.name,
                message
            );
        }
    }

    if let Some(reason) = result.skipped_execution_reason {
        println!("\nExecution skipped:\n{reason}");
    }

    if let Some(output) = result.command_output {
        println!("\nCommand output:");
        println!("exit_code: {}", output.exit_code);
        println!("stdout:\n{}", output.stdout);
        println!("stderr:\n{}", output.stderr);
    }

    if let Some(run) = runs::get_run_with_artifacts(database, run_id)
        .await?
        .map(|result| result.run)
    {
        println!("\nFinal status: {}", run.status);
    }
    Ok(())
}

fn preflight_status_label(status: PreflightStatus) -> &'static str {
    match status {
        PreflightStatus::Passed => "passed",
        PreflightStatus::Warning => "warning",
        PreflightStatus::Failed => "failed",
    }
}

async fn queue_openfold_run(
    database: &sea_orm::DatabaseConnection,
    args: OpenfoldQueueArgs,
) -> Result<(), DbErr> {
    if args.use_precomputed_alignments && args.alignment_dir.is_none() {
        return Err(DbErr::Custom(
            "--alignment-dir is required when --use-precomputed-alignments is set".into(),
        ));
    }

    let backend = model_backends::find_by_slug(database, "openfold")
        .await?
        .ok_or_else(seed_required_error)?;
    let target = execution_targets::find_by_slug(database, "local-openfold")
        .await?
        .ok_or_else(seed_required_error)?;
    let profile = model_invocation_profiles::list(database)
        .await?
        .into_iter()
        .find(|profile| {
            profile.model_backend_id == backend.id
                && profile.execution_target_id == target.id
                && profile.invocation_kind == "local_subprocess"
        })
        .ok_or_else(seed_required_error)?;

    let mut execution_parameters = serde_json::Map::from_iter([
        ("fasta_dir".into(), json!(args.fasta_dir)),
        ("data_dir".into(), json!(args.data_dir)),
        ("residue_idx".into(), json!(args.residue_idx)),
        (
            "use_precomputed_alignments".into(),
            json!(args.use_precomputed_alignments),
        ),
        ("model_device".into(), json!(args.model_device)),
        ("cpus".into(), json!(args.cpus)),
    ]);
    if let Some(alignment_dir) = args.alignment_dir {
        execution_parameters.insert("alignment_dir".into(), json!(alignment_dir));
    }

    let run = runs::submit_run(
        database,
        runs::SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "submitted".into(),
            input_id: args.input_id,
            input_sequence: args.input_sequence,
            model_parameters_json: json!({
                "save_outputs": args.save_outputs,
                "demo_attn": args.demo_attn,
                "num_recycles_save": args.num_recycles_save,
            })
            .to_string(),
            execution_parameters_json: serde_json::Value::Object(execution_parameters).to_string(),
        },
    )
    .await?;

    println!("Queued OpenFold run {}", run.id);
    println!("status: {}", run.status);
    println!("input_id: {}", run.input_id);
    println!("\nNext:");
    println!("  vizfold execute-run openfold {}", run.id);
    Ok(())
}

fn seed_required_error() -> DbErr {
    DbErr::Custom(
        "OpenFold backend, local-openfold target, or matching profile is missing; run `vizfold seed`"
            .into(),
    )
}

async fn list_models(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let models = model_backends::list(database).await?;
    print_table(
        &["ID", "SLUG", "LABEL", "VERSION"],
        models
            .iter()
            .map(|model| {
                vec![
                    model.id.to_string(),
                    model.slug.clone(),
                    model.label.clone(),
                    model.version.clone().unwrap_or_else(|| "-".into()),
                ]
            })
            .collect(),
    );
    Ok(())
}

async fn list_targets(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let targets = execution_targets::list(database).await?;
    print_table(
        &["ID", "SLUG", "TYPE", "DESCRIPTION"],
        targets
            .iter()
            .map(|target| {
                vec![
                    target.id.to_string(),
                    target.slug.clone(),
                    target.target_type.clone(),
                    target.description.clone().unwrap_or_else(|| "-".into()),
                ]
            })
            .collect(),
    );
    Ok(())
}

async fn list_profiles(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let profiles = model_invocation_profiles::list(database).await?;
    print_table(
        &["ID", "MODEL ID", "TARGET ID", "INVOCATION KIND"],
        profiles
            .iter()
            .map(|profile| {
                vec![
                    profile.id.to_string(),
                    profile.model_backend_id.to_string(),
                    profile.execution_target_id.to_string(),
                    profile.invocation_kind.clone(),
                ]
            })
            .collect(),
    );
    Ok(())
}

async fn list_runs(
    database: &sea_orm::DatabaseConnection,
    status: Option<&str>,
) -> Result<(), DbErr> {
    let runs = runs::list_runs(database).await?;
    print_table(
        &[
            "ID",
            "STATUS",
            "MODEL ID",
            "TARGET ID",
            "PROFILE ID",
            "INPUT ID",
            "SUBMITTED AT",
        ],
        runs.iter()
            .filter(|run| status.is_none_or(|value| run.status == value))
            .map(|run| {
                vec![
                    run.id.to_string(),
                    run.status.clone(),
                    run.model_backend_id.to_string(),
                    run.execution_target_id.to_string(),
                    run.invocation_profile_id.to_string(),
                    run.input_id.clone(),
                    run.submitted_at.to_rfc3339(),
                ]
            })
            .collect(),
    );
    Ok(())
}

async fn show_run(database: &sea_orm::DatabaseConnection, run_id: i32) -> Result<(), DbErr> {
    let Some(result) = runs::get_run_with_artifacts(database, run_id).await? else {
        return Err(DbErr::Custom(format!("run {run_id} does not exist")));
    };
    let run = result.run;

    println!("Run {}", run.id);
    println!("status: {}", run.status);
    println!("input_id: {}", run.input_id);
    println!("model_backend_id: {}", run.model_backend_id);
    println!("execution_target_id: {}", run.execution_target_id);
    println!("invocation_profile_id: {}", run.invocation_profile_id);
    println!("submitted_at: {}", run.submitted_at.to_rfc3339());
    println!("started_at: {}", format_time(run.started_at));
    println!("completed_at: {}", format_time(run.completed_at));
    if let Some(error_message) = run.error_message {
        println!("error_message: {error_message}");
    }

    println!("artifacts:");
    print_table(
        &["ID", "TYPE ID", "FORMAT", "STORAGE URI"],
        result
            .artifacts
            .iter()
            .map(|artifact| {
                vec![
                    artifact.id.to_string(),
                    artifact.artifact_type_id.to_string(),
                    artifact.format.clone(),
                    artifact.storage_uri.clone(),
                ]
            })
            .collect(),
    );
    Ok(())
}

fn format_time(value: Option<chrono::DateTime<chrono::Utc>>) -> String {
    value
        .map(|time| time.to_rfc3339())
        .unwrap_or_else(|| "-".into())
}

fn print_table(headers: &[&str], rows: Vec<Vec<String>>) {
    let mut widths: Vec<usize> = headers.iter().map(|header| header.len()).collect();
    for row in &rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(cell.len());
        }
    }

    print_row(headers.iter().copied(), &widths);
    let separator = widths
        .iter()
        .map(|width| "-".repeat(*width))
        .collect::<Vec<_>>();
    print_row(separator.iter().map(String::as_str), &widths);
    for row in rows {
        print_row(row.iter().map(String::as_str), &widths);
    }
}

fn print_row<'a>(cells: impl IntoIterator<Item = &'a str>, widths: &[usize]) {
    let rendered = cells
        .into_iter()
        .zip(widths)
        .map(|(cell, width)| format!("{cell:<width$}", width = width))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{rendered}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use sea_orm::{ConnectionTrait, Database, Statement};

    use crate::core::{db, seed};

    #[test]
    fn parses_list_runs_with_status_filter() {
        let cli = Cli::try_parse_from(["vizfold", "list", "runs", "--status", "failed"])
            .expect("list runs command should parse");

        assert!(matches!(
            cli.command,
            Command::List(ListArgs {
                resource: ListResource::Runs { status: Some(status) }
            }) if status == "failed"
        ));
    }

    #[test]
    fn parses_show_run() {
        let cli = Cli::try_parse_from(["vizfold", "show", "run", "1"])
            .expect("show run command should parse");

        assert!(matches!(
            cli.command,
            Command::Show(ShowArgs {
                resource: ShowResource::Run { run_id: 1 }
            })
        ));
    }

    #[test]
    fn parses_queue_openfold_required_arguments() {
        let cli = Cli::try_parse_from([
            "vizfold",
            "queue-run",
            "openfold",
            "--input-id",
            "6KWC_1",
            "--input-sequence",
            "GSTI",
            "--fasta-dir",
            "fasta",
            "--data-dir",
            "data",
        ])
        .expect("queue-run command should parse");

        assert!(matches!(
            cli.command,
            Command::QueueRun(QueueRunArgs {
                model: QueueRunModel::Openfold(OpenfoldQueueArgs {
                    input_id,
                    input_sequence,
                    fasta_dir,
                    data_dir,
                    demo_attn: false,
                    use_precomputed_alignments: false,
                    cpus: 1,
                    ..
                })
            }) if input_id == "6KWC_1" && input_sequence == "GSTI" && fasta_dir == "fasta" && data_dir == "data"
        ));
    }

    #[test]
    fn parses_queue_openfold_optional_flags() {
        let cli = Cli::try_parse_from([
            "vizfold",
            "queue-run",
            "openfold",
            "--input-id",
            "6KWC_1",
            "--input-sequence",
            "GSTI",
            "--fasta-dir",
            "fasta",
            "--data-dir",
            "data",
            "--cpus",
            "4",
            "--demo-attn",
            "--use-precomputed-alignments",
        ])
        .expect("queue-run command should parse");

        assert!(matches!(
            cli.command,
            Command::QueueRun(QueueRunArgs {
                model: QueueRunModel::Openfold(OpenfoldQueueArgs {
                    cpus: 4,
                    demo_attn: true,
                    use_precomputed_alignments: true,
                    ..
                })
            })
        ));
    }

    #[test]
    fn parses_execute_openfold_run() {
        let cli = Cli::try_parse_from(["vizfold", "execute-run", "openfold", "1"])
            .expect("execute-run command should parse");

        assert!(matches!(
            cli.command,
            Command::ExecuteRun(ExecuteRunArgs {
                model: ExecuteRunModel::Openfold { run_id: 1 }
            })
        ));
    }

    #[tokio::test]
    async fn queue_openfold_run_uses_seeded_records() -> Result<(), DbErr> {
        let database = Database::connect("sqlite::memory:").await?;
        database
            .execute(Statement::from_string(
                database.get_database_backend(),
                "PRAGMA foreign_keys = ON".to_owned(),
            ))
            .await?;
        db::migrate_database(&database).await?;
        seed::seed_defaults(&database).await?;

        queue_openfold_run(
            &database,
            OpenfoldQueueArgs {
                input_id: "6KWC_1".into(),
                input_sequence: "GSTI".into(),
                fasta_dir: "fasta".into(),
                data_dir: "data".into(),
                alignment_dir: Some("alignments".into()),
                model_device: "cpu".into(),
                cpus: 1,
                residue_idx: 1,
                demo_attn: true,
                save_outputs: true,
                num_recycles_save: 1,
                use_precomputed_alignments: true,
            },
        )
        .await?;

        let runs = runs::list_runs(&database).await?;
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].status, "submitted");
        assert_eq!(runs[0].input_id, "6KWC_1");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&runs[0].model_parameters_json)
                .expect("model parameters should be valid JSON"),
            json!({"save_outputs": true, "demo_attn": true, "num_recycles_save": 1})
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&runs[0].execution_parameters_json)
                .expect("execution parameters should be valid JSON"),
            json!({"fasta_dir": "fasta", "data_dir": "data", "alignment_dir": "alignments", "residue_idx": 1, "use_precomputed_alignments": true, "model_device": "cpu", "cpus": 1})
        );
        Ok(())
    }
}
