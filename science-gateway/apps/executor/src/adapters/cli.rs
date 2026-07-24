use clap::{ArgAction, Args, Parser, Subcommand};
use sea_orm::{ColumnTrait, DbErr, EntityTrait, QueryFilter};
use serde_json::json;
use std::path::{Path, PathBuf};

use crate::core::{
    commands::LocalCommandRunner,
    config, db,
    entities::{
        execution_targets as execution_target_entity, model_backends as model_backend_entity,
        model_invocation_profiles as model_invocation_profile_entity,
    },
    output_locations::resolve_output_location,
    preflight::PreflightStatus,
    seed::seed_defaults,
    services::{
        artifacts, execution_targets, model_backends, model_invocation_profiles,
        openfold_artifacts::register_known_openfold_artifacts,
        openfold_execution::execute_openfold_run, runs,
    },
};

#[derive(Debug, Parser)]
#[command(name = "vizfold", about = "VizFold executor administration CLI")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Install a model backend (OpenFold) on this machine.
    Install,
    /// Remove everything the install generated.
    Uninstall(UninstallArgs),
    /// Start the workbench dashboard.
    Serve(ServeArgs),
    /// Seed the default executor records.
    Seed,
    /// List executor records.
    List(ListArgs),
    /// Show one executor record.
    Show(ShowArgs),
    /// Queue a run for a supported model backend.
    QueueRun(QueueRunArgs),
    /// Execute a queued run.
    ExecuteRun { run_id: i32 },
    /// Register known artifacts for a completed run.
    RegisterArtifacts { run_id: i32 },
}

#[derive(Debug, Args)]
struct UninstallArgs {
    /// Remove without the confirmation prompt.
    #[arg(long, short = 'y')]
    yes: bool,
}

#[derive(Debug, Args)]
struct ServeArgs {
    /// Port for the dashboard dev server (defaults to 3000).
    #[arg(long)]
    port: Option<u16>,
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
    /// FASTA directory. Defaults to <OPENFOLD_HOME>/examples/monomer/fasta_dir_<id> (fold.sh convention).
    #[arg(long)]
    fasta_dir: Option<String>,
    /// OpenFold data directory. Defaults to the config `OPENFOLD_DATA_DIR`.
    #[arg(long)]
    data_dir: Option<String>,
    /// Precomputed alignments directory. Defaults to <OPENFOLD_HOME>/examples/monomer/alignments.
    #[arg(long)]
    alignment_dir: Option<String>,
    /// Torch device. Defaults to cuda:0 when a GPU partition is configured to srun onto (the
    /// HPC flow) or a GPU is visible locally, otherwise cpu.
    #[arg(long)]
    model_device: Option<String>,
    #[arg(long, default_value_t = default_cpus())]
    cpus: i64,
    #[arg(long, default_value_t = 1)]
    residue_idx: i64,
    #[arg(long)]
    demo_attn: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    save_outputs: bool,
    #[arg(long, default_value_t = 1)]
    num_recycles_save: i64,
    /// Use the precomputed alignments in <OPENFOLD_HOME>/examples/monomer/alignments
    /// (fold.sh default). Pass `--use-precomputed-alignments=false` for the full MSA pipeline.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    use_precomputed_alignments: bool,
}

/// No allocation held but a GPU partition configured means the fold will be srun'd onto a
/// GPU node -- the current host (e.g. a login node) is irrelevant in that case.
fn on_gpu_partition(context: config::SlurmContext, partition: Option<&str>) -> bool {
    matches!(context, config::SlurmContext::None) && partition.is_some_and(|p| !p.is_empty())
}

fn model_device_for(
    context: config::SlurmContext,
    partition: Option<&str>,
    detected: Option<&str>,
) -> String {
    if on_gpu_partition(context, partition) || detected.is_some() {
        "cuda:0".to_owned()
    } else {
        "cpu".to_owned()
    }
}

/// Skips the local GPU probe entirely when it wouldn't be consulted anyway (the login node has
/// no GPU to find), rather than just discarding its result.
fn default_model_device() -> String {
    let context = config::SlurmContext::detect();
    let partition = config::gpu_partition();
    let detected = if on_gpu_partition(context, partition.as_deref()) {
        None
    } else {
        crate::core::model_runners::openfold::detect_gpu()
    };
    model_device_for(context, partition.as_deref(), detected.as_deref())
}

fn default_cpus() -> i64 {
    std::thread::available_parallelism().map_or(1, |n| n.get() as i64)
}

/// Clamp the requested CPU count to the execution target's `cpus.maximum`, so a host with more
/// cores than the target allows (a beefy workstation, any HPC login node) still queues a run
/// that `execute-run` can plan -- rather than failing only once execution is attempted.
fn clamp_cpus(cpus: i64, available_resources_json: &str) -> i64 {
    let max_cpus = serde_json::from_str::<serde_json::Value>(available_resources_json)
        .ok()
        .and_then(|resources| resources["properties"]["cpus"]["maximum"].as_i64())
        .unwrap_or(i64::MAX);
    cpus.min(max_cpus)
}

pub async fn run() -> Result<(), DbErr> {
    let cli = Cli::parse();

    // `install` is the bootstrap and `uninstall` cleans up after a partial one; everything
    // else requires an initialized config.
    if !matches!(cli.command, Command::Install | Command::Uninstall(_)) && !config::is_initialized()
    {
        eprintln!("run `vizfold install` first");
        std::process::exit(1);
    }

    // These three touch the filesystem only; they need no database connection.
    match cli.command {
        Command::Install => return run_install(),
        Command::Uninstall(args) => return run_uninstall(args),
        Command::Serve(args) => return run_serve(args),
        _ => {}
    }

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
        Command::ExecuteRun { run_id } => execute_openfold(&database, run_id).await?,
        Command::RegisterArtifacts { run_id } => register_artifacts(&database, run_id).await?,
        Command::Install | Command::Uninstall(_) | Command::Serve(_) => {
            unreachable!("handled before DB connect")
        }
    }

    Ok(())
}

/// Install a model backend (OpenFold) by running the checkout's `install/init.sh` with
/// inherited stdio. The release binary ships only itself, so the checkout is cloned on
/// first install. Idempotent: its steps are sentinel-guarded, so re-running is safe.
fn run_install() -> Result<(), DbErr> {
    let src = config::vizfold_src();
    let installer = src.join("install/init.sh");
    if !installer.is_file() {
        clone_checkout(&src)?;
    }
    if !installer.is_file() {
        return Err(DbErr::Custom(format!(
            "no vizfold checkout at {}; set VIZFOLD_SRC to a checkout",
            src.display()
        )));
    }
    println!("Running model install: bash {}", installer.display());
    let status = std::process::Command::new("bash")
        .arg(&installer)
        .env("OPENFOLD_HOME", &src)
        .status()
        .map_err(|error| DbErr::Custom(format!("failed to launch model install: {error}")))?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| DbErr::Custom(format!("model install exited with status {status}")))
}

/// Clone the vizfold checkout `vizfold install` runs its scripts (and serves the dashboard)
/// from -- the release binary ships only itself. Pins the binary's own version tag
/// (`VIZFOLD_REF` overrides), falling back to the repo default branch.
fn clone_checkout(src: &std::path::Path) -> Result<(), DbErr> {
    let repo =
        std::env::var("VIZFOLD_REPO").unwrap_or_else(|_| "AI2Science/vizfold-foundation".into());
    let url = format!("https://github.com/{repo}.git");
    let dest = src.to_string_lossy().into_owned();
    let pinned = match std::env::var("VIZFOLD_REF") {
        Ok(r) if !r.is_empty() => Some(r),
        _ => Some(format!("v{}", env!("CARGO_PKG_VERSION"))),
    };
    println!("Fetching the vizfold checkout into {dest} ...");
    let clone = |args: &[&str]| std::process::Command::new("git").args(args).status();
    if let Some(r) = &pinned
        && let Ok(s) = clone(&["clone", "--depth", "1", "--branch", r, &url, &dest])
        && s.success()
    {
        return Ok(());
    }
    match clone(&["clone", "--depth", "1", &url, &dest]) {
        Ok(s) if s.success() => Ok(()),
        _ => Err(DbErr::Custom(format!(
            "failed to clone {url} into {dest}; set VIZFOLD_SRC to an existing checkout"
        ))),
    }
}

/// Undo `vizfold install`: the install prefix's generated trees, the caches beside it, what it
/// planted in the checkout, the run database and the install config. Resolved here rather than
/// in an `install/uninstall.sh` because the checkout holding that script is itself one of the
/// things being removed. Kept: fold outputs, a checkout the user pointed at, and this binary.
fn run_uninstall(args: UninstallArgs) -> Result<(), DbErr> {
    let prefix = config::prefix();
    let mut targets = install_paths(&prefix, &config::openfold_home());
    // Only the clone the install made itself: a checkout the user pointed at is theirs, and one
    // holding the prefix holds the fold outputs too.
    let src = config::vizfold_src();
    if src == config::default_src() && !prefix.starts_with(&src) {
        targets.push(src);
    }
    if let Some(database) = config::database_path() {
        let sidecar = |suffix| PathBuf::from(format!("{}{suffix}", database.display()));
        targets.extend([sidecar("-wal"), sidecar("-shm"), database]);
    }
    targets.push(config::config_file());

    // Relative paths mean an empty config value resolved into one; never delete off the cwd.
    targets.retain(|path| path.is_absolute() && std::fs::symlink_metadata(path).is_ok());
    targets.sort();
    targets.dedup();
    // Drop what an outer target already covers (the clone contains the checkout paths), so the
    // plan is what it removes. ponytail: O(n^2) over ~25 paths.
    let outer = targets.clone();
    targets.retain(|path| {
        !outer
            .iter()
            .any(|other| other != path && path.starts_with(other))
    });
    if targets.is_empty() {
        println!("Nothing to remove.");
        return Ok(());
    }

    println!("This removes:");
    for target in &targets {
        println!("  {}", target.display());
    }
    if !args.yes && !confirmed()? {
        println!("Aborted.");
        return Ok(());
    }
    for target in &targets {
        match remove_path(target) {
            Ok(()) => println!("removed {}", target.display()),
            Err(error) => eprintln!("warning: could not remove {}: {error}", target.display()),
        }
    }
    if let Some(dir) = config::config_file().parent() {
        let _ = std::fs::remove_dir(dir); // ours, but only if the uninstall emptied it
    }
    println!("\nKept: fold outputs, the vizfold checkout, and the vizfold binary itself.");
    Ok(())
}

/// What an install writes under the prefix and into the checkout (`install/setup.sh`), minus
/// `<prefix>/outputs` -- those are run results, not install state.
fn install_paths(prefix: &Path, home: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = [
        "bin/micromamba",
        "mamba",
        "cutlass",
        "tmp",
        "data",
        ".done",
        "params",
        "workbench",
        "vizfold.db",
    ]
    .iter()
    .map(|entry| prefix.join(entry))
    .collect();
    // One nvrtc-<driver-cuda> side prefix per driver version the install has pinned for.
    paths.extend(
        std::fs::read_dir(prefix)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| file_name(path).starts_with("nvrtc-")),
    );
    // Package caches, deliberately parked beside the prefix rather than in it.
    paths.extend(
        prefix
            .parent()
            .into_iter()
            .flat_map(|dir| [dir.join(".openfold-pkgs"), dir.join(".openfold-pip")]),
    );
    paths.extend(
        [
            "openfold/resources/params",
            "openfold/resources/stereo_chemical_props.txt",
            "tests/test_data/alphafold/common/stereo_chemical_props.txt",
            "openfold.egg-info", // both left by the editable install of the checkout
            "build",
        ]
        .map(|entry| home.join(entry)),
    );
    paths
}

fn file_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
}

/// Delete a file, directory tree, or symlink. `symlink_metadata` keeps a symlink to a directory
/// (the AF2 mirror links under `<prefix>/data`) from being followed into the directory it points at.
fn remove_path(path: &Path) -> std::io::Result<()> {
    if std::fs::symlink_metadata(path)?.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

fn confirmed() -> Result<bool, DbErr> {
    use std::io::Write;
    print!("Remove these? [y/N] ");
    std::io::stdout().flush().ok();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|error| DbErr::Custom(format!("could not read confirmation: {error}")))?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes" | "YES"))
}

/// Start the workbench dashboard, streaming its output to this shell.
fn run_serve(args: ServeArgs) -> Result<(), DbErr> {
    let workbench = serve_dir()?;

    // Serve run outputs to the browser by linking the seeded output_location under the
    // dashboard's public/, so Next serves <prefix>/runs/<id>/... at /runs/<id>/... with no
    // file-serving code of our own.
    // ponytail: targets the seeded output_location (prefix/runs). A profile with a different
    // output_location isn't reachable this way -- read it from the run's provenance if that ever
    // happens.
    let runs_dir = config::prefix().join("runs");
    std::fs::create_dir_all(&runs_dir).ok();
    // public/ may not exist (a workbench with no static assets); the symlink's parent must.
    std::fs::create_dir_all(workbench.join("public")).ok();
    match std::os::unix::fs::symlink(&runs_dir, workbench.join("public/runs")) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(DbErr::Custom(format!(
                "failed to link run outputs into the dashboard: {error}"
            )));
        }
    }

    let node_modules = workbench.join("node_modules");
    let empty =
        std::fs::read_dir(&node_modules).map_or(true, |mut entries| entries.next().is_none());
    if empty {
        println!("Installing workbench dependencies (npm install)...");
        run_npm(&workbench, &["install"])?;
    }

    let port = args.port.unwrap_or(3000);
    println!("Starting workbench at http://localhost:{port}");
    let port_arg = port.to_string();
    let mut npm_args = vec!["run", "dev"];
    if args.port.is_some() {
        npm_args.extend(["--", "--port", &port_arg]);
    }
    run_npm(&workbench, &npm_args)
}

/// Directory the dashboard runs from. A cluster home is inode-quota-capped NFS, so on a real
/// install (prefix on a separate work fs) run the dashboard from a copy on that work fs — then
/// npm's node_modules/.next land there, never on home. Dev checkout (prefix == home): run in
/// place. The copy skips node_modules/.next (build artifacts) and preserves any already staged
/// in the destination.
fn serve_dir() -> Result<PathBuf, DbErr> {
    let src = config::openfold_home().join("science-gateway/apps/workbench");
    if config::prefix() == config::openfold_home() {
        return Ok(src);
    }
    let dest = config::prefix().join("workbench");
    copy_tree(&src, &dest, &["node_modules", ".next"]).map_err(|error| {
        DbErr::Custom(format!(
            "failed to stage workbench at '{}': {error}",
            dest.display()
        ))
    })?;
    Ok(dest)
}

/// Recursively copy `src` into `dst`, overwriting files and merging directories, but skip the
/// given names at the top level (build artifacts we neither copy nor clobber in `dst`).
fn copy_tree(src: &Path, dst: &Path, skip: &[&str]) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_str().is_some_and(|n| skip.contains(&n)) {
            continue;
        }
        let file_type = entry.file_type()?;
        // Never copy a symlink (e.g. public/runs -> the run outputs): fs::copy would follow it,
        // and following a link-to-directory hits EISDIR. The link is recreated at serve time.
        if file_type.is_symlink() {
            continue;
        }
        let (from, to) = (entry.path(), dst.join(&name));
        if file_type.is_dir() {
            copy_tree(&from, &to, &[])?; // skip applies only at the workbench root
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn run_npm(dir: &Path, args: &[&str]) -> Result<(), DbErr> {
    let mut command = std::process::Command::new("npm");
    command.current_dir(dir).args(args);
    // The dashboard reads the sqlite file directly; hand it the plain path (database_url() carries
    // a sqlite://...?mode=rwc wrapper that node:sqlite can't open).
    if let Some(database) = config::database_path() {
        command.env("VIZFOLD_DB", database);
    }
    let status = command.status().map_err(|error| {
        DbErr::Custom(format!(
            "failed to run npm (is Node.js installed and on PATH?): {error}"
        ))
    })?;
    status.success().then_some(()).ok_or_else(|| {
        DbErr::Custom(format!(
            "npm {} exited with status {status}",
            args.join(" ")
        ))
    })
}

async fn register_artifacts(
    database: &sea_orm::DatabaseConnection,
    run_id: i32,
) -> Result<(), DbErr> {
    let run = runs::get_run_with_artifacts(database, run_id)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("run {run_id} does not exist")))?
        .run;
    if run.status != "completed" {
        println!(
            "Warning: run {run_id} has status '{}'; registered artifacts may be partial.",
            run.status
        );
    }
    let backend = model_backend_entity::Entity::find_by_id(run.model_backend_id)
        .one(database)
        .await?
        .ok_or_else(|| DbErr::Custom("run model backend does not exist".into()))?;
    if backend.slug != "openfold" {
        return Err(DbErr::Custom(format!(
            "artifact registration is currently only implemented for OpenFold runs (run {run_id} uses backend '{}')",
            backend.slug
        )));
    }

    let profile = model_invocation_profile_entity::Entity::find_by_id(run.invocation_profile_id)
        .one(database)
        .await?
        .ok_or_else(|| DbErr::Custom("model invocation profile does not exist".into()))?;
    let workspace = resolve_output_location(&profile, &run)?;
    let expected_paths = [
        ("run_output_directory", workspace.clone()),
        ("attention_output_directory", workspace.join("attention")),
    ];
    let existing = artifacts::list_artifacts_for_run(database, run_id).await?;
    let artifacts = register_known_openfold_artifacts(database, run_id).await?;

    println!("Registered artifacts for run {run_id}");
    println!("\nOutput workspace:\n  {}", workspace.display());
    println!("\nArtifacts:");
    for (artifact_type, path) in expected_paths {
        let storage_uri = path.display().to_string();
        if !path.is_dir() {
            println!("  [skipped] {artifact_type} -> path does not exist: {storage_uri}");
        } else if existing
            .iter()
            .any(|artifact| artifact.storage_uri == storage_uri)
        {
            println!("  [already present] {artifact_type} -> {storage_uri}");
        } else if artifacts
            .iter()
            .any(|artifact| artifact.storage_uri == storage_uri)
        {
            println!("  [registered] {artifact_type} -> {storage_uri}");
        } else {
            println!("  [skipped] {artifact_type} -> not registered");
        }
    }
    Ok(())
}

async fn execute_openfold(
    database: &sea_orm::DatabaseConnection,
    run_id: i32,
) -> Result<(), DbErr> {
    println!("Executing OpenFold run {run_id}");
    let outcome = execute_openfold_run(database, run_id, &LocalCommandRunner).await?;

    let label = if outcome.report.has_failures() {
        "failed"
    } else {
        "passed"
    };
    println!("\nPreflight: {label}");
    for check in outcome.report.checks {
        let message = check.message.as_deref().unwrap_or("no details");
        println!(
            "[{}] {}: {}",
            preflight_status_label(check.status),
            check.name,
            message
        );
    }

    if let Some(output) = outcome.output {
        println!("\nCommand output:");
        println!("exit_code: {}", output.exit_code);
        if !output.stdout.is_empty() {
            println!("stdout:\n{}", output.stdout);
        }
        if !output.stderr.is_empty() {
            println!("stderr:\n{}", output.stderr);
        }
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
    let backend = model_backend_entity::Entity::find()
        .filter(model_backend_entity::Column::Slug.eq("openfold"))
        .one(database)
        .await?
        .ok_or_else(seed_required_error)?;
    let target = execution_target_entity::Entity::find()
        .filter(execution_target_entity::Column::Slug.eq("local-openfold"))
        .one(database)
        .await?
        .ok_or_else(seed_required_error)?;
    let profile = model_invocation_profiles::list_model_invocation_profiles(database)
        .await?
        .into_iter()
        .find(|profile| {
            profile.model_backend_id == backend.id
                && profile.execution_target_id == target.id
                && profile.invocation_kind == "local_subprocess"
        })
        .ok_or_else(seed_required_error)?;
    let provenance = runs::provenance_snapshot(
        &backend.slug,
        backend.version.as_deref(),
        &target.slug,
        &profile.invocation_kind,
        &profile.config_json,
        &config::openfold_home(),
        &config::prefix(),
        &config::openfold_env_prefix(),
    );
    let working_dir = local_openfold_working_dir(&profile)?;
    let fasta_dir_input = args
        .fasta_dir
        .clone()
        .unwrap_or_else(|| default_fasta_dir(&args.input_id));
    let data_dir_input = args
        .data_dir
        .clone()
        .unwrap_or_else(|| config::data_dir().to_string_lossy().into_owned());
    let fasta_dir = canonicalize_local_path("--fasta-dir", &fasta_dir_input, &working_dir)?;
    let data_dir = canonicalize_local_path("--data-dir", &data_dir_input, &working_dir)?;
    let alignment_dir = if args.use_precomputed_alignments {
        let input = args
            .alignment_dir
            .clone()
            .unwrap_or_else(default_alignment_dir);
        Some(canonicalize_local_path(
            "--alignment-dir",
            &input,
            &working_dir,
        )?)
    } else {
        args.alignment_dir
            .as_deref()
            .map(|path| canonicalize_local_path("--alignment-dir", path, &working_dir))
            .transpose()?
    };
    let model_device = args
        .model_device
        .clone()
        .unwrap_or_else(default_model_device);

    let mut execution_parameters = serde_json::Map::from_iter([
        ("fasta_dir".into(), json!(fasta_dir)),
        ("data_dir".into(), json!(data_dir)),
        ("residue_idx".into(), json!(args.residue_idx)),
        (
            "use_precomputed_alignments".into(),
            json!(args.use_precomputed_alignments),
        ),
        ("model_device".into(), json!(model_device)),
        (
            "cpus".into(),
            json!(clamp_cpus(args.cpus, &target.available_resources_json)),
        ),
    ]);
    if let Some(alignment_dir) = alignment_dir {
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
            provenance_json: Some(provenance),
        },
    )
    .await?;

    println!("Queued OpenFold run {}", run.id);
    println!("status: {}", run.status);
    println!("input_id: {}", run.input_id);
    println!("\nNext:");
    println!("  vizfold execute-run {}", run.id);
    Ok(())
}

/// `<OPENFOLD_HOME>/examples/monomer/fasta_dir_<id-stem>`, matching fold.sh's `${INPUT_ID%_*}`.
fn default_fasta_dir(input_id: &str) -> String {
    let stem = input_id.rsplit_once('_').map_or(input_id, |(head, _)| head);
    config::openfold_home()
        .join("examples/monomer")
        .join(format!("fasta_dir_{stem}"))
        .to_string_lossy()
        .into_owned()
}

fn default_alignment_dir() -> String {
    config::openfold_home()
        .join("examples/monomer/alignments")
        .to_string_lossy()
        .into_owned()
}

fn local_openfold_working_dir(
    profile: &crate::core::entities::model_invocation_profiles::Model,
) -> Result<String, DbErr> {
    let config: serde_json::Value =
        serde_json::from_str(&profile.config_json).map_err(|error| {
            DbErr::Custom(format!(
                "local OpenFold invocation profile config_json must be valid JSON: {error}"
            ))
        })?;
    config
        .get("working_dir")
        .and_then(serde_json::Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            DbErr::Custom(
                "local OpenFold invocation profile config_json requires a non-empty working_dir"
                    .into(),
            )
        })
}

fn canonicalize_local_path(field: &str, path: &str, working_dir: &str) -> Result<String, DbErr> {
    let original_path = Path::new(path);
    let attempted_path = if original_path.is_absolute() {
        PathBuf::from(original_path)
    } else {
        PathBuf::from(working_dir).join(original_path)
    };

    std::fs::canonicalize(&attempted_path)
        .map(|path| path.display().to_string())
        .map_err(|error| {
            DbErr::Custom(format!(
                "{field} original path '{path}' could not be resolved at '{}': {error}",
                attempted_path.display()
            ))
        })
}

fn seed_required_error() -> DbErr {
    DbErr::Custom(
        "OpenFold backend, local-openfold target, or matching profile is missing; run `vizfold seed`"
            .into(),
    )
}

async fn list_models(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let models = model_backends::list_model_backends(database).await?;
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
    let targets = execution_targets::list_execution_targets(database).await?;
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
    let profiles = model_invocation_profiles::list_model_invocation_profiles(database).await?;
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
                    use_precomputed_alignments: true,
                    cpus,
                    ..
                })
            }) if input_id == "6KWC_1" && input_sequence == "GSTI"
                && fasta_dir.as_deref() == Some("fasta") && data_dir.as_deref() == Some("data")
                && cpus == default_cpus()
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
            "--use-precomputed-alignments=false",
        ])
        .expect("queue-run command should parse");

        assert!(matches!(
            cli.command,
            Command::QueueRun(QueueRunArgs {
                model: QueueRunModel::Openfold(OpenfoldQueueArgs {
                    cpus: 4,
                    demo_attn: true,
                    use_precomputed_alignments: false,
                    ..
                })
            })
        ));
    }

    #[test]
    fn parses_install() {
        let cli =
            Cli::try_parse_from(["vizfold", "install"]).expect("install command should parse");
        assert!(matches!(cli.command, Command::Install));
    }

    #[test]
    fn parses_uninstall() {
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "uninstall"])
                .expect("uninstall command should parse")
                .command,
            Command::Uninstall(UninstallArgs { yes: false })
        ));
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "uninstall", "--yes"])
                .expect("uninstall --yes should parse")
                .command,
            Command::Uninstall(UninstallArgs { yes: true })
        ));
    }

    #[test]
    fn install_paths_cover_generated_trees_but_not_run_outputs() {
        let base = std::env::temp_dir().join(format!("vizfold-uninstall-{}", std::process::id()));
        let (prefix, home) = (base.join("prefix"), base.join("checkout"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(prefix.join("nvrtc-12.2")).unwrap();
        std::fs::create_dir_all(prefix.join("outputs")).unwrap();

        let paths = super::install_paths(&prefix, &home);

        for expected in [
            prefix.join("mamba"),
            prefix.join("data"),
            prefix.join(".done"),
            prefix.join("nvrtc-12.2"),
            base.join(".openfold-pkgs"),
            home.join("openfold/resources/params"),
        ] {
            assert!(paths.contains(&expected), "missing {}", expected.display());
        }
        assert!(!paths.contains(&prefix.join("outputs")), "run outputs kept");
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn remove_path_unlinks_a_symlink_without_touching_its_target() {
        let base = std::env::temp_dir().join(format!("vizfold-rm-{}", std::process::id()));
        let (mirror, link) = (base.join("mirror"), base.join("params"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&mirror).unwrap();
        std::fs::write(mirror.join("params.npz"), "keep").unwrap();
        std::os::unix::fs::symlink(&mirror, &link).unwrap();

        super::remove_path(&link).unwrap();

        assert!(link.symlink_metadata().is_err(), "symlink should be gone");
        assert!(mirror.join("params.npz").is_file(), "target must survive");
        super::remove_path(&mirror).unwrap();
        assert!(!mirror.exists(), "a real directory is removed whole");
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn copy_tree_excludes_build_artifacts_and_preserves_dest() {
        let base = std::env::temp_dir().join(format!("vizfold-copytree-{}", std::process::id()));
        let (src, dst) = (base.join("src"), base.join("dst"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(src.join("node_modules")).unwrap();
        std::fs::create_dir_all(src.join(".next")).unwrap();
        std::fs::create_dir_all(src.join("app")).unwrap();
        std::fs::write(src.join("package.json"), "{}").unwrap();
        std::fs::write(src.join("node_modules/dep.js"), "src").unwrap();
        std::fs::write(src.join("app/page.tsx"), "x").unwrap();
        // A node_modules already staged in the destination must survive the copy.
        std::fs::create_dir_all(dst.join("node_modules")).unwrap();
        std::fs::write(dst.join("node_modules/installed.js"), "keep").unwrap();

        super::copy_tree(&src, &dst, &["node_modules", ".next"]).unwrap();

        assert!(dst.join("package.json").is_file());
        assert!(dst.join("app/page.tsx").is_file());
        assert!(!dst.join(".next").exists()); // excluded at top level
        assert!(dst.join("node_modules/installed.js").is_file()); // preserved
        assert!(!dst.join("node_modules/dep.js").exists()); // src node_modules not copied
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn copy_tree_skips_a_symlinked_directory() {
        // public/runs is a symlink to the run outputs; fs::copy would follow it into a directory
        // and fail with EISDIR. The stage must skip it, not choke on it.
        let base = std::env::temp_dir().join(format!("vizfold-copytree-link-{}", std::process::id()));
        let (src, dst, outputs) = (base.join("src"), base.join("dst"), base.join("outputs"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(src.join("public")).unwrap();
        std::fs::create_dir_all(&outputs).unwrap();
        std::fs::write(src.join("package.json"), "{}").unwrap();
        std::os::unix::fs::symlink(&outputs, src.join("public/runs")).unwrap();

        super::copy_tree(&src, &dst, &[]).unwrap();

        assert!(dst.join("package.json").is_file());
        assert!(!dst.join("public/runs").exists()); // the symlink is not staged
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn parses_serve_with_port() {
        let cli = Cli::try_parse_from(["vizfold", "serve", "--port", "3001"])
            .expect("serve command should parse");
        assert!(matches!(
            cli.command,
            Command::Serve(ServeArgs { port: Some(3001) })
        ));
    }

    #[test]
    fn parses_execute_run() {
        let cli = Cli::try_parse_from(["vizfold", "execute-run", "1"])
            .expect("execute-run command should parse");

        assert!(matches!(cli.command, Command::ExecuteRun { run_id: 1 }));
    }

    #[test]
    fn parses_register_artifacts() {
        let cli = Cli::try_parse_from(["vizfold", "register-artifacts", "1"])
            .expect("register-artifacts command should parse");

        assert!(matches!(
            cli.command,
            Command::RegisterArtifacts { run_id: 1 }
        ));
    }

    #[tokio::test]
    async fn queue_openfold_run_uses_seeded_records() -> Result<(), DbErr> {
        let local_path = std::fs::canonicalize(crate::core::config::openfold_home())
            .expect("OpenFold home should be canonicalizable")
            .display()
            .to_string();
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
                fasta_dir: Some(".".into()),
                data_dir: Some(".".into()),
                alignment_dir: Some(".".into()),
                model_device: Some("cpu".into()),
                // Exceeds the seeded local-openfold target's cpus.maximum of 14, so the queued
                // run must reflect the clamped value, not the raw request.
                cpus: 18,
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
            json!({"fasta_dir": local_path, "data_dir": local_path, "alignment_dir": local_path, "residue_idx": 1, "use_precomputed_alignments": true, "model_device": "cpu", "cpus": 14})
        );

        let provenance: serde_json::Value = serde_json::from_str(
            runs[0]
                .provenance_json
                .as_deref()
                .expect("provenance_json should be set"),
        )
        .expect("provenance_json should be valid JSON");
        assert_eq!(
            provenance["profile"]["config"]["output_location"],
            json!(config::prefix().join("runs"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn queue_openfold_run_reports_missing_local_path() -> Result<(), DbErr> {
        let database = Database::connect("sqlite::memory:").await?;
        database
            .execute(Statement::from_string(
                database.get_database_backend(),
                "PRAGMA foreign_keys = ON".to_owned(),
            ))
            .await?;
        db::migrate_database(&database).await?;
        seed::seed_defaults(&database).await?;
        let missing_path = "definitely-missing-vizfold-local-path";

        let error = queue_openfold_run(
            &database,
            OpenfoldQueueArgs {
                input_id: "6KWC_1".into(),
                input_sequence: "GSTI".into(),
                fasta_dir: Some(missing_path.into()),
                data_dir: Some(".".into()),
                alignment_dir: None,
                model_device: Some("cpu".into()),
                cpus: 1,
                residue_idx: 1,
                demo_attn: false,
                save_outputs: true,
                num_recycles_save: 1,
                use_precomputed_alignments: false,
            },
        )
        .await
        .expect_err("missing local path should fail");

        assert!(error.to_string().contains(
            "--fasta-dir original path 'definitely-missing-vizfold-local-path' could not be resolved"
        ));
        assert!(
            error
                .to_string()
                .contains(&crate::core::config::openfold_home().display().to_string())
        );
        Ok(())
    }

    #[test]
    fn model_device_workstation_defaults_to_cpu_without_a_gpu() {
        assert_eq!(
            super::model_device_for(config::SlurmContext::None, None, None),
            "cpu"
        );
    }

    #[test]
    fn model_device_workstation_defaults_to_cuda_with_a_gpu() {
        assert_eq!(
            super::model_device_for(config::SlurmContext::None, None, Some("NVIDIA A100")),
            "cuda:0"
        );
    }

    #[test]
    fn model_device_prefers_the_configured_gpu_partition_without_probing() {
        // No allocation held + a GPU partition configured: the fold will be srun'd onto a GPU
        // node, so cuda:0 is correct even though the probe result (None here) says no GPU here.
        assert_eq!(
            super::model_device_for(config::SlurmContext::None, Some("gpuA100x4"), None),
            "cuda:0"
        );
    }

    #[test]
    fn model_device_inside_an_allocation_trusts_the_local_probe() {
        // Already on the node the fold runs on, so the local probe -- not the partition config
        // -- decides, even though a partition is configured.
        assert_eq!(
            super::model_device_for(config::SlurmContext::InAllocation, Some("gpuA100x4"), None),
            "cpu"
        );
    }

    #[test]
    fn cpus_default_follows_available_parallelism() {
        let expected = std::thread::available_parallelism().map_or(1, |n| n.get() as i64);
        assert_eq!(super::default_cpus(), expected);
    }
}
