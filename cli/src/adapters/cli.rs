use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
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
    examples,
    output_locations::resolve_output_location,
    preflight::PreflightStatus,
    release,
    seed::seed_defaults,
    services::{
        artifacts, execution_targets, model_backends, model_invocation_profiles,
        run_artifacts::{self, register_known_run_artifacts},
        run_execution::execute_run,
        runs,
    },
};

#[derive(Debug, Parser)]
#[command(
    name = "vizfold",
    version,
    about = "VizFold executor administration CLI",
    arg_required_else_help = true
)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Install a model backend (openfold or esmfold) on this machine.
    Install(InstallArgs),
    /// Download a backend's data (OpenFold AlphaFold2 databases/params).
    Download(DownloadArgs),
    /// Show resolved config, which backends are installed, and whether it all checks out.
    Status,
    /// Remove one backend, or everything the install generated.
    Uninstall(UninstallArgs),
    /// Update the vizfold checkout the installers and dashboard come from.
    Update(UpdateArgs),
    /// Replace this binary with the latest release, then update the checkout to match.
    SelfUpdate(SelfUpdateArgs),
    /// Start the workbench dashboard.
    Serve(ServeArgs),
    /// List executor records.
    List(ListArgs),
    /// Show one executor record.
    Show(ShowArgs),
    /// Queue a run for a supported model backend, without executing it.
    Queue(QueueArgs),
    /// Run a fold: a bundled example, a FASTA, or a queued run by id.
    Run(RunArgs),
    /// Register known artifacts for a completed run.
    RegisterArtifacts { run_id: i32 },
}

#[derive(Debug, Args)]
struct InstallArgs {
    /// Model backend to install.
    #[arg(value_enum)]
    backend: Backend,
}

#[derive(Debug, Args)]
struct DownloadArgs {
    /// Model backend whose data to download.
    #[arg(value_enum)]
    backend: Backend,
    /// Dataset to fetch: `all` (the full AlphaFold2 set) or a single db name (e.g. `uniref90`,
    /// `pdb70`, `bfd`, `alphafold_params`), mapped to `downloaders/<backend>/download_<name>.sh`.
    #[arg(default_value = "all")]
    dataset: String,
}

/// A model backend vizfold supports.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    Openfold,
    Esmfold,
}

impl Backend {
    fn slug(self) -> &'static str {
        match self {
            Self::Openfold => "openfold",
            Self::Esmfold => "esmfold",
        }
    }

    /// Installer script, relative to the vizfold checkout: each backend owns one under
    /// `backends/<name>/install/`.
    fn installer(self) -> &'static str {
        match self {
            Self::Openfold => config::INSTALLER,
            Self::Esmfold => "backends/esmfold/install/install.sh",
        }
    }

    /// Data-download entrypoint, relative to the checkout. `None` for ESMFold: it pulls its
    /// weights from HuggingFace at run time.
    fn downloader_dir(self) -> Option<&'static str> {
        match self {
            Self::Openfold => Some("downloaders/openfold"),
            Self::Esmfold => None,
        }
    }

    fn env_prefix(self) -> PathBuf {
        match self {
            Self::Openfold => config::openfold_env_prefix(),
            Self::Esmfold => config::esmfold_env_prefix(),
        }
    }

    fn is_installed(self) -> bool {
        self.env_prefix().is_dir()
    }

    fn label(self) -> &'static str {
        match self {
            Self::Openfold => "OpenFold",
            Self::Esmfold => "ESMFold",
        }
    }

    /// This backend's subtree of the checkout: its project, its installer, its build droppings.
    fn dir(self, home: &Path) -> PathBuf {
        home.join("backends").join(self.slug())
    }

    /// Exactly what `install <backend>` puts back. Fold outputs are results, never install state.
    fn install_paths(self, prefix: &Path, home: &Path) -> Vec<PathBuf> {
        let backend = self.dir(home);
        // One state dir per backend (`vizfold::state`) covers everything under the prefix.
        let mut paths = vec![self.env_prefix(), prefix.join(self.slug())];
        // `pip install` builds in-tree, so setuptools plants these in the checkout.
        paths.extend(
            ["build", &format!("{}.egg-info", self.slug())].map(|entry| backend.join(entry)),
        );
        if self == Self::Openfold {
            // Where both the install and `vizfold download` write; the state dir is the default.
            paths.push(config::data_dir());
            paths.extend(
                [
                    "openfold/resources/stereo_chemical_props.txt",
                    "tests/test_data/alphafold/common/stereo_chemical_props.txt",
                ]
                .map(|entry| backend.join(entry)),
            );
            // ABI-tagged, so one left behind is importable against the wrong environment.
            paths.extend(
                std::fs::read_dir(&backend)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|ext| ext == "so")),
            );
        }
        paths
    }
}

#[derive(Debug, Args)]
struct UninstallArgs {
    /// Model backend to remove. Omit to remove every backend, the config, and the run database too.
    #[arg(value_enum)]
    backend: Option<Backend>,
    /// Remove without the confirmation prompt.
    #[arg(long, short = 'y')]
    yes: bool,
}

#[derive(Debug, Args)]
struct UpdateArgs {
    /// Tag or branch to move the checkout to. Defaults to this binary's own release tag.
    #[arg(long, value_name = "REF")]
    r#ref: Option<String>,
}

#[derive(Debug, Args)]
struct SelfUpdateArgs {
    /// Release to install (e.g. v0.5.1). Defaults to the latest published release.
    #[arg(long, value_name = "TAG")]
    version: Option<String>,
    /// Re-download even when this binary already is that release.
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Args)]
struct ServeArgs {
    /// Port for the dashboard dev server. Defaults to 3000.
    #[arg(long)]
    port: Option<u16>,
}

#[derive(Debug, Args)]
struct ListArgs {
    #[command(subcommand)]
    resource: ListResource,
}

#[derive(Debug, Args)]
struct RunArgs {
    /// What to fold: a bundled example id (`vizfold list examples`), a path to a FASTA, or a
    /// queued run's id.
    target: String,
    /// Backend. Defaults to the only one installed, else openfold. A queued run carries its own.
    #[arg(long, value_enum)]
    backend: Option<Backend>,
    /// Dump per-layer, per-head attention maps (OpenFold). A queued run keeps what it was queued with.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    attn: bool,
    /// Print only the run as JSON, for tools driving the CLI.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum ListResource {
    /// List the bundled examples that can fold without an MSA search.
    Examples {
        /// Emit JSON including each sequence, for tools driving the CLI.
        #[arg(long)]
        json: bool,
    },
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
struct QueueArgs {
    #[command(subcommand)]
    model: QueueModel,
}

#[derive(Clone, Debug, Subcommand)]
enum QueueModel {
    /// Queue an OpenFold run.
    Openfold(OpenfoldQueueArgs),
    /// Queue an ESMFold run.
    Esmfold(EsmfoldQueueArgs),
}

#[derive(Clone, Debug, Args)]
struct EsmfoldQueueArgs {
    /// Name recorded for this run. Defaults to the FASTA's header tag.
    #[arg(long)]
    input_id: Option<String>,
    /// FASTA to fold: the file, or a directory holding exactly one.
    #[arg(long)]
    fasta: String,
    /// Torch device. Defaults to cuda:0 when a GPU partition is configured to srun onto (the
    /// HPC flow) or a GPU is visible locally, otherwise cpu.
    #[arg(long)]
    model_device: Option<String>,
    /// HuggingFace model id.
    #[arg(long, default_value = "facebook/esmfold_v1")]
    model: String,
    /// What to extract: none, attention, activations, or attention+activations.
    #[arg(long, default_value = "attention+activations")]
    trace_mode: String,
    /// Layers to save: `all` or a comma/colon list.
    #[arg(long, default_value = "all")]
    layers: String,
    /// Model dtype.
    #[arg(long, default_value = "float32")]
    dtype: String,
    /// Save trace tensors in fp16 to reduce size.
    #[arg(long)]
    save_fp16: bool,
    /// Capture IPA attention and per-recycle backbone from the structure module.
    #[arg(long)]
    structure_traces: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Args)]
struct OpenfoldQueueArgs {
    /// Name recorded for this run. Defaults to the FASTA's header tag, the only value preflight takes.
    #[arg(long)]
    input_id: Option<String>,
    /// FASTA to fold: the file, or a directory holding exactly one.
    /// Defaults to <OPENFOLD_HOME>/examples/monomer/fasta_dir_<id>.
    #[arg(long)]
    fasta: Option<String>,
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
    /// CPU threads. Defaults to this machine's core count, clamped to the execution target's maximum.
    #[arg(long)]
    cpus: Option<i64>,
    /// Residue index offset passed through to the model.
    #[arg(long, default_value_t = 1)]
    residue_idx: i64,
    /// Dump per-layer, per-head attention maps.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    attn: bool,
    /// Write the model's raw output tensors alongside the structure.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    save_outputs: bool,
    /// How many recycling iterations to keep outputs for.
    #[arg(long, default_value_t = 1)]
    num_recycles_save: i64,
    /// Use the precomputed alignments in `--alignment-dir`. Pass
    /// `--use-precomputed-alignments=false` for the full MSA pipeline.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    use_precomputed_alignments: bool,
}

impl OpenfoldQueueArgs {
    /// The same defaults clap gives `queue openfold`; a test pins the two together.
    fn for_example(example: &examples::Example, attn: bool) -> Self {
        Self {
            input_id: Some(example.id.clone()),
            attn,
            fasta: None,
            data_dir: None,
            alignment_dir: None,
            model_device: None,
            cpus: None,
            residue_idx: 1,
            save_outputs: true,
            num_recycles_save: 1,
            use_precomputed_alignments: true,
        }
    }
}

/// A GPU partition with no allocation held means the fold is srun'd onto a GPU node regardless of this host.
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

/// Skips the local probe when it would not be consulted, rather than discarding its result.
fn default_model_device() -> String {
    let context = config::SlurmContext::detect();
    let partition = config::gpu_partition();
    let detected = if on_gpu_partition(context, partition.as_deref()) {
        None
    } else {
        crate::core::preflight::detect_gpu()
    };
    model_device_for(context, partition.as_deref(), detected.as_deref())
}

fn default_cpus() -> i64 {
    std::thread::available_parallelism().map_or(1, |n| n.get() as i64)
}

/// Clamp to the target's `cpus.maximum`, so a host with more cores still queues a runnable plan.
fn clamp_cpus(cpus: i64, available_resources_json: &str) -> i64 {
    let max_cpus = serde_json::from_str::<serde_json::Value>(available_resources_json)
        .ok()
        .and_then(|resources| resources["properties"]["cpus"]["maximum"].as_i64())
        .unwrap_or(i64::MAX);
    cpus.min(max_cpus)
}

/// What each command needs sound before it starts, in `health`'s vocabulary -- the one place a
/// prerequisite is refused, so no command grows its own check. `Status` names none: it reports.
fn prereqs(command: &Command) -> Vec<Component> {
    // `run <id>` replays the run's own backend, which this gate cannot read without the DB; it
    // checks the default instead, so uninstalling a backend out from under a queued run still
    // reaches the runner -- which then names the missing interpreter by path.
    let backend = |explicit: Option<Backend>| {
        backend_health(
            explicit
                .or_else(|| default_backend().ok())
                .unwrap_or(Backend::Openfold),
        )
    };
    match command {
        Command::Status | Command::Uninstall(_) | Command::Update(_) | Command::SelfUpdate(_) => {
            vec![]
        }
        Command::Install(_) => vec![core_deps_health()],
        Command::List(ListArgs {
            resource: ListResource::Examples { .. },
        }) => vec![repo_health()],
        Command::Serve(_) => vec![core_deps_health(), repo_health(), config_health()],
        Command::Download(args) => vec![config_health(), backend(Some(args.backend))],
        Command::Queue(args) => vec![
            config_health(),
            backend(Some(match args.model {
                QueueModel::Openfold(_) => Backend::Openfold,
                QueueModel::Esmfold(_) => Backend::Esmfold,
            })),
        ],
        Command::Run(args) => vec![core_deps_health(), config_health(), backend(args.backend)],
        _ => vec![config_health()],
    }
}

/// Ok and Unverified proceed: an unreachable scheduler must not stop a local fold.
fn refuses(state: State) -> bool {
    matches!(state, State::Absent | State::Broken)
}

pub async fn run() -> Result<(), DbErr> {
    let cli = Cli::parse();

    for component in prereqs(&cli.command).into_iter().map(settled) {
        if refuses(component.state) {
            eprintln!("{}: {}", component.name, component.detail);
            for problem in &component.problems {
                eprintln!("  {problem}");
            }
            eprintln!("  -> {}", component.remedy);
            std::process::exit(1);
        }
    }

    // These touch the filesystem only; they need no database connection.
    match cli.command {
        Command::Install(args) => return run_install(args.backend),
        Command::Download(args) => return run_download(args.backend, args.dataset),
        Command::Status => return run_status(),
        Command::Uninstall(args) => return run_uninstall(args),
        Command::Update(args) => return run_update(args.r#ref.as_deref()),
        Command::SelfUpdate(args) => return run_self_update(args),
        Command::Serve(args) => return run_serve(args),
        Command::List(ListArgs {
            resource: ListResource::Examples { json },
        }) => return list_examples(json),
        _ => {}
    }

    let database = db::connect_and_migrate().await?;
    match cli.command {
        Command::List(list) => match list.resource {
            ListResource::Models => list_models(&database).await?,
            ListResource::Targets => list_targets(&database).await?,
            ListResource::Profiles => list_profiles(&database).await?,
            ListResource::Runs { status } => list_runs(&database, status.as_deref()).await?,
            ListResource::Examples { .. } => unreachable!("handled before DB connect"),
        },
        Command::Show(show) => match show.resource {
            ShowResource::Run { run_id } => show_run(&database, run_id).await?,
        },
        Command::Queue(queue) => match queue.model {
            QueueModel::Openfold(args) => queue_openfold_run(&database, args).await?,
            QueueModel::Esmfold(args) => queue_esmfold_run(&database, args).await?,
        },
        Command::Run(args) => run_run(&database, args).await?,
        Command::RegisterArtifacts { run_id } => register_artifacts(&database, run_id).await?,
        Command::Install(_)
        | Command::Download(_)
        | Command::Status
        | Command::Uninstall(_)
        | Command::Update(_)
        | Command::SelfUpdate(_)
        | Command::Serve(_) => {
            unreachable!("handled before DB connect")
        }
    }

    Ok(())
}

/// Run the checkout's installer, cloning the checkout first -- the binary ships only itself. Idempotent.
fn run_install(backend: Backend) -> Result<(), DbErr> {
    let src = config::vizfold_src();
    let installer = src.join(backend.installer());
    if !installer.is_file() {
        clone_checkout(&src)?;
    }
    if !installer.is_file() {
        return Err(DbErr::Custom(format!(
            "no {} installer at {}; set OPENFOLD_HOME to a checkout",
            backend.slug(),
            installer.display()
        )));
    }
    println!(
        "Installing {}: bash {}",
        backend.slug(),
        installer.display()
    );
    run_to_completion(
        "model install",
        std::process::Command::new("bash")
            .arg(&installer)
            .env("OPENFOLD_HOME", &src),
    )
}

/// Run a child to completion with inherited stdio, naming it in either failure.
fn run_to_completion(what: &str, command: &mut std::process::Command) -> Result<(), DbErr> {
    let status = command
        .status()
        .map_err(|error| DbErr::Custom(format!("failed to launch {what}: {error}")))?;
    status
        .success()
        .then_some(())
        // ExitStatus already renders as "exit status: N", so no "exited with status" prefix.
        .ok_or_else(|| DbErr::Custom(format!("{what}: {status}")))
}

/// Download a backend's data into `config::data_dir()`, cloning the checkout if absent.
fn run_download(backend: Backend, dataset: String) -> Result<(), DbErr> {
    let Some(dir) = backend.downloader_dir() else {
        println!(
            "{} fetches its weights from HuggingFace at run time; nothing to download.",
            backend.slug()
        );
        return Ok(());
    };
    let src = config::vizfold_src();
    if !src.join(dir).is_dir() {
        clone_checkout(&src)?;
    }
    let script_name = if dataset == "all" {
        "download_alphafold_dbs.sh".to_string()
    } else {
        format!("download_{dataset}.sh")
    };
    let script = src.join(dir).join(&script_name);
    if !script.is_file() {
        return Err(DbErr::Custom(format!(
            "no downloader `{script_name}` at {}; pass `all` or a db name (e.g. uniref90, pdb70, bfd)",
            script.display()
        )));
    }
    let dest = config::data_dir();
    std::fs::create_dir_all(&dest)
        .map_err(|error| DbErr::Custom(format!("cannot create {}: {error}", dest.display())))?;
    println!(
        "Downloading {} `{dataset}`: bash {} {}",
        backend.slug(),
        script.display(),
        dest.display()
    );
    // The downloaders need aria2c and aws, which live only inside the OpenFold environment.
    let path = std::env::var("PATH").unwrap_or_default();
    let env_bin = config::openfold_env_prefix().join("bin");
    run_to_completion(
        "downloader",
        std::process::Command::new("bash")
            .arg(&script)
            .arg(&dest)
            .env("OPENFOLD_HOME", &src)
            .env("PATH", format!("{}:{path}", env_bin.display())),
    )
}

/// Runs before any install, so it needs no database.
fn run_status() -> Result<(), DbErr> {
    println!("VizFold status\n");
    let components = health();
    print_table(
        &["COMPONENT", "STATUS", "DETAIL"],
        components.iter().map(|component| {
            vec![
                component.name.to_owned(),
                component.state.label().to_owned(),
                component.detail.clone(),
            ]
        }),
    );
    let broken = components.iter().filter(|c| !c.problems.is_empty());
    for (nth, component) in broken.enumerate() {
        println!("{}", if nth == 0 { "\nProblems:" } else { "" });
        for problem in &component.problems {
            println!("  {}: {problem}", component.name);
        }
        println!("  -> {}", component.remedy);
    }
    println!("\n{}\n", summary(&components));

    let config_file = config::config_file();
    if !config::is_initialized() {
        println!("Config: {} (not initialized)", config_file.display());
        return Ok(());
    }
    println!("Config: {}", config_file.display());
    for (key, value) in config::config_entries() {
        // What the checks above read: an inline env var can differ from the file.
        match std::env::var(&key) {
            Ok(inline) if !inline.is_empty() && inline != value => {
                println!("  {key} = {inline}  (env, overriding {value:?})");
            }
            _ => println!("  {key} = {value}"),
        }
    }
    if let Some(database) = config::database_path() {
        let state = if database.is_file() {
            "present"
        } else {
            "not created yet"
        };
        println!("  database = {} ({state})", database.display());
    }
    Ok(())
}

/// One independently breakable part. `health` derives `Broken` from the problem list, so no builder tracks both.
#[derive(Default)]
struct Component {
    name: &'static str,
    state: State,
    detail: String,
    problems: Vec<String>,
    /// Printed under the problems, so it can be set whether or not any were found.
    remedy: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum State {
    #[default]
    Ok,
    /// Nothing installed it -- most installs run one backend.
    Absent,
    /// Could not be checked from here: no scheduler, no network.
    Unverified,
    Broken,
}

impl State {
    fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Absent => "absent",
            Self::Unverified => "unverified",
            Self::Broken => "BROKEN",
        }
    }
}

/// `Broken` is derived, never tracked: a component that found problems is broken by definition.
fn settled(component: Component) -> Component {
    match component.problems.is_empty() {
        true => component,
        false => Component {
            state: State::Broken,
            ..component
        },
    }
}

/// Everything `vizfold status` can settle about an install without folding anything.
fn health() -> Vec<Component> {
    [
        core_deps_health(),
        binary_health(),
        repo_health(),
        config_health(),
        backend_health(Backend::Openfold),
        backend_health(Backend::Esmfold),
        scheduler_health(),
    ]
    .into_iter()
    .map(settled)
    .collect()
}

/// Executable, not merely present: a failed fetch leaves the truncated file `install.sh` skips over.
fn executable(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| {
        meta.is_file() && std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o111 != 0
    })
}

/// PATH lookup, so nothing has to record where the bootstrap put a core dependency.
fn on_path(program: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(program))
        .find(|path| executable(path))
}

/// What `install.sh` puts beside the vizfold binary. Every environment is created and run through it.
/// The one binary `install.sh` bootstraps and everything downstream resolves off PATH.
const CORE_DEP: &str = "micromamba";

fn core_deps_health() -> Component {
    let found = on_path(CORE_DEP);
    Component {
        name: "core deps",
        detail: found
            .as_ref()
            .map_or_else(|| CORE_DEP.to_owned(), |path| path.display().to_string()),
        problems: found
            .is_none()
            .then(|| format!("no executable `{CORE_DEP}` on PATH"))
            .into_iter()
            .collect(),
        remedy: format!(
            "curl -fsSL https://raw.githubusercontent.com/{}/main/install.sh | bash",
            release::repo()
        ),
        ..Default::default()
    }
}

fn binary_health() -> Component {
    Component {
        name: "binary",
        detail: release::version_line(release::latest_tag().as_deref()),
        ..Default::default()
    }
}

/// The checkout everything runs from. Drift is flagged only for the clone vizfold made itself.
fn repo_health() -> Component {
    let src = config::vizfold_src();
    let at = checkout_ref(&src);
    let expected = release::tag();
    let problems = match &at {
        _ if !src.join(config::INSTALLER).is_file() => {
            vec![format!("{} is not a vizfold checkout", src.display())]
        }
        Some(at) if *at != expected && src == config::default_src() => {
            vec![format!(
                "the scripts are {at}, but this binary is {expected}"
            )]
        }
        _ => Vec::new(),
    };
    Component {
        name: "repo",
        detail: match &at {
            Some(at) => format!("{} at {at}", src.display()),
            None => src.display().to_string(),
        },
        problems,
        remedy: "vizfold update".to_owned(),
        ..Default::default()
    }
}

/// Path keys, each with the file proving the claim (empty: the path itself). All must be in `CONFIG_KEYS`.
const CHECKED_PATHS: &[(&str, &str)] = &[
    ("OPENFOLD_HOME", config::INSTALLER),
    ("OPENFOLD_PREFIX", ""),
    ("VIZFOLD_ENV_BASE", ""),
];

/// The same, but only while OpenFold is installed: its own uninstall must not read as a broken config.
const OPENFOLD_PATHS: &[(&str, &str)] = &[("OPENFOLD_DATA_DIR", "")];

/// Config keys only the scheduler can settle, grouped by the one question that answers them.
const CHECKED_PARTITIONS: &[&str] = &["OPENFOLD_PARTITION", "OPENFOLD_GPU_PARTITION"];
const CHECKED_ACCOUNTS: &[&str] = &["OPENFOLD_ACCOUNT", "OPENFOLD_GPU_ACCOUNT"];

fn config_health() -> Component {
    if !config::is_initialized() {
        return Component {
            name: "config",
            state: State::Absent,
            detail: "not initialized".to_owned(),
            remedy: "vizfold install <backend>".to_owned(),
            ..Default::default()
        };
    }
    let openfold_paths: &[(&str, &str)] = if Backend::Openfold.is_installed() {
        OPENFOLD_PATHS
    } else {
        &[]
    };
    let mut problems: Vec<String> = schema_problem().into_iter().collect();
    problems.extend(
        CHECKED_PATHS
            .iter()
            .chain(openfold_paths)
            .filter_map(|(key, marker)| path_problem(key, marker)),
    );
    Component {
        name: "config",
        detail: format!("{} keys", config::config_keys().len()),
        problems,
        remedy: "vizfold install <backend>".to_owned(),
        ..Default::default()
    }
}

/// A different key set means a different installer wrote it, so every value under it is suspect.
fn schema_problem() -> Option<String> {
    let present = config::config_keys();
    let missing: Vec<&str> = config::CONFIG_KEYS
        .iter()
        .filter(|key| !present.iter().any(|had| had == *key))
        .copied()
        .collect();
    let unknown: Vec<&str> = present
        .iter()
        .filter(|key| !config::CONFIG_KEYS.contains(&key.as_str()))
        .map(String::as_str)
        .collect();
    if missing.is_empty() && unknown.is_empty() {
        return None;
    }
    Some(format!(
        "written by a different vizfold ({})",
        [
            (!missing.is_empty()).then(|| format!("missing {}", missing.join(", "))),
            (!unknown.is_empty()).then(|| format!("unknown {}", unknown.join(", "))),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("; ")
    ))
}

fn path_problem(key: &str, marker: &str) -> Option<String> {
    let value = config::resolved(key)?;
    let path = PathBuf::from(&value);
    let proof = if marker.is_empty() {
        path
    } else {
        path.join(marker)
    };
    if proof.exists() {
        return None;
    }
    Some(if marker.is_empty() {
        format!("{key} = {value}, which does not exist")
    } else {
        format!("{key} = {value}, which holds no {marker}")
    })
}

/// A backend's own installation: its environment runs, and its fold inputs are where they belong.
fn backend_health(backend: Backend) -> Component {
    let env = backend.env_prefix();
    if !backend.is_installed() {
        return Component {
            name: backend.slug(),
            state: State::Absent,
            detail: format!("not installed ({})", env.display()),
            remedy: format!("vizfold install {}", backend.slug()),
            ..Default::default()
        };
    }
    let mut problems: Vec<String> = missing("interpreter", env.join("bin/python"))
        .into_iter()
        .collect();
    // The backend's own CLI, in the environment that runs it.
    problems.extend(missing("entrypoint", env.join("bin").join(backend.slug())));
    if backend == Backend::Openfold {
        problems.extend(params_problem());
        problems.extend(example_problem());
    }
    Component {
        name: backend.slug(),
        detail: env.display().to_string(),
        problems,
        remedy: format!("vizfold install {}", backend.slug()),
        ..Default::default()
    }
}

fn missing(what: &str, path: PathBuf) -> Option<String> {
    (!path.is_file()).then(|| format!("no {what} at {}", path.display()))
}

/// Weights resolve under the one data root: `$OPENFOLD_DATA_DIR/params/params_<preset>.npz`, nothing else.
fn params_problem() -> Option<String> {
    let weights = config::data_dir().join("params/params_model_1_ptm.npz");
    (!weights.exists()).then(|| {
        format!(
            "AlphaFold2 parameters missing or a dangling link: {}",
            weights.display()
        )
    })
}

/// Half an example -- a FASTA with no alignments -- silently falls back to a full MSA search.
fn example_problem() -> Option<String> {
    let id = config::resolved("OPENFOLD_EXAMPLE")?;
    examples::find(&id).is_none().then(|| {
        format!(
            "no FASTA and alignments for OPENFOLD_EXAMPLE {id} under {}",
            examples::monomer_dir().display()
        )
    })
}

/// The difference between a fold that queues and one rejected at submission.
fn scheduler_health() -> Component {
    let wanted: Vec<String> = CHECKED_PARTITIONS
        .iter()
        .chain(CHECKED_ACCOUNTS)
        .filter_map(|key| config::resolved(key))
        .collect();
    if wanted.is_empty() {
        return Component {
            name: "scheduler",
            state: State::Absent,
            detail: "no accounts or partitions configured".to_owned(),
            ..Default::default()
        };
    }
    // %100P, not %P: the default width truncates, so a long name reads as a missing partition.
    let partitions = scheduler_values("sinfo", &["-h", "-o", "%100P"]);
    let user = format!("user={}", std::env::var("USER").unwrap_or_default());
    let accounts = scheduler_values(
        "sacctmgr",
        &["-nP", "show", "assoc", &user, "format=Account"],
    );
    if partitions.is_none() && accounts.is_none() {
        return Component {
            name: "scheduler",
            state: State::Unverified,
            detail: format!("{} unchecked: no scheduler here", wanted.join(", ")),
            ..Default::default()
        };
    }
    let problems: Vec<String> = [
        (CHECKED_PARTITIONS, &partitions, "partition"),
        (CHECKED_ACCOUNTS, &accounts, "account"),
    ]
    .into_iter()
    .flat_map(|(keys, known, noun)| {
        keys.iter().filter_map(move |key| {
            unknown_to_scheduler(key, &config::resolved(key)?, known.as_deref(), noun)
        })
    })
    .collect();
    Component {
        name: "scheduler",
        detail: wanted.join(", "),
        problems,
        remedy: "correct them in ~/.config/vizfold/vizfold.json".to_owned(),
        ..Default::default()
    }
}

/// What the scheduler has. `None` means it could not be asked (no command, or slurmctld unreachable).
fn scheduler_values(program: &str, args: &[&str]) -> Option<Vec<String>> {
    let output = output_within(
        std::process::Command::new(program).args(args),
        SCHEDULER_TIMEOUT,
    )?;
    if !output.status.success() {
        return None;
    }
    let values = scheduler_names(&String::from_utf8_lossy(&output.stdout));
    (!values.is_empty()).then_some(values)
}

/// Where slurmctld is unreachable, sinfo and sacctmgr block for Slurm's MessageTimeout instead.
const SCHEDULER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// Output, or `None` if unrunnable or killed for running long. Only for output that cannot fill a pipe.
fn output_within(
    command: &mut std::process::Command,
    limit: std::time::Duration,
) -> Option<std::process::Output> {
    let mut child = command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let deadline = std::time::Instant::now() + limit;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Err(_) => return None,
            Ok(None) if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(25)),
        }
    }
}

/// One name per line. Trim before stripping `sinfo`'s default-partition `*` -- its padding comes after.
fn scheduler_names(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .map(|line| line.trim().trim_end_matches('*').to_owned())
        .filter(|line| !line.is_empty())
        .collect()
}

/// A name the scheduler does not have. `None` known set = unanswered, which is not evidence against it.
fn unknown_to_scheduler(
    key: &str,
    value: &str,
    known: Option<&[String]>,
    noun: &str,
) -> Option<String> {
    let known = known?;
    (!known.iter().any(|name| name == value))
        .then(|| format!("{key} names {noun} {value}, which this cluster does not have"))
}

/// The last line of `status`: what to do, in one sentence.
fn summary(components: &[Component]) -> String {
    let broken: Vec<&str> = components
        .iter()
        .filter(|component| component.state == State::Broken)
        .map(|component| component.name)
        .collect();
    if broken.is_empty() {
        return "Everything checks out.".to_owned();
    }
    format!(
        "{} of {} components need attention: {}.",
        broken.len(),
        components.len(),
        broken.join(", ")
    )
}

/// Clone the checkout, pinned to `release::tag()` -- the default branch when no such tag exists.
fn clone_checkout(src: &std::path::Path) -> Result<(), DbErr> {
    let url = format!("https://github.com/{}.git", release::repo());
    let dest = src.to_string_lossy().into_owned();
    println!("Fetching the vizfold checkout into {dest} ...");
    let clone = |args: &[&str]| std::process::Command::new("git").args(args).status();
    if let Ok(status) = clone(&[
        "clone",
        "--depth",
        "1",
        "--branch",
        &release::tag(),
        &url,
        &dest,
    ]) && status.success()
    {
        return Ok(());
    }
    match clone(&["clone", "--depth", "1", &url, &dest]) {
        Ok(status) if status.success() => Ok(()),
        _ => Err(DbErr::Custom(format!(
            "failed to clone {url} into {dest}; set OPENFOLD_HOME to an existing checkout"
        ))),
    }
}

/// Move the checkout to a ref, by default this binary's own release tag. Install droppings are gitignored.
fn run_update(wanted: Option<&str>) -> Result<(), DbErr> {
    let src = config::vizfold_src();
    let target = wanted.unwrap_or(&release::tag()).to_owned();
    if !src.join(config::INSTALLER).is_file() {
        return clone_checkout(&src);
    }
    if !src.join(".git").exists() {
        return Err(DbErr::Custom(format!(
            "{} is not a git checkout; nothing to update",
            src.display()
        )));
    }
    // Tracked edits only: the install builds OpenFold's CUDA extension in this checkout.
    match git(&src, &["status", "--porcelain", "--untracked-files=no"]) {
        None => {
            return Err(DbErr::Custom(format!(
                "cannot read `git status` in {}; is git on PATH and the checkout yours to read?",
                src.display()
            )));
        }
        Some(changes) if !changes.trim().is_empty() => {
            return Err(DbErr::Custom(format!(
                "{} has uncommitted changes; commit or discard them first",
                src.display()
            )));
        }
        _ => {}
    }
    println!("Updating {} to {target} ...", src.display());
    // Shallow single-branch clone: the ref must be fetched by name, and only FETCH_HEAD names it after.
    run_to_completion(
        "fetch",
        &mut git_cmd(
            &src,
            &["fetch", "--depth", "1", "--tags", "origin", &target],
        ),
    )?;
    run_to_completion(
        "checkout",
        &mut git_cmd(&src, &["checkout", "--force", "FETCH_HEAD"]),
    )?;
    println!(
        "{} is now at {}",
        src.display(),
        checkout_ref(&src).unwrap_or(target)
    );
    Ok(())
}

fn git_cmd(dir: &Path, args: &[&str]) -> std::process::Command {
    let mut command = std::process::Command::new("git");
    command.arg("-C").arg(dir).args(args);
    command
}

/// One read-only git question as trimmed stdout. `None` when git cannot answer at all.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = git_cmd(dir, args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// What the checkout is actually on: its tag if it sits exactly on one, else a short commit.
fn checkout_ref(src: &Path) -> Option<String> {
    git(src, &["describe", "--tags", "--exact-match"])
        .or_else(|| git(src, &["rev-parse", "--short", "HEAD"]))
        .filter(|value| !value.is_empty())
}

/// Replace this binary, then let the new one update its own checkout. Staged beside it: rename is per-fs.
fn run_self_update(args: SelfUpdateArgs) -> Result<(), DbErr> {
    let exe = std::env::current_exe()
        .map_err(|error| DbErr::Custom(format!("cannot locate this binary: {error}")))?;
    let current = release::current();
    let tag = match args.version {
        Some(version) => version,
        None => release::latest_tag().ok_or_else(|| {
            DbErr::Custom(
                "could not reach github.com for the latest release; pass --version to pin one"
                    .to_owned(),
            )
        })?,
    };
    let wanted = release::version_of(&tag).to_owned();
    if wanted == current && !args.force {
        println!("vizfold {current} is already the latest release.");
        return Ok(());
    }
    let asset = release::asset(std::env::consts::ARCH);
    let url = release::asset_url(&tag, &asset);
    let staged = exe.with_file_name(format!(".{asset}.incoming"));
    println!("Updating vizfold {current} -> {wanted}\n  {url}");
    let fetched = fetch_release(&url, &staged, &wanted);
    if fetched.is_err() {
        let _ = std::fs::remove_file(&staged);
    }
    fetched?;
    std::fs::rename(&staged, &exe).map_err(|error| {
        let _ = std::fs::remove_file(&staged);
        DbErr::Custom(format!(
            "cannot replace {}: {error}; re-run where you can write it",
            exe.display()
        ))
    })?;
    println!("vizfold {wanted} is installed at {}", exe.display());

    // The new binary, not this one, knows which checkout its own scripts came from.
    println!();
    match std::process::Command::new(&exe).arg("update").status() {
        Ok(status) if status.success() => Ok(()),
        _ => {
            println!("The checkout could not be updated; run `vizfold update` yourself.");
            Ok(())
        }
    }
}

/// Prove the download is a working binary of the version it claims before it replaces anything.
fn fetch_release(url: &str, staged: &Path, wanted: &str) -> Result<(), DbErr> {
    run_to_completion(
        "download",
        std::process::Command::new("curl")
            .args(["-fSL", url, "-o"])
            .arg(staged),
    )?;
    std::fs::set_permissions(staged, std::os::unix::fs::PermissionsExt::from_mode(0o755))
        .map_err(|error| DbErr::Custom(format!("cannot make the download executable: {error}")))?;
    let reported = std::process::Command::new(staged)
        .arg("--version")
        .output()
        .map_err(|error| DbErr::Custom(format!("the download will not run here: {error}")))?;
    let reported = String::from_utf8_lossy(&reported.stdout).trim().to_owned();
    if !reported.contains(wanted) {
        return Err(DbErr::Custom(format!(
            "the download reports itself as {reported:?}, not {wanted}; leaving this binary alone"
        )));
    }
    Ok(())
}

/// Undo `vizfold install`. Not a script, because the checkout holding it is one of the things removed.
fn run_uninstall(args: UninstallArgs) -> Result<(), DbErr> {
    let (prefix, home) = (config::prefix(), config::openfold_home());
    let targets = match args.backend {
        Some(backend) => backend.install_paths(&prefix, &home),
        None => [Backend::Openfold, Backend::Esmfold]
            .into_iter()
            .flat_map(|backend| backend.install_paths(&prefix, &home))
            .chain(shared_paths(&prefix, &home))
            .collect(),
    };

    let targets = removal_plan(targets);
    let what = args.backend.map_or("vizfold", Backend::slug);
    if targets.is_empty() {
        println!("Nothing to remove for {what}.");
        return Ok(());
    }

    println!("This removes everything {what} installed:");
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
    match args.backend {
        Some(_) => println!(
            "\nKept: the config, the run database, and every other backend.\nReinstall with: vizfold install {what}"
        ),
        None => {
            // Only once emptied: remove_dir refuses otherwise, which is the whole guard.
            for dir in [
                config::config_file().parent().map(Path::to_path_buf),
                Some(config::env_base()),
                Some(prefix.join("envs")),
            ]
            .into_iter()
            .flatten()
            {
                let _ = std::fs::remove_dir(dir);
            }
            println!(
                "\nKept: fold outputs, the vizfold checkout, and the binaries in ~/.local/bin (vizfold, micromamba)."
            );
        }
    }
    Ok(())
}

/// What `uninstall` removes, and prints for confirmation. A relative path means an empty config
/// value resolved into one -- never delete off the cwd.
fn removal_plan(mut targets: Vec<PathBuf>) -> Vec<PathBuf> {
    targets.retain(|path| path.is_absolute() && std::fs::symlink_metadata(path).is_ok());
    targets.sort();
    targets.dedup();
    // Drop what an outer target already covers. ponytail: O(n^2) over ~25 paths.
    let outer = targets.clone();
    targets.retain(|path| {
        !outer
            .iter()
            .any(|other| other != path && path.starts_with(other))
    });
    targets
}

/// What no backend owns, so only a full uninstall removes it. The checkout only if vizfold cloned it.
fn shared_paths(prefix: &Path, home: &Path) -> Vec<PathBuf> {
    // Named entries, never the env base: only the `vizfold-` ones under it are ours.
    let mut paths = vec![
        config::env_dir("workbench"),
        prefix.join("vizfold.db"),
        config::config_file(),
        // The package cache outlives any one backend; the binary is bootstrap state, beside `vizfold` itself.
        prefix.join("mamba"),
    ];
    // The staged copy only; with no prefix settled the two are one path, the source tree.
    if prefix != home {
        paths.push(prefix.join("workbench"));
    }
    let src = config::vizfold_src();
    if src == config::default_src() && !prefix.starts_with(&src) {
        paths.push(src);
    }
    if let Some(database) = config::database_path() {
        let sidecar = |suffix| PathBuf::from(format!("{}{suffix}", database.display()));
        paths.extend([sidecar("-wal"), sidecar("-shm"), database]);
    }
    paths
}

/// `symlink_metadata`, so an AF2 mirror link is removed rather than followed into its directory.
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

    // Next serves the run outputs off public/, with no file-serving code of ours.
    // ponytail: seeded output_location only; read it from provenance if a profile ever differs.

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

    let node_bin = ensure_node()?;

    let node_modules = workbench.join("node_modules");
    let empty =
        std::fs::read_dir(&node_modules).map_or(true, |mut entries| entries.next().is_none());
    if empty {
        println!("Installing workbench dependencies (npm install)...");
        run_npm(&workbench, &node_bin, &["install"])?;
    }

    let port = args.port.unwrap_or(3000);
    println!("Starting workbench at http://localhost:{port}");
    let port_arg = port.to_string();
    let mut npm_args = vec!["run", "dev"];
    if args.port.is_some() {
        npm_args.extend(["--", "--port", &port_arg]);
    }
    run_npm(&workbench, &node_bin, &npm_args)
}

/// Where the dashboard runs from: a copy on the prefix's filesystem, so node_modules never lands on home.
fn serve_dir() -> Result<PathBuf, DbErr> {
    let src = config::openfold_home().join("workbench");
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

/// Merge `src` into `dst`, skipping the named top-level entries (build output, neither copied nor clobbered).
fn copy_tree(src: &Path, dst: &Path, skip: &[&str]) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_str().is_some_and(|n| skip.contains(&n)) {
            continue;
        }
        let file_type = entry.file_type()?;
        // fs::copy follows a symlink, and a link-to-directory hits EISDIR. Recreated at serve time.
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

/// The workbench needs Node >=22.13 for `node:sqlite`.
const MIN_NODE: (u32, u32) = (22, 13);

/// Unparseable reads as too old: provisioning a known-good Node beats a failure deep in `next dev`.
fn node_is_new_enough(version: &str) -> bool {
    let mut parts = version.trim().trim_start_matches('v').split('.');
    let (Some(Ok(major)), Some(Ok(minor))) = (
        parts.next().map(str::parse::<u32>),
        parts.next().map(str::parse::<u32>),
    ) else {
        return false;
    };
    (major, minor) >= MIN_NODE
}

/// The bin directory of a usable Node already on PATH, if there is one.
fn system_node_bin() -> Option<PathBuf> {
    let output = std::process::Command::new("node")
        .args([
            "-e",
            "process.stdout.write(process.versions.node + '\\n' + process.execPath)",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let (version, path) = text.split_once('\n')?;
    node_is_new_enough(version)
        .then(|| PathBuf::from(path).parent().map(Path::to_path_buf))
        .flatten()
}

/// Node from PATH when new enough, else its own env -- kept out of every backend env. Returns the bin dir.
fn ensure_node() -> Result<PathBuf, DbErr> {
    let env_dir = config::env_dir("workbench");
    let bin = env_dir.join("bin");
    if bin.join("node").is_file() {
        return Ok(bin);
    }
    if let Some(system) = system_node_bin() {
        return Ok(system);
    }
    println!("Provisioning Node (first run only)...");
    // --no-rc so a user ~/.condarc envs_dirs/channels can't hijack it, as the backend installs do.
    run_to_completion(
        "provisioning Node",
        std::process::Command::new("micromamba")
            .args([
                "create",
                "-y",
                "--no-rc",
                "-c",
                "conda-forge",
                "nodejs>=22.13",
                "-p",
            ])
            .arg(&env_dir)
            .env("MAMBA_ROOT_PREFIX", config::prefix().join("mamba")),
    )?;
    Ok(bin)
}

fn run_npm(dir: &Path, node_bin: &Path, args: &[&str]) -> Result<(), DbErr> {
    let mut command = std::process::Command::new(node_bin.join("npm"));
    command.current_dir(dir).args(args);
    // npm's shebang and `next`'s spawn both resolve `node` off PATH, so the env has to lead it.
    let path = std::env::var_os("PATH").unwrap_or_default();
    let dirs = std::iter::once(node_bin.to_path_buf()).chain(std::env::split_paths(&path));
    command.env(
        "PATH",
        std::env::join_paths(dirs)
            .map_err(|error| DbErr::Custom(format!("failed to build PATH: {error}")))?,
    );
    // The dashboard shells out to this binary, and logs each background fold under the prefix.
    if let Ok(binary) = std::env::current_exe() {
        command.env("VIZFOLD_BIN", binary);
    }
    // npm caches in $HOME/.npm by default -- inodes on the quota'd home this staging exists to avoid.
    command.env(
        "npm_config_cache",
        config::env_dir("workbench").join(".npm"),
    );
    command.env("OPENFOLD_PREFIX", config::prefix());
    // node:sqlite cannot open database_url()'s sqlite://...?mode=rwc wrapper; hand it the plain path.
    if let Some(database) = config::database_path() {
        command.env("VIZFOLD_DB", database);
    }
    run_to_completion(&format!("npm {}", args.join(" ")), &mut command)
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
    if !matches!(backend.slug.as_str(), "openfold" | "esmfold") {
        return Err(DbErr::Custom(format!(
            "artifact registration is currently only implemented for OpenFold and ESMFold runs (run {run_id} uses backend '{}')",
            backend.slug
        )));
    }

    let profile = model_invocation_profile_entity::Entity::find_by_id(run.invocation_profile_id)
        .one(database)
        .await?
        .ok_or_else(|| DbErr::Custom("model invocation profile does not exist".into()))?;
    let workspace = resolve_output_location(&profile, &run)?;
    let existing = artifacts::list_artifacts_for_run(database, run_id).await?;
    register_known_run_artifacts(database, run_id).await?;

    println!("Registered artifacts for run {run_id}");
    println!("\nOutput workspace:\n  {}", workspace.display());
    println!("\nArtifacts:");
    // The service's own list: the call above registers what exists, so the two questions are one.
    for (artifact_type, path) in run_artifacts::known_directories(&workspace) {
        let storage_uri = path.display().to_string();
        let state = if !path.is_dir() {
            "skipped -- no such directory"
        } else if existing
            .iter()
            .any(|artifact| artifact.storage_uri == storage_uri)
        {
            "already present"
        } else {
            "registered"
        };
        println!("  [{state}] {artifact_type} -> {storage_uri}");
    }
    Ok(())
}

/// The execution alone; `run_run` owns the queueing and registration around it.
async fn report_execution(
    database: &sea_orm::DatabaseConnection,
    run_id: i32,
) -> Result<(), DbErr> {
    println!("Executing run {run_id}");
    let outcome = execute_run(database, run_id, &LocalCommandRunner).await?;

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

    // Only exit_code: it streamed, so stdout/stderr are empty by construction.
    if let Some(output) = outcome.output {
        println!("\nCommand exit_code: {}", output.exit_code);
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

/// Filesystem-only, so the dashboard can draw its dropdown without a connect and migrate.
fn list_examples(json: bool) -> Result<(), DbErr> {
    let found = examples::scan_default();
    if json {
        println!(
            "{}",
            serde_json::Value::Array(
                found
                    .iter()
                    .map(|example| json!({
                        "id": example.id,
                        "residues": example.residues,
                        "description": example.description,
                        "sequence": example.sequence,
                    }))
                    .collect()
            )
        );
        return Ok(());
    }
    if found.is_empty() {
        println!(
            "No examples under {}. Re-run `vizfold install openfold`.",
            examples::monomer_dir().display()
        );
        return Ok(());
    }
    print_table(
        &["ID", "RESIDUES", "DESCRIPTION"],
        found.iter().map(|example| {
            vec![
                example.id.clone(),
                example.residues.to_string(),
                example.description.clone(),
            ]
        }),
    );
    Ok(())
}

/// Fold a queued run by id, or an example/FASTA queued first -- one verb. Re-registers artifacts (idempotent).
async fn run_run(database: &sea_orm::DatabaseConnection, args: RunArgs) -> Result<(), DbErr> {
    let run_id = match args.target.parse::<i32>() {
        Ok(run_id) => run_id,
        Err(_) => {
            // A bundled example id, else a path to a FASTA of the user's own.
            let (example, fasta) = match examples::find(&args.target) {
                Some(example) => (example, None),
                None => {
                    let path = Path::new(&args.target);
                    let example = (path.extension().is_some() || path.exists())
                        .then(|| examples::from_path(path))
                        .flatten()
                        .ok_or_else(|| unknown_target(&args.target))?;
                    (example, Some(args.target.clone()))
                }
            };
            let backend = args.backend.map_or_else(default_backend, Ok)?;
            let run = match backend {
                Backend::Openfold => {
                    submit_openfold_run(
                        database,
                        OpenfoldQueueArgs {
                            // A user's own FASTA has no alignments, and must not borrow the examples'.
                            use_precomputed_alignments: fasta.is_none(),
                            fasta: fasta.clone(),
                            ..OpenfoldQueueArgs::for_example(&example, args.attn)
                        },
                    )
                    .await?
                }
                Backend::Esmfold => {
                    let fasta = fasta.unwrap_or_else(|| default_fasta_dir(&example.id));
                    submit_esmfold_run(database, EsmfoldQueueArgs::for_fasta(fasta)).await?
                }
            };
            if !args.json {
                println!(
                    "Queued {} run {} ({}, {} residues)\n",
                    backend.label(),
                    run.id,
                    example.id,
                    example.residues
                );
            }
            run.id
        }
    };

    if args.json {
        execute_run(database, run_id, &LocalCommandRunner).await?;
    } else {
        report_execution(database, run_id).await?;
        println!();
    }
    register_known_run_artifacts(database, run_id).await?;

    let run = runs::get_run_with_artifacts(database, run_id)
        .await?
        .map(|result| result.run)
        .ok_or_else(|| DbErr::Custom(format!("run {run_id} does not exist")))?;
    let elapsed = run
        .started_at
        .zip(run.completed_at)
        .map(|(started, completed)| (completed - started).num_seconds());

    if args.json {
        println!(
            "{}",
            json!({ "run_id": run.id, "status": run.status, "elapsed_s": elapsed })
        );
    } else if run.status == "completed" {
        let took = elapsed.map_or(String::new(), |seconds| format!(" in {seconds}s"));
        println!(
            "Run {} completed{took}. View it with: vizfold serve",
            run.id
        );
    }

    // A failed fold must exit non-zero: a `set -e` script or SLURM step has nothing else to test.
    if run.status == "completed" {
        return Ok(());
    }
    Err(DbErr::Custom(format!(
        "run {} finished with status: {}{}",
        run.id,
        run.status,
        run.error_message
            .map(|message| format!("\n{message}"))
            .unwrap_or_default()
    )))
}

fn unknown_target(target: &str) -> DbErr {
    let available: Vec<String> = examples::scan_default()
        .into_iter()
        .map(|example| example.id)
        .collect();
    DbErr::Custom(if available.is_empty() {
        format!(
            "'{target}' is not a run id, and no bundled examples were found under {}; re-run `vizfold install openfold`",
            examples::monomer_dir().display()
        )
    } else {
        format!(
            "'{target}' is not a run id or a bundled example; available examples: {}",
            available.join(", ")
        )
    })
}

async fn queue_openfold_run(
    database: &sea_orm::DatabaseConnection,
    args: OpenfoldQueueArgs,
) -> Result<(), DbErr> {
    let run = submit_openfold_run(database, args).await?;
    report_queued("OpenFold", &run);
    Ok(())
}

/// The seeded records a local run is built from. Both queue paths differ only in the backend.
struct LocalCatalog {
    backend_id: i32,
    target_id: i32,
    profile_id: i32,
    available_resources_json: String,
    provenance: String,
    working_dir: String,
}

async fn local_catalog(
    database: &sea_orm::DatabaseConnection,
    backend: Backend,
) -> Result<LocalCatalog, DbErr> {
    // Submitting needs the catalog, so seed here rather than per caller. Guarded, so repeating is free.
    seed_defaults(database).await?;
    let model = model_backend_entity::Entity::find()
        .filter(model_backend_entity::Column::Slug.eq(backend.slug()))
        .one(database)
        .await?
        .ok_or_else(seed_required_error)?;
    let target = execution_target_entity::Entity::find()
        .filter(execution_target_entity::Column::Slug.eq(format!("local-{}", backend.slug())))
        .one(database)
        .await?
        .ok_or_else(seed_required_error)?;
    let profile = model_invocation_profiles::list_model_invocation_profiles(database)
        .await?
        .into_iter()
        .find(|profile| {
            profile.model_backend_id == model.id
                && profile.execution_target_id == target.id
                && profile.invocation_kind == "local_subprocess"
        })
        .ok_or_else(seed_required_error)?;
    Ok(LocalCatalog {
        backend_id: model.id,
        target_id: target.id,
        profile_id: profile.id,
        available_resources_json: target.available_resources_json,
        provenance: runs::provenance_snapshot(
            &model.slug,
            model.version.as_deref(),
            &target.slug,
            &profile.invocation_kind,
            &profile.config_json,
            &config::openfold_home(),
            &config::prefix(),
            &backend.env_prefix(),
        ),
        working_dir: local_working_dir(&profile)?,
    })
}

/// What every queue path prints once the run exists.
fn report_queued(label: &str, run: &crate::core::entities::runs::Model) {
    println!("Queued {label} run {}", run.id);
    println!("status: {}", run.status);
    println!("input_id: {}", run.input_id);
    println!("\nNext:");
    println!("  vizfold run {}", run.id);
}

async fn submit_openfold_run(
    database: &sea_orm::DatabaseConnection,
    args: OpenfoldQueueArgs,
) -> Result<crate::core::entities::runs::Model, DbErr> {
    let catalog = local_catalog(database, Backend::Openfold).await?;
    let working_dir = &catalog.working_dir;
    let fasta_input = match (&args.fasta, &args.input_id) {
        (Some(fasta), _) => fasta.clone(),
        (None, Some(id)) => default_fasta_dir(id),
        (None, None) => return Err(DbErr::Custom("pass --fasta or --input-id".to_owned())),
    };
    let data_dir_input = args
        .data_dir
        .clone()
        .unwrap_or_else(|| config::data_dir().to_string_lossy().into_owned());
    let fasta_dir = canonicalize_local_path("--fasta", &fasta_input, working_dir)?;
    // Read from the FASTA, so they cannot contradict what is folded -- preflight allows nothing else.
    let example = read_fasta(&fasta_dir)?;
    let data_dir = canonicalize_local_path("--data-dir", &data_dir_input, working_dir)?;
    let alignment_dir = if args.use_precomputed_alignments {
        let input = args
            .alignment_dir
            .clone()
            .unwrap_or_else(default_alignment_dir);
        Some(canonicalize_local_path(
            "--alignment-dir",
            &input,
            working_dir,
        )?)
    } else {
        args.alignment_dir
            .as_deref()
            .map(|path| canonicalize_local_path("--alignment-dir", path, working_dir))
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
            json!(clamp_cpus(
                args.cpus.unwrap_or_else(default_cpus),
                &catalog.available_resources_json
            )),
        ),
    ]);
    if let Some(alignment_dir) = alignment_dir {
        execution_parameters.insert("alignment_dir".into(), json!(alignment_dir));
    }

    let run = runs::submit_run(
        database,
        runs::SubmitRunInput {
            model_backend_id: catalog.backend_id,
            execution_target_id: catalog.target_id,
            invocation_profile_id: catalog.profile_id,
            status: "submitted".into(),
            input_id: args.input_id.unwrap_or(example.id),
            input_sequence: example.sequence,
            // demo_attn on the wire: the Python argument name and the seed schema key.
            model_parameters_json: json!({
                "save_outputs": args.save_outputs,
                "demo_attn": args.attn,
                "num_recycles_save": args.num_recycles_save,
            })
            .to_string(),
            execution_parameters_json: serde_json::Value::Object(execution_parameters).to_string(),
            provenance_json: Some(catalog.provenance),
        },
    )
    .await?;

    Ok(run)
}

async fn queue_esmfold_run(
    database: &sea_orm::DatabaseConnection,
    args: EsmfoldQueueArgs,
) -> Result<(), DbErr> {
    let run = submit_esmfold_run(database, args).await?;
    report_queued("ESMFold", &run);
    Ok(())
}

async fn submit_esmfold_run(
    database: &sea_orm::DatabaseConnection,
    args: EsmfoldQueueArgs,
) -> Result<crate::core::entities::runs::Model, DbErr> {
    let catalog = local_catalog(database, Backend::Esmfold).await?;
    let working_dir = &catalog.working_dir;
    let fasta = canonicalize_local_path("--fasta", &args.fasta, working_dir)?;
    let example = read_fasta(&fasta)?;
    let model_device = args
        .model_device
        .clone()
        .unwrap_or_else(default_model_device);

    let run = runs::submit_run(
        database,
        runs::SubmitRunInput {
            model_backend_id: catalog.backend_id,
            execution_target_id: catalog.target_id,
            invocation_profile_id: catalog.profile_id,
            status: "submitted".into(),
            input_id: args.input_id.unwrap_or(example.id),
            input_sequence: example.sequence,
            model_parameters_json: json!({
                "model": args.model,
                "trace_mode": args.trace_mode,
                "layers": args.layers,
                "dtype": args.dtype,
                "save_fp16": args.save_fp16,
                "structure_traces": args.structure_traces,
            })
            .to_string(),
            execution_parameters_json: json!({
                "fasta": fasta,
                "model_device": model_device,
            })
            .to_string(),
            provenance_json: Some(catalog.provenance),
        },
    )
    .await?;

    Ok(run)
}

impl EsmfoldQueueArgs {
    /// The one-command path's defaults, matching what clap applies to `queue esmfold`.
    fn for_fasta(fasta: String) -> Self {
        Self {
            input_id: None,
            fasta,
            model_device: None,
            model: "facebook/esmfold_v1".to_owned(),
            trace_mode: "attention+activations".to_owned(),
            layers: "all".to_owned(),
            dtype: "float32".to_owned(),
            save_fp16: false,
            structure_traces: false,
        }
    }
}

/// The backend a bare `vizfold run` uses: the only one installed, else OpenFold. `--backend` always wins.
fn default_backend() -> Result<Backend, DbErr> {
    match (
        Backend::Openfold.is_installed(),
        Backend::Esmfold.is_installed(),
    ) {
        (false, true) => Ok(Backend::Esmfold),
        (true, _) => Ok(Backend::Openfold),
        (false, false) => Err(DbErr::Custom(
            "no backend is installed; run `vizfold install openfold` or `vizfold install esmfold`"
                .to_owned(),
        )),
    }
}

/// `<OPENFOLD_HOME>/examples/monomer/fasta_dir_<id-stem>` -- the id up to its last underscore.
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

fn local_working_dir(
    profile: &crate::core::entities::model_invocation_profiles::Model,
) -> Result<String, DbErr> {
    let config: serde_json::Value =
        serde_json::from_str(&profile.config_json).map_err(|error| {
            DbErr::Custom(format!(
                "local invocation profile config_json must be valid JSON: {error}"
            ))
        })?;
    config
        .get("working_dir")
        .and_then(serde_json::Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            DbErr::Custom(
                "local invocation profile config_json requires a non-empty working_dir".into(),
            )
        })
}

/// cwd first, then the target's working dir: a relative path means what the shell means by it.
fn canonicalize_local_path(field: &str, path: &str, working_dir: &str) -> Result<String, DbErr> {
    let original_path = Path::new(path);
    if original_path.is_absolute() {
        return canonicalize_at(field, path, original_path);
    }
    std::env::current_dir()
        .ok()
        .map(|cwd| cwd.join(original_path))
        .filter(|candidate| candidate.exists())
        .map_or_else(
            || canonicalize_at(field, path, &PathBuf::from(working_dir).join(original_path)),
            |candidate| canonicalize_at(field, path, &candidate),
        )
}

fn canonicalize_at(field: &str, original: &str, attempted: &Path) -> Result<String, DbErr> {
    std::fs::canonicalize(attempted)
        .map(|path| path.display().to_string())
        .map_err(|error| {
            DbErr::Custom(format!(
                "{field} original path '{original}' could not be resolved at '{}': {error}",
                attempted.display()
            ))
        })
}

/// The one sequence at a resolved FASTA path, as the source of a run's id and sequence.
fn read_fasta(path: &str) -> Result<examples::Example, DbErr> {
    examples::from_path(Path::new(path)).ok_or_else(|| {
        DbErr::Custom(format!(
            "no FASTA record at '{path}': expected a .fasta/.fa file, or a directory holding one, \
             with a '>' header and a sequence"
        ))
    })
}

/// Seeding runs immediately before every lookup, so a miss here means the database is not one
/// vizfold wrote -- naming the file is the only thing that helps.
fn seed_required_error() -> DbErr {
    DbErr::Custom(format!(
        "the run's backend, local execution target, or matching profile is missing from {} \
         even after seeding; point VIZFOLD_DB at a vizfold database, or remove that file",
        config::database_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(config::database_url)
    ))
}

async fn list_models(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let models = model_backends::list_model_backends(database).await?;
    print_table(
        &["ID", "SLUG", "LABEL", "VERSION"],
        models.iter().map(|model| {
            vec![
                model.id.to_string(),
                model.slug.clone(),
                model.label.clone(),
                model.version.clone().unwrap_or_else(|| "-".into()),
            ]
        }),
    );
    Ok(())
}

async fn list_targets(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let targets = execution_targets::list_execution_targets(database).await?;
    print_table(
        &["ID", "SLUG", "TYPE", "DESCRIPTION"],
        targets.iter().map(|target| {
            vec![
                target.id.to_string(),
                target.slug.clone(),
                target.target_type.clone(),
                target.description.clone().unwrap_or_else(|| "-".into()),
            ]
        }),
    );
    Ok(())
}

async fn list_profiles(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let profiles = model_invocation_profiles::list_model_invocation_profiles(database).await?;
    print_table(
        &["ID", "MODEL ID", "TARGET ID", "INVOCATION KIND"],
        profiles.iter().map(|profile| {
            vec![
                profile.id.to_string(),
                profile.model_backend_id.to_string(),
                profile.execution_target_id.to_string(),
                profile.invocation_kind.clone(),
            ]
        }),
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
            }),
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
        result.artifacts.iter().map(|artifact| {
            vec![
                artifact.id.to_string(),
                artifact.artifact_type_id.to_string(),
                artifact.format.clone(),
                artifact.storage_uri.clone(),
            ]
        }),
    );
    Ok(())
}

fn format_time(value: Option<chrono::DateTime<chrono::Utc>>) -> String {
    value
        .map(|time| time.to_rfc3339())
        .unwrap_or_else(|| "-".into())
}

fn print_table(headers: &[&str], rows: impl IntoIterator<Item = Vec<String>>) {
    let rows: Vec<Vec<String>> = rows.into_iter().collect();
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

    /// The defaults, and that `--save-fp16` is a bare presence flag while `--trace-mode` takes a value.
    #[test]
    fn parses_queue_esmfold_arguments() {
        let cli = Cli::try_parse_from([
            "vizfold",
            "queue",
            "esmfold",
            "--input-id",
            "6KWC_1",
            "--fasta",
            "6KWC.fasta",
            "--trace-mode",
            "attention",
            "--save-fp16",
        ])
        .expect("queue esmfold command should parse");

        assert!(matches!(
            cli.command,
            Command::Queue(QueueArgs {
                model: QueueModel::Esmfold(EsmfoldQueueArgs {
                    trace_mode,
                    save_fp16: true,
                    structure_traces: false,
                    ref model,
                    ..
                })
            }) if trace_mode == "attention" && model == "facebook/esmfold_v1"
        ));
    }

    /// The default-true bools take `ArgAction::Set`, so `--flag=false` is a legal spelling.
    #[test]
    fn parses_queue_openfold_optional_flags() {
        let cli = Cli::try_parse_from([
            "vizfold",
            "queue",
            "openfold",
            "--input-id",
            "6KWC_1",
            "--attn=true",
            "--use-precomputed-alignments=false",
        ])
        .expect("queue command should parse");

        assert!(matches!(
            cli.command,
            Command::Queue(QueueArgs {
                model: QueueModel::Openfold(OpenfoldQueueArgs {
                    attn: true,
                    use_precomputed_alignments: false,
                    ..
                })
            })
        ));
    }

    #[test]
    fn parses_install_backend() {
        for (arg, want) in [
            ("openfold", Backend::Openfold),
            ("esmfold", Backend::Esmfold),
        ] {
            let cli = Cli::try_parse_from(["vizfold", "install", arg])
                .expect("install <backend> should parse");
            assert!(
                matches!(cli.command, Command::Install(InstallArgs { backend }) if backend == want)
            );
        }
        // The backend is required and constrained to the known set.
        assert!(Cli::try_parse_from(["vizfold", "install"]).is_err());
        assert!(Cli::try_parse_from(["vizfold", "install", "rosetta"]).is_err());
    }

    /// Each backend reads its own key, so a stray `ESMFOLD_ENV_PREFIX` cannot move OpenFold's env.
    #[test]
    fn backend_is_installed_tracks_its_own_env_prefix_key() {
        let base = std::env::temp_dir().join(format!("vizfold-backend-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        // SAFETY: single-threaded test; pinned so env_prefix() does not read the real config.
        unsafe { std::env::set_var("ESMFOLD_ENV_PREFIX", &base) };
        assert_eq!(Backend::Esmfold.env_prefix(), base);
        assert_ne!(Backend::Openfold.env_prefix(), base);
        assert!(
            Backend::Esmfold.is_installed(),
            "an existing env dir reads as installed"
        );
        std::fs::remove_dir_all(&base).unwrap();
        assert!(
            !Backend::Esmfold.is_installed(),
            "a missing env dir reads as not installed"
        );
        unsafe { std::env::remove_var("ESMFOLD_ENV_PREFIX") };
    }

    #[test]
    fn parses_uninstall() {
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "uninstall"])
                .expect("uninstall command should parse")
                .command,
            Command::Uninstall(UninstallArgs {
                backend: None,
                yes: false
            })
        ));
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "uninstall", "--yes"])
                .expect("uninstall --yes should parse")
                .command,
            Command::Uninstall(UninstallArgs {
                backend: None,
                yes: true
            })
        ));
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "uninstall", "esmfold"])
                .expect("uninstall <backend> should parse")
                .command,
            Command::Uninstall(UninstallArgs {
                backend: Some(Backend::Esmfold),
                yes: false
            })
        ));
    }

    #[test]
    fn openfold_install_paths_cover_generated_trees_but_not_run_outputs() {
        let base = std::env::temp_dir().join(format!("vizfold-uninstall-{}", std::process::id()));
        let (prefix, home) = (base.join("prefix"), base.join("checkout"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(prefix.join("outputs")).unwrap();
        // The editable install's extension, named for the Python ABI and arch that built it.
        let backend = home.join("backends/openfold");
        std::fs::create_dir_all(&backend).unwrap();
        let extension = backend.join("attn_core_inplace_cuda.cpython-311-x86_64-linux-gnu.so");
        std::fs::write(&extension, "").unwrap();

        let paths = Backend::Openfold.install_paths(&prefix, &home);

        for expected in [
            // One state dir covers cutlass, tmp, the caches, the sentinel, and every nvrtc pin.
            prefix.join("openfold"),
            backend.join("openfold.egg-info"),
            backend.join("openfold/resources/stereo_chemical_props.txt"),
            extension,
        ] {
            assert!(paths.contains(&expected), "missing {}", expected.display());
        }
        assert!(!paths.contains(&prefix.join("outputs")), "run outputs kept");
        // The package cache serves the workbench env too; one backend's uninstall must not take it.
        assert!(
            !paths.contains(&prefix.join("mamba")),
            "the cache is shared"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// The reinstall invariant: uninstall takes exactly what install puts back, and nothing else.
    #[test]
    fn one_backend_leaves_the_other_and_everything_shared_alone() {
        let base = std::env::temp_dir().join(format!("vizfold-scoped-{}", std::process::id()));
        let (prefix, home) = (base.join("prefix"), base.join("checkout"));

        let openfold = Backend::Openfold.install_paths(&prefix, &home);
        let esmfold = Backend::Esmfold.install_paths(&prefix, &home);
        let shared = super::shared_paths(&prefix, &home);

        assert!(
            openfold.iter().all(|path| !esmfold.contains(path)),
            "the two backends share a path"
        );
        for owned in openfold.iter().chain(&esmfold) {
            assert!(
                !shared.contains(owned),
                "{} is a backend's, not shared",
                owned.display()
            );
        }
        assert!(shared.contains(&config::config_file()), "config is shared");
        // Removing the base itself once took a whole home directory with it.
        assert!(
            !shared.contains(&config::env_base()),
            "the env base is the user's; only its vizfold- entries are ours"
        );
        assert!(
            shared.contains(&config::env_dir("workbench")),
            "no backend owns the workbench env, so only a full uninstall takes it"
        );
        assert!(
            shared.contains(&prefix.join("workbench")),
            "a staged workbench is shared"
        );
        // With no prefix settled the two are one path, and that one is the source tree.
        assert!(
            !super::shared_paths(&home, &home).contains(&home.join("workbench")),
            "the checkout's own workbench must survive"
        );
    }

    /// The checks name keys as strings; one outside the schema reports on a value nothing writes.
    /// `uninstall` is `rm -rf`: no relative path may reach it, and no path an outer target covers.
    #[test]
    fn the_removal_plan_keeps_only_absolute_uncovered_paths_that_exist() {
        let base = std::env::temp_dir().join(format!("vizfold-plan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("outer/inner")).expect("fixture");

        let plan = super::removal_plan(vec![
            // Exists relative to the cwd tests run in: without the guard, uninstall deletes it.
            PathBuf::from("Cargo.toml"),
            base.join("does-not-exist"), // nothing to remove
            base.join("outer/inner"),    // covered by its parent below
            base.join("outer"),
            base.join("outer"), // duplicate
        ]);

        assert_eq!(plan, vec![base.join("outer")]);
        std::fs::remove_dir_all(&base).ok();
    }

    /// The gate table by the names it produces. `status` and the updaters stay ungated -- they are
    /// what repairs a broken install.
    #[test]
    fn every_command_gates_on_the_prereqs_it_actually_needs() {
        let names = |argv: &[&str]| {
            let cli = Cli::try_parse_from(argv).expect("argv should parse");
            super::prereqs(&cli.command)
                .into_iter()
                .map(|component| component.name)
                .collect::<Vec<_>>()
        };

        assert_eq!(names(&["vizfold", "status"]), Vec::<&str>::new());
        assert_eq!(names(&["vizfold", "update"]), Vec::<&str>::new());
        assert_eq!(names(&["vizfold", "install", "openfold"]), ["core deps"]);
        assert_eq!(names(&["vizfold", "list", "examples"]), ["repo"]);
        assert_eq!(
            names(&["vizfold", "serve"]),
            ["core deps", "repo", "config"]
        );
        assert_eq!(
            names(&["vizfold", "run", "1UBQ_1"]),
            ["core deps", "config", "openfold"]
        );
        assert_eq!(
            names(&["vizfold", "queue", "esmfold", "--fasta", "x.fasta"]),
            ["config", "esmfold"]
        );

        // The one binary the bootstrap installs, named in the detail whether or not it is found.
        let core = super::core_deps_health();
        assert!(
            core.detail.ends_with("micromamba"),
            "core deps must look for micromamba, got {}",
            core.detail
        );
    }

    /// The rule `refuses` encodes: Unverified is not a failure.
    #[test]
    fn only_absent_and_broken_refuse() {
        assert!(super::refuses(State::Absent));
        assert!(super::refuses(State::Broken));
        assert!(!super::refuses(State::Unverified));
        assert!(!super::refuses(State::Ok));
    }

    #[test]
    fn checked_keys_are_all_in_the_schema() {
        let checked = super::CHECKED_PATHS
            .iter()
            .chain(super::OPENFOLD_PATHS)
            .map(|(key, _)| *key)
            .chain(super::CHECKED_PARTITIONS.iter().copied())
            .chain(super::CHECKED_ACCOUNTS.iter().copied())
            .chain(["OPENFOLD_EXAMPLE"]);
        for key in checked {
            assert!(
                config::CONFIG_KEYS.contains(&key),
                "{key} is checked but is not a config key"
            );
        }
    }

    /// Real `sinfo -h -o %100P`: a name must survive the `*` marker and the padding, or it reads as missing.
    #[test]
    fn scheduler_names_survive_the_padding_and_the_default_marker() {
        let stdout = format!(
            "{:<100}\n{:<100}\n{:<100}\n\n",
            "cpu*", "cpu*", "gpuA100x4-interactive"
        );
        assert_eq!(
            super::scheduler_names(&stdout),
            ["cpu", "cpu", "gpuA100x4-interactive"]
        );
        assert_eq!(
            super::scheduler_names("bbol-delta-gpu\n"),
            ["bbol-delta-gpu"]
        );
        assert!(super::scheduler_names("\n \n").is_empty());
    }

    #[test]
    fn an_unanswered_scheduler_question_is_not_a_problem() {
        let known = ["cpu".to_owned(), "gpuA100x4-interactive".to_owned()];
        let problem = |known: Option<&[String]>, value| {
            super::unknown_to_scheduler("OPENFOLD_PARTITION", value, known, "partition")
        };
        assert_eq!(problem(None, "cpu"), None, "unasked is not answered");
        assert_eq!(problem(Some(&known), "cpu"), None, "a name it has");
        assert!(
            problem(Some(&known), "gpu-nope")
                .is_some_and(|p| p.contains("gpu-nope") && p.contains("OPENFOLD_PARTITION")),
            "a name it does not have, said with the key that set it"
        );
    }

    /// Only `Broken` counts: an absent backend or unreachable scheduler is not a problem.
    #[test]
    fn the_summary_counts_only_what_is_broken() {
        let component = |name, state| super::Component {
            name,
            state,
            ..Default::default()
        };
        assert_eq!(
            super::summary(&[
                component("binary", super::State::Ok),
                component("esmfold", super::State::Absent),
                component("scheduler", super::State::Unverified),
            ]),
            "Everything checks out."
        );
        assert_eq!(
            super::summary(&[
                component("binary", super::State::Ok),
                component("repo", super::State::Broken),
                component("config", super::State::Broken),
            ]),
            "2 of 3 components need attention: repo, config."
        );
    }

    /// Broken exactly when it carries a problem, so no builder keeps two fields in step by hand.
    #[test]
    fn health_promotes_any_component_with_a_problem() {
        for component in super::health() {
            assert_eq!(
                component.state == super::State::Broken,
                !component.problems.is_empty(),
                "{} reports {:?} with {} problem(s)",
                component.name,
                component.state,
                component.problems.len()
            );
        }
    }

    /// A failed micromamba fetch leaves a truncated, non-executable file; reading that as installed
    /// sends the user into `Permission denied` from inside an installer instead of at the gate.
    #[test]
    fn a_present_but_non_executable_file_is_not_the_dependency() {
        let dir = std::env::temp_dir().join(format!("vizfold-exec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let program = dir.join("micromamba");
        std::fs::write(&program, "").unwrap();

        assert!(!super::executable(&program), "0644 is not runnable");
        std::fs::set_permissions(
            &program,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();
        assert!(super::executable(&program));
        assert!(!super::executable(&dir), "a directory is not a program");
        std::fs::remove_dir_all(&dir).ok();
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
        // public/runs is a symlink; fs::copy would follow it into a directory and hit EISDIR.
        let base =
            std::env::temp_dir().join(format!("vizfold-copytree-link-{}", std::process::id()));
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
    fn node_version_gate_is_major_then_minor() {
        assert!(node_is_new_enough("22.13.0"));
        assert!(node_is_new_enough("v26.5.0"));
        assert!(node_is_new_enough("23.0.1"));
        assert!(!node_is_new_enough("22.12.9"), "minor below the floor");
        assert!(!node_is_new_enough("20.19.0"), "major below the floor");
        // 22.9 must not beat 22.13 -- the reason this compares numbers, not strings.
        assert!(!node_is_new_enough("22.9.0"));
        assert!(!node_is_new_enough(""), "unparseable reads as too old");
        assert!(!node_is_new_enough("garbage"));
    }

    /// The target is a free string: `run_run` tells a run id from an example id by whether it
    /// parses as an integer.
    #[test]
    fn parses_run() {
        let cli = Cli::try_parse_from(["vizfold", "run", "1"]).expect("run command should parse");

        assert!(matches!(
            cli.command,
            Command::Run(RunArgs {
                ref target,
                backend: None,
                attn: true,
                json: false,
            }) if target == "1"
        ));

        let cli = Cli::try_parse_from(["vizfold", "run", "6KWC_1", "--attn=false"])
            .expect("run command should parse");

        assert!(matches!(
            cli.command,
            Command::Run(RunArgs {
                ref target,
                attn: false,
                ..
            }) if target == "6KWC_1"
        ));
    }

    /// No flags at all: `defaults_match_for_example` passes `--attn`, so it cannot see these.
    #[test]
    fn queue_openfold_defaults_to_attention_and_precomputed_alignments() {
        let cli = Cli::try_parse_from(["vizfold", "queue", "openfold"])
            .expect("queue openfold should parse with no flags");
        let Command::Queue(QueueArgs {
            model: QueueModel::Openfold(parsed),
        }) = cli.command
        else {
            panic!("expected an openfold queue");
        };

        assert!(parsed.attn, "attention maps are on unless asked otherwise");
        assert!(
            parsed.use_precomputed_alignments,
            "a full MSA search is opt-in"
        );
        assert_eq!(parsed.residue_idx, 1);
    }

    /// `for_example` hardcodes clap's `queue openfold` defaults; this fails if one drifts.
    #[test]
    fn queue_openfold_defaults_match_for_example() {
        let cli = Cli::try_parse_from([
            "vizfold",
            "queue",
            "openfold",
            "--input-id",
            "1UBQ_1",
            "--attn=true",
        ])
        .expect("queue command should parse");
        let Command::Queue(QueueArgs {
            model: QueueModel::Openfold(parsed),
        }) = cli.command
        else {
            panic!("expected an openfold queue");
        };

        let example = examples::Example {
            id: "1UBQ_1".into(),
            residues: 8,
            description: "UBIQUITIN".into(),
            sequence: "MQIFVKTL".into(),
        };
        assert_eq!(OpenfoldQueueArgs::for_example(&example, true), parsed);
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

        // A real FASTA, so this pins the id and sequence derivation too.
        let fasta_dir =
            std::fs::canonicalize(crate::core::examples::monomer_dir().join("fasta_dir_6KWC"))
                .expect("the bundled 6KWC example should exist")
                .display()
                .to_string();

        queue_openfold_run(
            &database,
            OpenfoldQueueArgs {
                input_id: None,
                fasta: Some(fasta_dir.clone()),
                data_dir: Some(local_path.clone()),
                alignment_dir: Some(local_path.clone()),
                model_device: Some("cpu".into()),
                // Over the seeded target's cpus.maximum of 14, so the run must record the clamped value.
                cpus: Some(18),
                residue_idx: 1,
                attn: true,
                save_outputs: true,
                num_recycles_save: 1,
                use_precomputed_alignments: true,
            },
        )
        .await?;

        let runs = runs::list_runs(&database).await?;
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].status, "submitted");
        assert_eq!(
            runs[0].input_id, "6KWC_1",
            "the id comes from the FASTA header"
        );
        assert_eq!(
            runs[0].input_sequence.len(),
            191,
            "the sequence comes from the FASTA"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&runs[0].model_parameters_json)
                .expect("model parameters should be valid JSON"),
            json!({"save_outputs": true, "demo_attn": true, "num_recycles_save": 1})
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&runs[0].execution_parameters_json)
                .expect("execution parameters should be valid JSON"),
            json!({"fasta_dir": fasta_dir, "data_dir": local_path, "alignment_dir": local_path, "residue_idx": 1, "use_precomputed_alignments": true, "model_device": "cpu", "cpus": 14})
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
                input_id: Some("6KWC_1".into()),
                fasta: Some(missing_path.into()),
                data_dir: Some(".".into()),
                alignment_dir: None,
                model_device: Some("cpu".into()),
                cpus: Some(1),
                residue_idx: 1,
                attn: false,
                save_outputs: true,
                num_recycles_save: 1,
                use_precomputed_alignments: false,
            },
        )
        .await
        .expect_err("missing local path should fail");

        assert!(error.to_string().contains(
            "--fasta original path 'definitely-missing-vizfold-local-path' could not be resolved"
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
        // A GPU partition with no allocation held: cuda:0 is right even though no GPU is visible here.
        assert_eq!(
            super::model_device_for(config::SlurmContext::None, Some("gpuA100x4"), None),
            "cuda:0"
        );
    }

    #[test]
    fn model_device_inside_an_allocation_trusts_the_local_probe() {
        // Already on the node the fold runs on, so the local probe decides, not the partition config.
        assert_eq!(
            super::model_device_for(config::SlurmContext::InAllocation, Some("gpuA100x4"), None),
            "cpu"
        );
    }

    /// No declared maximum, or unparseable resources, must not clamp the request to something arbitrary.
    #[test]
    fn cpus_clamp_only_where_the_target_declares_a_maximum() {
        let with_max = json!({"properties": {"cpus": {"maximum": 14}}}).to_string();
        assert_eq!(super::clamp_cpus(18, &with_max), 14);
        assert_eq!(super::clamp_cpus(8, &with_max), 8);
        assert_eq!(
            super::clamp_cpus(18, &json!({"properties": {}}).to_string()),
            18
        );
        assert_eq!(super::clamp_cpus(18, "not-json"), 18);
    }
}
