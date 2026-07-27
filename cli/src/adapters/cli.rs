use clap::{ArgAction, Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
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
// `run` carries every fold flag; one enum is parsed, once, per process.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Install the checkout everything runs from (`base`), or a model backend from it.
    Install(InstallArgs),
    /// Download a backend's data (OpenFold AlphaFold2 databases/params).
    Download(DownloadArgs),
    /// Show resolved config, which backends are installed, and whether it all checks out.
    Status,
    /// Remove one part, or everything the install generated.
    Uninstall(UninstallArgs),
    /// Move the checkout to this binary's release (`base`), or reinstall a backend from it.
    Update(UpdateArgs),
    /// Replace this binary with the latest release. Run `update base` after, for the checkout.
    SelfUpdate(SelfUpdateArgs),
    /// Start the workbench dashboard, over the given backends (default: all installed).
    Serve(ServeArgs),
    /// List executor records.
    List(ListArgs),
    /// Show one executor record.
    Show(ShowArgs),
    /// Fold targets in one execution: bundled examples, FASTAs, directories of FASTAs -- or a
    /// queued run by id.
    Run(RunArgs),
    /// Register known artifacts for a completed run.
    RegisterArtifacts { run_id: i32 },
    /// Print this shell's tab-completion script. `install.sh` wires it into your shell rc.
    Completions(CompletionsArgs),
}

#[derive(Debug, Args)]
struct CompletionsArgs {
    /// Shell to emit for. Defaults to the one `$SHELL` names.
    #[arg(value_enum)]
    shell: Option<Shell>,
}

#[derive(Debug, Args)]
struct InstallArgs {
    /// What to install: the checkout everything runs from, or a model backend from it.
    #[arg(value_enum)]
    part: Part,
}

/// What a lifecycle verb acts on: the checkout every backend installs from, or one backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Part {
    Base,
    Openfold,
    Esmfold,
}

impl Part {
    fn backend(self) -> Option<Backend> {
        match self {
            Self::Base => None,
            Self::Openfold => Some(Backend::Openfold),
            Self::Esmfold => Some(Backend::Esmfold),
        }
    }

    fn slug(self) -> &'static str {
        self.backend().map_or("base", Backend::slug)
    }

    /// Exactly what `install <part>` puts there, so `uninstall <part>` takes back the same set.
    fn install_paths(self, prefix: &Path, home: &Path) -> Vec<PathBuf> {
        self.backend().map_or_else(
            || base_paths(prefix),
            |backend| backend.install_paths(prefix, home),
        )
    }
}

/// The checkout, and only the clone vizfold made itself: never a user-supplied `OPENFOLD_HOME`.
fn base_paths(prefix: &Path) -> Vec<PathBuf> {
    let src = config::vizfold_src();
    (src == config::default_src() && !prefix.starts_with(&src))
        .then_some(src)
        .into_iter()
        .collect()
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

    /// Installer script, relative to the checkout: each backend owns one under `backends/<name>/install/`.
    fn installer(self) -> &'static str {
        match self {
            Self::Openfold => config::INSTALLER,
            Self::Esmfold => "backends/esmfold/install/install.sh",
        }
    }

    /// Downloader dir, relative to the checkout. `None` for ESMFold: it fetches weights at run time.
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
    /// Part to remove. Omit to remove every part, the config, and the run database too.
    #[arg(value_enum)]
    part: Option<Part>,
    /// Remove without the confirmation prompt.
    #[arg(long, short = 'y')]
    yes: bool,
}

#[derive(Debug, Args)]
struct UpdateArgs {
    /// What to bring current: the checkout, or a backend reinstalled from it.
    #[arg(value_enum)]
    part: Part,
    /// Tag or branch to move the checkout to. Defaults to this binary's own release tag. `base` only.
    #[arg(long, value_name = "REF")]
    r#ref: Option<String>,
    /// Reinstall without the confirmation prompt.
    #[arg(long, short = 'y')]
    yes: bool,
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
    /// Backends the dashboard folds with and lists runs for. Defaults to every one installed.
    #[arg(value_enum)]
    backends: Vec<Backend>,
    /// Port for the dashboard dev server. Defaults to 3000.
    #[arg(long)]
    port: Option<u16>,
}

impl ServeArgs {
    /// For the dashboard: the named backends in order, else all installed -- never one that cannot run.
    fn backends_env(&self) -> String {
        let served: Vec<Backend> = if self.backends.is_empty() {
            Backend::value_variants()
                .iter()
                .copied()
                .filter(|backend| backend.is_installed())
                .collect()
        } else {
            self.backends.clone()
        };
        served
            .iter()
            .map(|backend| backend.slug())
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[derive(Debug, Args)]
struct ListArgs {
    #[command(subcommand)]
    resource: ListResource,
}

#[derive(Debug, Args)]
struct RunArgs {
    /// What to fold: bundled example ids (`vizfold list examples`), FASTA files, or directories of
    /// FASTAs -- several fold in one execution. A lone run id replays that queued run instead.
    #[arg(required = true)]
    targets: Vec<String>,
    /// Backend. Defaults to the only one installed, else openfold. A queued run carries its own.
    #[arg(long, value_enum)]
    backend: Option<Backend>,
    /// Record the run and stop, without folding it. `vizfold run <id>` folds it later.
    #[arg(long)]
    no_exec: bool,
    /// Name recorded for this run. Defaults to the folded tags joined with `+`, the only value
    /// preflight takes.
    #[arg(long)]
    input_id: Option<String>,
    /// Print only the run as JSON, for tools driving the CLI.
    #[arg(long)]
    json: bool,
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
    /// Residue index offset passed through to the model (OpenFold).
    #[arg(long, default_value_t = 1)]
    residue_idx: i64,
    /// Dump per-layer, per-head attention maps, under `attention/<tag>/` (OpenFold).
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    attn: bool,
    /// Write the model's raw output tensors alongside the structure (OpenFold).
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    save_outputs: bool,
    /// How many recycling iterations to keep outputs for (OpenFold).
    #[arg(long, default_value_t = 1)]
    num_recycles_save: i64,
    /// Use the precomputed alignments in `--alignment-dir`. On by default only when every target is
    /// a bundled example; `--use-precomputed-alignments=false` forces the full MSA pipeline.
    #[arg(long, action = ArgAction::Set)]
    use_precomputed_alignments: Option<bool>,
    /// HuggingFace model id (ESMFold).
    #[arg(long, default_value = "facebook/esmfold_v1")]
    model: String,
    /// What to extract: none, attention, activations, or attention+activations (ESMFold).
    #[arg(long, default_value = "attention+activations")]
    trace_mode: String,
    /// Layers to save: `all` or a comma/colon list (ESMFold).
    #[arg(long, default_value = "all")]
    layers: String,
    /// Model dtype (ESMFold).
    #[arg(long, default_value = "float32")]
    dtype: String,
    /// Save trace tensors in fp16 to reduce size (ESMFold).
    #[arg(long)]
    save_fp16: bool,
    /// Capture IPA attention and per-recycle backbone from the structure module (ESMFold).
    #[arg(long)]
    structure_traces: bool,
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

/// The one place a prerequisite is refused, so no command grows its own check.
fn prereqs(command: &Command) -> Vec<Component> {
    // `run <id>` cannot read the run's own backend without the DB, so it gates on the default:
    // uninstalling a backend under a queued run still reaches the runner, which names what is missing.
    let backend = |explicit: Option<Backend>| {
        backend_health(
            explicit
                .or_else(|| default_backend().ok())
                .unwrap_or(Backend::Openfold),
        )
    };
    match command {
        Command::Status
        | Command::Uninstall(_)
        | Command::SelfUpdate(_)
        | Command::Completions(_) => vec![],
        // A backend installs from the checkout; base's own verbs are what repair it, so they stay off it.
        Command::Install(InstallArgs { part }) | Command::Update(UpdateArgs { part, .. }) => {
            std::iter::once(core_deps_health())
                .chain(part.backend().map(|_| repo_health()))
                .collect()
        }
        Command::List(ListArgs {
            resource: ListResource::Examples { .. },
        }) => vec![repo_health()],
        // Only the backends named: bare `serve` resolves to what is installed, so it can gate on nothing.
        Command::Serve(args) => [core_deps_health(), repo_health(), config_health()]
            .into_iter()
            .chain(args.backends.iter().copied().map(backend_health))
            .collect(),
        Command::Download(args) => {
            vec![repo_health(), config_health(), backend(Some(args.backend))]
        }
        // Recording a run touches no environment, so a submit host with no micromamba still can.
        Command::Run(args) if args.no_exec => vec![config_health(), backend(args.backend)],
        Command::Run(args) => vec![
            core_deps_health(),
            repo_health(),
            config_health(),
            backend(args.backend),
        ],
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

    match cli.command {
        Command::Install(args) => return run_install(args.part),
        Command::Download(args) => return run_download(args.backend, args.dataset),
        Command::Status => return run_status(),
        Command::Uninstall(args) => return run_uninstall(args),
        Command::Update(args) => return run_update(args),
        Command::SelfUpdate(args) => return run_self_update(args),
        Command::Serve(args) => return run_serve(args),
        Command::Completions(args) => return run_completions(args.shell),
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
        Command::Run(args) => run_run(&database, args).await?,
        Command::RegisterArtifacts { run_id } => register_artifacts(&database, run_id).await?,
        Command::Install(_)
        | Command::Download(_)
        | Command::Status
        | Command::Uninstall(_)
        | Command::Update(_)
        | Command::SelfUpdate(_)
        | Command::Serve(_)
        | Command::Completions(_) => {
            unreachable!("handled before DB connect")
        }
    }

    Ok(())
}

/// Make a part exist. Idempotent: both backend installers short-circuit on presence, as base does.
fn run_install(part: Part) -> Result<(), DbErr> {
    match part.backend() {
        Some(backend) => install_backend(backend),
        None => install_base(),
    }
}

/// Fetch the checkout the binary installs everything from -- it ships only itself. Never moves one.
fn install_base() -> Result<(), DbErr> {
    let src = config::vizfold_src();
    if src.join(config::INSTALLER).is_file() {
        println!("{} already is a vizfold checkout.", src.display());
        return Ok(());
    }
    clone_checkout(&src)
}

fn install_backend(backend: Backend) -> Result<(), DbErr> {
    let src = config::vizfold_src();
    let installer = src.join(backend.installer());
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

fn run_download(backend: Backend, dataset: String) -> Result<(), DbErr> {
    let Some(dir) = backend.downloader_dir() else {
        println!(
            "{} fetches its weights from HuggingFace at run time; nothing to download.",
            backend.slug()
        );
        return Ok(());
    };
    let src = config::vizfold_src();
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

/// One independently breakable part.
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

/// The one binary `install.sh` bootstraps; every environment is created and run through it.
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
    repo_health_at(&config::vizfold_src())
}

fn repo_health_at(src: &Path) -> Component {
    if !src.join(config::INSTALLER).is_file() {
        return Component {
            name: "repo",
            state: State::Absent,
            detail: format!("no checkout at {}", src.display()),
            remedy: "vizfold install base".to_owned(),
            ..Default::default()
        };
    }
    let at = checkout_ref(src);
    let expected = release::tag();
    let problems = match &at {
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
        remedy: "vizfold update base".to_owned(),
        ..Default::default()
    }
}

/// Path keys, each with the file proving the claim (empty: the path itself). All must be in `CONFIG_KEYS`.
const CHECKED_PATHS: &[(&str, &str)] = &[
    // Not OPENFOLD_HOME: `repo` settles the checkout, and one missing directory must redden one component.
    ("OPENFOLD_PREFIX", ""),
    ("VIZFOLD_ENV_BASE", ""),
];

/// The same, but only while OpenFold is installed: its own uninstall must not read as a broken config.
const OPENFOLD_PATHS: &[(&str, &str)] = &[("OPENFOLD_DATA_DIR", "")];

/// Config keys only the scheduler can settle, grouped by the question that answers them.
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
    problems.extend(missing("entrypoint", env.join("bin").join(backend.slug())));
    if backend == Backend::Openfold {
        problems.extend(checkout_problem());
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

/// An editable install into the checkout: losing it breaks every fold at import, env still healthy.
fn checkout_problem() -> Option<String> {
    let src = config::vizfold_src();
    (!src.join(config::INSTALLER).is_file()).then(|| {
        format!(
            "the checkout it is installed from is gone: {}",
            src.display()
        )
    })
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

fn run_update(args: UpdateArgs) -> Result<(), DbErr> {
    match args.part.backend() {
        None => update_base(args.r#ref.as_deref()),
        Some(backend) if args.r#ref.is_some() => Err(DbErr::Custom(format!(
            "--ref moves the checkout, so it belongs to `vizfold update base`, not {}",
            backend.slug()
        ))),
        Some(backend) => reinstall(backend, args.yes),
    }
}

/// Neither installer reruns on drift, so a fresh checkout's scripts reach the env only through this.
fn reinstall(backend: Backend, yes: bool) -> Result<(), DbErr> {
    let targets = removal_plan(reinstall_paths(
        backend,
        &config::prefix(),
        &config::openfold_home(),
        &config::data_dir(),
    ));
    let headline = format!("Reinstalling {} first removes:", backend.slug());
    if !targets.is_empty() && !remove_confirmed(&headline, &targets, yes)? {
        return Ok(());
    }
    install_backend(backend)
}

/// Everything install planted except downloads: params are ~4 GB, or a mirror symlink tree, and
/// neither is install state. `data` may sit under a planted directory, spent entry by entry instead.
fn reinstall_paths(backend: Backend, prefix: &Path, home: &Path, data: &Path) -> Vec<PathBuf> {
    backend
        .install_paths(prefix, home)
        .into_iter()
        .flat_map(|path| {
            if !data.starts_with(&path) {
                return vec![path];
            }
            std::fs::read_dir(&path)
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| entry.path())
                .filter(|entry| !data.starts_with(entry))
                .collect()
        })
        .collect()
}

/// Move the checkout to a ref, by default this binary's own release tag.
fn update_base(wanted: Option<&str>) -> Result<(), DbErr> {
    let src = config::vizfold_src();
    let target = wanted.unwrap_or(&release::tag()).to_owned();
    if !src.join(config::INSTALLER).is_file() {
        return Err(DbErr::Custom(format!(
            "no vizfold checkout at {}; create one with `vizfold install base`",
            src.display()
        )));
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

/// One read-only git question as trimmed stdout; `None` when git cannot answer.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = git_cmd(dir, args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

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
    println!(
        "The checkout still runs {current}'s scripts. Bring it along with: vizfold update base"
    );
    Ok(())
}

/// Prove the download is a working binary of the version it claims before it replaces anything.
fn fetch_release(url: &str, staged: &Path, wanted: &str) -> Result<(), DbErr> {
    run_to_completion(
        "download",
        std::process::Command::new("curl")
            .args(["-fsSL", url, "-o"])
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
    // One plan for all parts: `removal_plan` folds paths inside the checkout under it, which
    // removing part by part would instead drop as already gone.
    let targets = match args.part {
        Some(part) => part.install_paths(&prefix, &home),
        None => Part::value_variants()
            .iter()
            .flat_map(|part| part.install_paths(&prefix, &home))
            .chain(shared_paths(&prefix, &home))
            .collect(),
    };

    let targets = removal_plan(targets);
    let what = args.part.map_or("vizfold", Part::slug);
    if targets.is_empty() {
        println!("Nothing to remove for {what}.");
        return Ok(());
    }

    let headline = format!("This removes everything {what} installed:");
    if !remove_confirmed(&headline, &targets, args.yes)? {
        return Ok(());
    }
    match args.part {
        Some(_) => println!(
            "\nKept: the config, the run database, and every other part.\nReinstall with: vizfold install {what}"
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

/// Print, confirm unless `yes`, remove. `false`: the user declined and nothing went.
fn remove_confirmed(headline: &str, targets: &[PathBuf], yes: bool) -> Result<bool, DbErr> {
    println!("{headline}");
    for target in targets {
        println!("  {}", target.display());
    }
    if !yes && !confirmed()? {
        println!("Aborted.");
        return Ok(false);
    }
    for target in targets {
        match remove_path(target) {
            Ok(()) => println!("removed {}", target.display()),
            Err(error) => eprintln!("warning: could not remove {}: {error}", target.display()),
        }
    }
    Ok(true)
}

/// A relative path means an empty config value resolved into one -- never delete off the cwd.
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

/// What no part owns, so only a full uninstall removes it.
fn shared_paths(prefix: &Path, home: &Path) -> Vec<PathBuf> {
    // Named entries, never the env base: only the `vizfold-` ones under it are ours.
    let mut paths = vec![
        config::env_dir("workbench"),
        prefix.join("vizfold.db"),
        config::config_file(),
        // The package cache outlives any one backend.
        prefix.join("mamba"),
    ];
    // The staged copy only; with no prefix settled the two are one path, the source tree.
    if prefix != home {
        paths.push(prefix.join("workbench"));
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

/// clap_complete's zsh script registers through `compdef`, which exists only once compinit has run,
/// and a bare ~/.zshrc never runs it. Carried here rather than in each place that evals this.
fn completion_script(shell: Shell) -> Vec<u8> {
    let mut script = match shell {
        Shell::Zsh => {
            b"autoload -Uz compinit; (( $+functions[compdef] )) || compinit -C\n".to_vec()
        }
        _ => Vec::new(),
    };
    clap_complete::generate(shell, &mut Cli::command(), "vizfold", &mut script);
    script
}

fn run_completions(shell: Option<Shell>) -> Result<(), DbErr> {
    let shell = shell.or_else(Shell::from_env).ok_or_else(|| {
        DbErr::Custom(format!(
            "cannot tell which shell {} is; name one, e.g. `vizfold completions bash`",
            std::env::var("SHELL").unwrap_or_else(|_| "$SHELL".to_owned())
        ))
    })?;
    std::io::Write::write_all(&mut std::io::stdout(), &completion_script(shell))
        .map_err(|error| DbErr::Custom(format!("failed to write completions: {error}")))
}

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

    let backends = args.backends_env();

    let node_modules = workbench.join("node_modules");
    let empty =
        std::fs::read_dir(&node_modules).map_or(true, |mut entries| entries.next().is_none());
    if empty {
        println!("Installing workbench dependencies (npm install)...");
        run_npm(&workbench, &node_bin, &["install"], &backends)?;
    }

    let port = args.port.unwrap_or(3000);
    println!(
        "Starting workbench at http://localhost:{port} ({})",
        if backends.is_empty() {
            "no backend installed"
        } else {
            &backends
        }
    );
    let port_arg = port.to_string();
    let mut npm_args = vec!["run", "dev"];
    if args.port.is_some() {
        npm_args.extend(["--", "--port", &port_arg]);
    }
    run_npm(&workbench, &node_bin, &npm_args, &backends)
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

/// Node from PATH when new enough, else its own env -- kept out of every backend env.
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

fn run_npm(dir: &Path, node_bin: &Path, args: &[&str], backends: &str) -> Result<(), DbErr> {
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
    // The resolved set, always explicit, so the dashboard needs no notion of what "all" means.
    command.env("VIZFOLD_BACKENDS", backends);
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

/// Fold every target in one execution, or replay a queued run by id. Re-registers artifacts (idempotent).
async fn run_run(database: &sea_orm::DatabaseConnection, args: RunArgs) -> Result<(), DbErr> {
    let run_id = match queued_run_id(&args.targets)? {
        Some(run_id) => run_id,
        None => {
            let resolved = resolve_targets(&args.targets)?;
            let backend = args.backend.map_or_else(default_backend, Ok)?;
            let run = match backend {
                Backend::Openfold => {
                    submit_openfold_run(database, &args, &resolved, &batch_inputs_dir()).await?
                }
                Backend::Esmfold => submit_esmfold_run(database, &args, &resolved).await?,
            };
            if !args.json {
                println!(
                    "Queued {} run {} ({}, {} residues)\n",
                    backend.label(),
                    run.id,
                    run.input_id,
                    resolved
                        .iter()
                        .map(|target| target.example.residues)
                        .sum::<usize>()
                );
            }
            if args.no_exec {
                if args.json {
                    println!("{}", json!({ "run_id": run.id, "status": run.status }));
                }
                return Ok(());
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

/// A lone integer target replays that run; mixed with anything else it is ambiguous.
fn queued_run_id(targets: &[String]) -> Result<Option<i32>, DbErr> {
    let ids: Vec<i32> = targets
        .iter()
        .filter_map(|target| target.parse().ok())
        .collect();
    match (ids.first(), targets.len()) {
        (None, _) => Ok(None),
        (Some(&run_id), 1) => Ok(Some(run_id)),
        (Some(_), _) => Err(DbErr::Custom(
            "a queued run id folds on its own; it cannot be mixed with other targets".to_owned(),
        )),
    }
}

#[derive(Debug)]
struct Target {
    fasta: PathBuf,
    example: examples::Example,
    bundled: bool,
}

fn tags(resolved: &[Target]) -> Vec<&str> {
    resolved
        .iter()
        .map(|target| target.example.id.as_str())
        .collect()
}

/// Every FASTA the targets name, in order: an example id, a path, or a directory of FASTAs.
fn resolve_targets(targets: &[String]) -> Result<Vec<Target>, DbErr> {
    let mut resolved: Vec<Target> = Vec::new();
    for target in targets {
        let path = local_path(target);
        let (fastas, bundled) = match examples::find(target) {
            // The example's file, not its directory: a stray sibling FASTA cannot join the fold.
            Some(example) => (
                examples::first_fasta(Path::new(&default_fasta_dir(&example.id)))
                    .into_iter()
                    .collect(),
                true,
            ),
            None if path.is_dir() => (examples::fasta_files(&path), false),
            // Anything path-shaped falls through to `read_fasta`, which names what it could not read.
            None if path.extension().is_some() || path.exists() => (vec![path], false),
            None => return Err(unknown_target(target)),
        };
        if fastas.is_empty() {
            return Err(DbErr::Custom(format!(
                "'{target}' holds no .fasta or .fa files"
            )));
        }
        for fasta in fastas {
            let example = read_fasta(&fasta)?;
            // The tag becomes a directory and a link name here; preflight takes nothing else either.
            if !example
                .id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return Err(DbErr::Custom(format!(
                    "'{}' is tagged '{}'; an OpenFold tag is letters, digits and underscores",
                    fasta.display(),
                    example.id
                )));
            }
            // Every OpenFold output is keyed by tag, so a repeat would overwrite the first.
            if resolved.iter().any(|other| other.example.id == example.id) {
                return Err(DbErr::Custom(format!(
                    "'{}' is folded twice by these targets",
                    example.id
                )));
            }
            let fasta = std::fs::canonicalize(&fasta).map_err(|error| {
                DbErr::Custom(format!("cannot resolve '{}': {error}", fasta.display()))
            })?;
            resolved.push(Target {
                fasta,
                example,
                bundled,
            });
        }
    }
    Ok(resolved)
}

/// Where staged batches live: outside the checkout, beside the run outputs the workbench serves.
fn batch_inputs_dir() -> PathBuf {
    config::prefix().join("runs/inputs")
}

/// cwd first, then the checkout: a relative target means what the shell means by it.
fn local_path(target: &str) -> PathBuf {
    let path = Path::new(target);
    std::env::current_dir()
        .ok()
        .map(|cwd| cwd.join(path))
        .filter(|candidate| candidate.exists())
        .unwrap_or_else(|| config::openfold_home().join(path))
}

/// One directory of symlinks, so OpenFold's single `fasta_dir` can name a whole batch. Each link
/// is replaced in place rather than the directory wiped, which would empty a concurrent run of it.
fn stage_batch(inputs_dir: &Path, resolved: &[Target]) -> Result<PathBuf, DbErr> {
    let dir = inputs_dir.join(batch_name(resolved));
    std::fs::create_dir_all(&dir).map_err(|error| {
        DbErr::Custom(format!(
            "cannot stage the batch at {}: {error}",
            dir.display()
        ))
    })?;
    for target in resolved {
        let link = dir.join(format!("{}.fasta", target.example.id));
        std::fs::remove_file(&link).ok();
        std::os::unix::fs::symlink(&target.fasta, &link).map_err(|error| {
            DbErr::Custom(format!(
                "cannot link {} into {}: {error}",
                target.fasta.display(),
                dir.display()
            ))
        })?;
    }
    Ok(dir)
}

/// `<tags>-<hash of the files behind them>`: same batch, same directory; same tags over different
/// files, a different one -- so a submit cannot relink a batch already recorded and waiting to fold.
/// Long tag lists collapse to first+count.
fn batch_name(resolved: &[Target]) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for target in resolved {
        target.fasta.hash(&mut hasher);
    }
    let joined = tags(resolved).join("+");
    let head = if joined.len() <= 80 {
        joined
    } else {
        format!("{}+{}more", resolved[0].example.id, resolved.len() - 1)
    };
    format!("{head}-{:x}", hasher.finish())
}

/// The seeded records a local run is built from; both backends differ only in which they name.
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

async fn submit_openfold_run(
    database: &sea_orm::DatabaseConnection,
    args: &RunArgs,
    resolved: &[Target],
    inputs_dir: &Path,
) -> Result<crate::core::entities::runs::Model, DbErr> {
    let catalog = local_catalog(database, Backend::Openfold).await?;
    let working_dir = &catalog.working_dir;
    // One target is its own `fasta_dir`; more than one needs a directory that names them all.
    let fasta_dir = match resolved {
        [only] => only.fasta.clone(),
        many => stage_batch(inputs_dir, many)?,
    };
    let data_dir_input = args
        .data_dir
        .clone()
        .unwrap_or_else(|| config::data_dir().to_string_lossy().into_owned());
    let data_dir = canonicalize_local_path("--data-dir", &data_dir_input, working_dir)?;
    // A user's own FASTA has no alignments, and must not borrow the examples'.
    let use_precomputed_alignments = args
        .use_precomputed_alignments
        .unwrap_or_else(|| resolved.iter().all(|target| target.bundled));
    let alignment_dir = if use_precomputed_alignments {
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
            json!(use_precomputed_alignments),
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
            // ponytail: `input_id.split('+')` is how preflight learns what the batch must hold.
            input_id: args
                .input_id
                .clone()
                .unwrap_or_else(|| tags(resolved).join("+")),
            input_sequence: resolved
                .iter()
                .map(|target| target.example.sequence.as_str())
                .collect::<Vec<_>>()
                .join(":"),
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

/// ESMFold reads one file and loads its model inside the fold, so a batch has nowhere to go.
async fn submit_esmfold_run(
    database: &sea_orm::DatabaseConnection,
    args: &RunArgs,
    resolved: &[Target],
) -> Result<crate::core::entities::runs::Model, DbErr> {
    let [target] = resolved else {
        return Err(DbErr::Custom(
            "ESMFold folds one target at a time; pass a single FASTA.".to_owned(),
        ));
    };
    let catalog = local_catalog(database, Backend::Esmfold).await?;
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
            input_id: args
                .input_id
                .clone()
                .unwrap_or_else(|| target.example.id.clone()),
            input_sequence: target.example.sequence.clone(),
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
                "fasta": target.fasta,
                "model_device": model_device,
            })
            .to_string(),
            provenance_json: Some(catalog.provenance),
        },
    )
    .await?;

    Ok(run)
}

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
fn read_fasta(path: &Path) -> Result<examples::Example, DbErr> {
    examples::from_path(path).ok_or_else(|| {
        DbErr::Custom(format!(
            "no FASTA record at '{}': expected a .fasta/.fa file with a '>' header and a sequence",
            path.display()
        ))
    })
}

/// Seeding runs immediately before every lookup, so a miss means the database is not one vizfold wrote.
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

    fn run_args(argv: &[&str]) -> RunArgs {
        let full: Vec<&str> = ["vizfold", "run"]
            .into_iter()
            .chain(argv.iter().copied())
            .collect();
        match Cli::try_parse_from(full)
            .expect("run argv should parse")
            .command
        {
            Command::Run(args) => args,
            other => panic!("not a run: {other:?}"),
        }
    }

    #[test]

    fn parses_install_part() {
        for (arg, want) in [
            ("base", Part::Base),
            ("openfold", Part::Openfold),
            ("esmfold", Part::Esmfold),
        ] {
            let cli = Cli::try_parse_from(["vizfold", "install", arg])
                .expect("install <part> should parse");
            assert!(matches!(cli.command, Command::Install(InstallArgs { part }) if part == want));
        }
        assert!(Cli::try_parse_from(["vizfold", "install"]).is_err());
        assert!(Cli::try_parse_from(["vizfold", "install", "rosetta"]).is_err());
    }

    /// `update` takes the same vocabulary `install` does; a bare one is a parse error, not a default.
    #[test]
    fn parses_update_part() {
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "update", "base", "--ref", "v0.1.0"])
                .expect("update base --ref should parse")
                .command,
            Command::Update(UpdateArgs { part: Part::Base, r#ref: Some(at), yes: false }) if at == "v0.1.0"
        ));
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "update", "openfold", "--yes"])
                .expect("update <backend> --yes should parse")
                .command,
            Command::Update(UpdateArgs {
                part: Part::Openfold,
                r#ref: None,
                yes: true
            })
        ));
        assert!(Cli::try_parse_from(["vizfold", "update"]).is_err());
        assert!(Cli::try_parse_from(["vizfold", "update", "vizfold"]).is_err());
    }

    /// Only base takes a ref; silently ignoring it on a backend would fake a version move.
    #[test]
    fn a_ref_on_a_backend_update_is_refused() {
        let update = |argv: &[&str]| match Cli::try_parse_from(argv).expect("argv").command {
            Command::Update(args) => args,
            other => panic!("not an update: {other:?}"),
        };
        assert!(
            super::run_update(update(&["vizfold", "update", "esmfold", "--ref", "v0.1.0"]))
                .is_err_and(|error| format!("{error}").contains("vizfold update base"))
        );
    }

    #[test]
    fn a_reinstall_keeps_the_downloads_and_takes_everything_else() {
        let base = std::env::temp_dir().join(format!("vizfold-reinstall-{}", std::process::id()));
        let (prefix, home) = (base.join("prefix"), base.join("checkout"));
        let full = Backend::Openfold.install_paths(&prefix, &home);
        let kept = super::reinstall_paths(Backend::Openfold, &prefix, &home, &config::data_dir());

        assert!(
            full.contains(&config::data_dir()),
            "install writes the data dir, or there is nothing to exclude"
        );
        assert!(
            !kept.contains(&config::data_dir()),
            "downloads are not install state"
        );
        assert!(
            kept.contains(&Backend::Openfold.env_prefix()),
            "the env is rebuilt"
        );
        assert_eq!(kept.len(), full.len() - 1, "only the data dir is spared");
    }

    /// The data dir sits inside the state dir install also plants: removing that whole takes the databases.
    #[test]
    fn a_reinstall_removes_no_directory_the_downloads_live_under() {
        let base = std::env::temp_dir().join(format!("vizfold-nested-data-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (prefix, home) = (base.join("prefix"), base.join("checkout"));
        let state = prefix.join("openfold");
        let data = state.join("data");
        std::fs::create_dir_all(&data).expect("fixture");
        std::fs::create_dir_all(state.join("pkgs")).expect("fixture");
        std::fs::write(state.join(".done"), "").expect("fixture");

        let kept = super::reinstall_paths(Backend::Openfold, &prefix, &home, &data);
        std::fs::remove_dir_all(&base).ok();

        assert!(
            !kept.iter().any(|path| data.starts_with(path)),
            "a removal target holds the downloads: {kept:?}"
        );
        // Without these the installer short-circuits on the sentinel.
        assert!(kept.contains(&state.join(".done")), "the sentinel must go");
        assert!(
            kept.contains(&state.join("pkgs")),
            "the rest of the state dir must go"
        );
    }

    /// Each backend reads its own key: a stray `ESMFOLD_ENV_PREFIX` cannot move OpenFold's env, and
    /// that reading is what bare `serve` hands the dashboard. One test, not two: they would race.
    #[test]
    fn backend_is_installed_tracks_its_own_env_prefix_key() {
        let base = std::env::temp_dir().join(format!("vizfold-backend-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("openfold")).unwrap();
        std::fs::create_dir_all(base.join("esmfold")).unwrap();
        // SAFETY: single-threaded test; pinned so env_prefix() does not read the real config.
        unsafe {
            std::env::set_var("OPENFOLD_ENV_PREFIX", base.join("openfold"));
            std::env::set_var("ESMFOLD_ENV_PREFIX", base.join("esmfold"));
        }
        assert_eq!(Backend::Esmfold.env_prefix(), base.join("esmfold"));
        assert_ne!(Backend::Openfold.env_prefix(), base.join("esmfold"));
        assert!(
            Backend::Esmfold.is_installed(),
            "an existing env dir reads as installed"
        );

        let served = |argv: &[&str]| match Cli::try_parse_from(argv)
            .expect("argv should parse")
            .command
        {
            Command::Serve(args) => args.backends_env(),
            other => panic!("expected serve, got {other:?}"),
        };
        assert_eq!(served(&["vizfold", "serve"]), "openfold,esmfold");
        // Named backends are honoured as given -- order included, since it picks the Fold default.
        assert_eq!(served(&["vizfold", "serve", "esmfold"]), "esmfold");
        assert_eq!(
            served(&["vizfold", "serve", "esmfold", "openfold"]),
            "esmfold,openfold"
        );

        std::fs::remove_dir_all(base.join("esmfold")).unwrap();
        assert!(
            !Backend::Esmfold.is_installed(),
            "a missing env dir reads as not installed"
        );
        assert_eq!(
            served(&["vizfold", "serve"]),
            "openfold",
            "bare serve offers what is installed, not what exists"
        );
        std::fs::remove_dir_all(&base).unwrap();
        unsafe {
            std::env::remove_var("OPENFOLD_ENV_PREFIX");
            std::env::remove_var("ESMFOLD_ENV_PREFIX");
        }
    }

    /// Verified load-bearing: a zsh that never ran compinit registers nothing, and it errors in bash.
    #[test]
    fn only_zsh_carries_the_compinit_prelude() {
        let script = |shell| String::from_utf8(completion_script(shell)).expect("utf-8");
        let zsh = script(Shell::Zsh);
        assert!(
            zsh.starts_with("autoload -Uz compinit;"),
            "the prelude must come before the script that needs compdef, got: {}",
            zsh.lines().next().unwrap_or_default()
        );
        for shell in [Shell::Bash, Shell::Fish] {
            assert!(
                !script(shell).contains("compinit"),
                "{shell} has no compdef to arrange for"
            );
        }
        // Nothing fails visibly if this name is wrong -- completion would simply do nothing.
        assert!(
            zsh.contains("compdef _vizfold vizfold\n"),
            "zsh binds `vizfold`"
        );
        assert!(
            script(Shell::Bash).contains("-F _vizfold -o bashdefault -o default vizfold\n"),
            "bash binds `vizfold`"
        );
    }

    /// A bare `uninstall` stays: it is the only thing that removes what no part owns.
    #[test]
    fn parses_uninstall() {
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "uninstall"])
                .expect("uninstall command should parse")
                .command,
            Command::Uninstall(UninstallArgs {
                part: None,
                yes: false
            })
        ));
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "uninstall", "--yes"])
                .expect("uninstall --yes should parse")
                .command,
            Command::Uninstall(UninstallArgs {
                part: None,
                yes: true
            })
        ));
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "uninstall", "esmfold"])
                .expect("uninstall <part> should parse")
                .command,
            Command::Uninstall(UninstallArgs {
                part: Some(Part::Esmfold),
                yes: false
            })
        ));
        assert!(matches!(
            Cli::try_parse_from(["vizfold", "uninstall", "base"])
                .expect("uninstall base should parse")
                .command,
            Command::Uninstall(UninstallArgs {
                part: Some(Part::Base),
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
        // The cache serves the workbench env too.
        assert!(
            !paths.contains(&prefix.join("mamba")),
            "the cache is shared"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// The reinstall invariant: uninstall takes exactly what install puts back, and nothing else.
    #[test]
    fn one_part_leaves_the_others_and_everything_shared_alone() {
        let base = std::env::temp_dir().join(format!("vizfold-scoped-{}", std::process::id()));
        let (prefix, home) = (base.join("prefix"), base.join("checkout"));

        let owned: Vec<Vec<PathBuf>> = Part::value_variants()
            .iter()
            .map(|part| part.install_paths(&prefix, &home))
            .collect();
        let shared = super::shared_paths(&prefix, &home);

        for (nth, paths) in owned.iter().enumerate() {
            for other in owned.iter().enumerate().filter(|(i, _)| *i != nth) {
                assert!(
                    paths.iter().all(|path| !other.1.contains(path)),
                    "two parts share a path"
                );
            }
        }
        for owned in owned.iter().flatten() {
            assert!(
                !shared.contains(owned),
                "{} is a part's, not shared",
                owned.display()
            );
        }
        assert!(
            super::base_paths(&prefix) == vec![config::default_src()]
                || config::vizfold_src() != config::default_src(),
            "base owns the default checkout and nothing else"
        );
        assert!(
            super::base_paths(&config::default_src().join("prefix")).is_empty(),
            "a prefix inside the checkout keeps the checkout"
        );
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
        assert!(
            !super::shared_paths(&home, &home).contains(&home.join("workbench")),
            "the checkout's own workbench must survive"
        );
    }

    /// `uninstall` is `rm -rf`: no relative path may reach it, and no path an outer target covers.
    #[test]
    fn the_removal_plan_keeps_only_absolute_uncovered_paths_that_exist() {
        let base = std::env::temp_dir().join(format!("vizfold-plan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("outer/inner")).expect("fixture");

        let plan = super::removal_plan(vec![
            // Exists relative to the cwd tests run in: without the guard, uninstall deletes it.
            PathBuf::from("Cargo.toml"),
            base.join("does-not-exist"),
            base.join("outer/inner"),
            base.join("outer"),
            base.join("outer"), // duplicate: dedup is what keeps the plan one entry
        ]);

        assert_eq!(plan, vec![base.join("outer")]);
        std::fs::remove_dir_all(&base).ok();
    }

    /// The gate table by the names it produces: `status` and base's own verbs stay off `repo`.
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
        assert_eq!(names(&["vizfold", "uninstall"]), Vec::<&str>::new());
        assert_eq!(names(&["vizfold", "install", "base"]), ["core deps"]);
        assert_eq!(names(&["vizfold", "update", "base"]), ["core deps"]);
        assert_eq!(
            names(&["vizfold", "install", "openfold"]),
            ["core deps", "repo"]
        );
        assert_eq!(
            names(&["vizfold", "update", "openfold"]),
            ["core deps", "repo"]
        );
        assert_eq!(names(&["vizfold", "list", "examples"]), ["repo"]);
        assert_eq!(
            names(&["vizfold", "serve"]),
            ["core deps", "repo", "config"]
        );
        assert_eq!(
            names(&["vizfold", "serve", "openfold", "esmfold"]),
            ["core deps", "repo", "config", "openfold", "esmfold"]
        );
        assert!(Cli::try_parse_from(["vizfold", "serve", "base"]).is_err());
        assert_eq!(
            names(&["vizfold", "download", "openfold"]),
            ["repo", "config", "openfold"]
        );
        assert_eq!(
            names(&["vizfold", "run", "1UBQ_1"]),
            ["core deps", "repo", "config", "openfold"]
        );
        // `--no-exec` only writes the row, so it must not gate on an installed environment.
        assert_eq!(
            names(&[
                "vizfold",
                "run",
                "x.fasta",
                "--no-exec",
                "--backend",
                "esmfold"
            ]),
            ["config", "esmfold"]
        );

        let core = super::core_deps_health();
        assert!(
            core.detail.ends_with("micromamba"),
            "core deps must look for micromamba, got {}",
            core.detail
        );
    }

    /// Never cloned and drifted need different fixes; both once read BROKEN and pointed at the updater.
    #[test]
    fn an_absent_checkout_is_not_a_broken_one() {
        let dir = std::env::temp_dir().join(format!("vizfold-repo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(config::INSTALLER).parent().unwrap()).unwrap();
        let absent = super::repo_health_at(&dir);
        std::fs::write(dir.join(config::INSTALLER), "").unwrap();
        let present = super::repo_health_at(&dir);
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(absent.state, State::Absent);
        assert!(
            absent.problems.is_empty(),
            "absent is not a list of problems"
        );
        assert_eq!(absent.remedy, "vizfold install base");
        assert_ne!(
            present.state,
            State::Absent,
            "a checkout is there to be judged"
        );
        assert_eq!(present.remedy, "vizfold update base");
    }

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

    /// A failed fetch leaves a non-executable file; reading it as installed fails deep in an installer.
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
        assert!(!dst.join(".next").exists());
        assert!(dst.join("node_modules/installed.js").is_file());
        assert!(!dst.join("node_modules/dep.js").exists());
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn copy_tree_skips_a_symlinked_directory() {
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
        assert!(!dst.join("public/runs").exists());
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

    /// One positional list, both backends' flags on it, and the default-true bools taking `--flag=false`.
    #[test]
    fn parses_run() {
        let one = run_args(&["1"]);
        assert_eq!(one.targets, ["1"]);
        assert!(one.backend.is_none());
        assert!(!one.no_exec && !one.json);
        assert!(one.attn && one.save_outputs);
        assert_eq!((one.residue_idx, one.num_recycles_save), (1, 1));
        // Unset, not false: whether alignments are precomputed depends on what the targets are.
        assert_eq!(one.use_precomputed_alignments, None);
        assert_eq!(one.model, "facebook/esmfold_v1");
        assert_eq!(one.trace_mode, "attention+activations");
        assert_eq!(one.layers, "all");

        let batch = run_args(&[
            "1UBQ_1",
            "./some/dir",
            "--attn=false",
            "--use-precomputed-alignments=false",
            "--no-exec",
            "--trace-mode",
            "attention",
            "--save-fp16",
        ]);
        assert_eq!(batch.targets, ["1UBQ_1", "./some/dir"]);
        assert!(!batch.attn && batch.no_exec && batch.save_fp16);
        assert_eq!(batch.use_precomputed_alignments, Some(false));
        assert_eq!(batch.trace_mode, "attention");

        // A bare `run` is a parse error, not an empty batch.
        assert!(Cli::try_parse_from(["vizfold", "run"]).is_err());
    }

    #[test]
    fn a_run_id_refuses_to_share_the_command_line() {
        let ids = |argv: &[&str]| super::queued_run_id(&run_args(argv).targets);
        assert_eq!(ids(&["42"]).expect("a lone id replays"), Some(42));
        assert_eq!(ids(&["1UBQ_1"]).expect("no id here"), None);
        assert_eq!(ids(&["1UBQ_1", "6KWC_1"]).expect("no id here"), None);
        assert!(ids(&["42", "6KWC_1"]).is_err());
    }

    fn fake_target(id: &str, fasta: &str) -> Target {
        Target {
            fasta: PathBuf::from(fasta),
            example: examples::Example {
                id: id.to_owned(),
                residues: 1,
                description: String::new(),
                sequence: "M".to_owned(),
            },
            bundled: false,
        }
    }

    /// Two tags name the directory; twenty collapse to a stable short name.
    #[test]
    fn a_batch_directory_is_named_after_its_tags_and_its_files() {
        let pair = [
            fake_target("1UBQ_1", "/a/1UBQ.fasta"),
            fake_target("6KWC_1", "/a/6KWC.fasta"),
        ];
        let name = super::batch_name(&pair);
        assert!(name.starts_with("1UBQ_1+6KWC_1-"), "{name}");
        assert_eq!(
            name,
            super::batch_name(&pair),
            "the same batch, the same directory"
        );

        // Same tags, other files: a recorded batch cannot be relinked by a later submit.
        let elsewhere = [
            fake_target("1UBQ_1", "/b/1UBQ.fasta"),
            fake_target("6KWC_1", "/a/6KWC.fasta"),
        ];
        assert_ne!(name, super::batch_name(&elsewhere));

        let many: Vec<Target> = (0..20)
            .map(|index| fake_target("1UBQ_1", &format!("/a/{index}.fasta")))
            .collect();
        let long = super::batch_name(&many);
        assert!(long.len() < 40, "{long} is too long for a directory name");
        assert!(long.starts_with("1UBQ_1+19more-"), "{long}");
    }

    async fn seeded_database() -> Result<sea_orm::DatabaseConnection, DbErr> {
        let database = Database::connect("sqlite::memory:").await?;
        database
            .execute(Statement::from_string(
                database.get_database_backend(),
                "PRAGMA foreign_keys = ON".to_owned(),
            ))
            .await?;
        db::migrate_database(&database).await?;
        seed::seed_defaults(&database).await?;
        Ok(database)
    }

    #[tokio::test]
    async fn submit_openfold_run_uses_seeded_records() -> Result<(), DbErr> {
        let local_path = std::fs::canonicalize(crate::core::config::openfold_home())
            .expect("OpenFold home should be canonicalizable")
            .display()
            .to_string();
        let database = seeded_database().await?;

        // A real FASTA, so this pins the id and sequence derivation too.
        let fasta = std::fs::canonicalize(
            crate::core::examples::monomer_dir().join("fasta_dir_6KWC/6KWC.fasta"),
        )
        .expect("the bundled 6KWC example should exist")
        .display()
        .to_string();

        let args = run_args(&[
            &fasta,
            "--data-dir",
            &local_path,
            "--alignment-dir",
            &local_path,
            "--model-device",
            "cpu",
            // Over the seeded target's cpus.maximum of 14, so the run must record the clamped value.
            "--cpus",
            "18",
            "--use-precomputed-alignments=true",
        ]);
        submit_openfold_run(
            &database,
            &args,
            &super::resolve_targets(&args.targets)?,
            &super::batch_inputs_dir(),
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
            // The file itself, not the directory holding it.
            json!({"fasta_dir": fasta, "data_dir": local_path, "alignment_dir": local_path, "residue_idx": 1, "use_precomputed_alignments": true, "model_device": "cpu", "cpus": 14})
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

    /// Regression: a bundled example names a directory, and ESMFold's `--fasta` needs the file in it.
    #[test]
    fn an_example_resolves_to_its_fasta_file_not_its_directory() {
        let resolved = super::resolve_targets(&["6KWC_1".to_owned()]).expect("6KWC_1 resolves");
        let [target] = resolved.as_slice() else {
            panic!("one target, got {}", resolved.len());
        };
        assert!(
            std::path::Path::new(&target.fasta).is_file(),
            "resolved '{}' should be the file, not the directory",
            target.fasta.display()
        );
    }

    #[tokio::test]
    async fn two_targets_submit_one_run_over_a_staged_directory() -> Result<(), DbErr> {
        let database = seeded_database().await?;
        let home = config::openfold_home().display().to_string();
        let args = run_args(&["1UBQ_1", "6KWC_1", "--data-dir", &home]);
        let resolved = super::resolve_targets(&args.targets)?;

        assert!(
            resolved.iter().all(|target| target.bundled),
            "bundled examples default to their precomputed alignments"
        );
        // Not the real prefix: a submit host may have it read-only, and this test writes there.
        let inputs = std::env::temp_dir().join(format!("vizfold-batch-{}", std::process::id()));

        submit_openfold_run(&database, &args, &resolved, &inputs).await?;

        let runs = runs::list_runs(&database).await?;
        assert_eq!(runs.len(), 1, "one row per invocation");
        assert_eq!(runs[0].input_id, "1UBQ_1+6KWC_1");
        let execution: serde_json::Value =
            serde_json::from_str(&runs[0].execution_parameters_json).expect("valid JSON");
        assert_eq!(execution["use_precomputed_alignments"], json!(true));

        let staged = PathBuf::from(execution["fasta_dir"].as_str().expect("a fasta_dir"));
        assert_eq!(staged.parent(), Some(inputs.as_path()));
        assert!(
            staged
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("1UBQ_1+6KWC_1-")),
            "{}",
            staged.display()
        );
        let mut staged_names: Vec<String> = std::fs::read_dir(&staged)
            .expect("the batch should be staged")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        staged_names.sort();
        assert_eq!(staged_names, ["1UBQ_1.fasta", "6KWC_1.fasta"]);
        // Links, not copies, so the batch directory costs nothing and cannot drift from its source.
        assert!(staged.join("1UBQ_1.fasta").is_file());
        std::fs::remove_dir_all(&inputs).ok();
        Ok(())
    }

    #[tokio::test]
    async fn esmfold_refuses_more_than_one_target() -> Result<(), DbErr> {
        let database = seeded_database().await?;
        let args = run_args(&["1UBQ_1", "6KWC_1", "--backend", "esmfold"]);
        let resolved = super::resolve_targets(&args.targets)?;

        let error = submit_esmfold_run(&database, &args, &resolved)
            .await
            .expect_err("a batch should be refused");

        assert!(error.to_string().contains("one target at a time"));
        Ok(())
    }

    /// A repeated tag is refused, not overwritten.
    #[test]
    fn a_directory_target_resolves_to_every_fasta_in_it() {
        let dir = crate::core::examples::monomer_dir().join("fasta_dir_6KWC");
        let resolved = super::resolve_targets(&[dir.display().to_string()]).expect("6KWC exists");
        assert_eq!(super::tags(&resolved), ["6KWC_1"]);
        assert!(
            !resolved[0].bundled,
            "a path is the user's own, so it borrows no alignments"
        );

        let twice = super::resolve_targets(&["6KWC_1".into(), dir.display().to_string()]);
        assert!(
            twice.is_err_and(|error| error.to_string().contains("folded twice")),
            "one tag cannot appear twice in a batch"
        );
    }

    /// A tag names a staged directory and a link, so a path in a FASTA header must not reach the disk.
    #[test]
    fn a_header_that_is_not_a_tag_is_refused() {
        let dir = std::env::temp_dir().join(format!("vizfold-tag-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture");
        let fasta = dir.join("evil.fasta");
        std::fs::write(&fasta, ">../../../etc/passwd\nMQIFVKTL\n").expect("fixture");

        let error = super::resolve_targets(&[fasta.display().to_string()])
            .expect_err("a traversal tag should be refused");

        assert!(
            error
                .to_string()
                .contains("letters, digits and underscores"),
            "{error}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn submit_openfold_run_reports_missing_local_path() -> Result<(), DbErr> {
        let database = seeded_database().await?;
        let missing_path = "definitely-missing-vizfold-local-path";
        let args = run_args(&["6KWC_1", "--data-dir", missing_path]);

        let error = submit_openfold_run(
            &database,
            &args,
            &super::resolve_targets(&args.targets)?,
            &super::batch_inputs_dir(),
        )
        .await
        .expect_err("missing local path should fail");

        assert!(error.to_string().contains(
            "--data-dir original path 'definitely-missing-vizfold-local-path' could not be resolved"
        ));
        assert!(
            error
                .to_string()
                .contains(&crate::core::config::openfold_home().display().to_string())
        );
        Ok(())
    }

    #[test]
    fn an_unknown_target_is_refused_before_anything_is_submitted() {
        assert!(super::resolve_targets(&["not-a-thing".into()]).is_err());
        assert!(
            super::resolve_targets(&["./typo.fasta".into()])
                .is_err_and(|error| error.to_string().contains("no FASTA record at")),
            "a path-shaped target is reported as a path"
        );
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
        assert_eq!(
            super::model_device_for(config::SlurmContext::None, Some("gpuA100x4"), None),
            "cuda:0"
        );
    }

    #[test]
    fn model_device_inside_an_allocation_trusts_the_local_probe() {
        assert_eq!(
            super::model_device_for(config::SlurmContext::InAllocation, Some("gpuA100x4"), None),
            "cpu"
        );
    }

    /// An absent or unparseable maximum must not clamp the request to something arbitrary.
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
