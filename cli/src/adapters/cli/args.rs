use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use std::path::{Path, PathBuf};

use crate::core::config;

#[derive(Debug, Parser)]
#[command(
    name = "vizfold",
    version,
    about = "VizFold executor administration CLI",
    arg_required_else_help = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub(super) command: Command,
}

#[derive(Debug, Subcommand)]
// `run` carries every fold flag; one enum is parsed, once, per process.
#[allow(clippy::large_enum_variant)]
pub(super) enum Command {
    /// Install the checkout everything runs from (`repo`), or a model backend from it.
    Install(InstallArgs),
    /// Download a backend's data (OpenFold AlphaFold2 databases/params).
    Download(DownloadArgs),
    /// Show resolved config, which backends are installed, and whether it all checks out.
    Status,
    /// Remove one part, or everything the install generated.
    Uninstall(UninstallArgs),
    /// Move the checkout to this binary's release (`repo`), or reinstall a backend from it.
    Update(UpdateArgs),
    /// Replace this binary with the latest release. Run `update repo` after, for the checkout.
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
pub(super) struct CompletionsArgs {
    /// Shell to emit for. Defaults to the one `$SHELL` names.
    #[arg(value_enum)]
    pub(super) shell: Option<Shell>,
}

#[derive(Debug, Args)]
pub(super) struct InstallArgs {
    /// What to install: the checkout everything runs from, or a model backend from it.
    #[arg(value_enum)]
    pub(super) part: Part,
}

/// What a lifecycle verb acts on: the checkout every backend installs from, or one backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(super) enum Part {
    Repo,
    Openfold,
    Esmfold,
}

impl Part {
    pub(super) fn backend(self) -> Option<Backend> {
        match self {
            Self::Repo => None,
            Self::Openfold => Some(Backend::Openfold),
            Self::Esmfold => Some(Backend::Esmfold),
        }
    }

    pub(super) fn slug(self) -> &'static str {
        self.backend().map_or("repo", Backend::slug)
    }

    /// Exactly what `install <part>` puts there, so `uninstall <part>` takes back the same set.
    pub(super) fn install_paths(self, prefix: &Path, home: &Path) -> Vec<PathBuf> {
        self.backend().map_or_else(
            || repo_paths(prefix),
            |backend| backend.install_paths(prefix, home),
        )
    }
}

/// The checkout, and only the clone vizfold made itself: never a user-supplied `OPENFOLD_HOME`.
pub(super) fn repo_paths(prefix: &Path) -> Vec<PathBuf> {
    let repo = config::vizfold_repo();
    (repo == config::default_repo() && !prefix.starts_with(&repo))
        .then_some(repo)
        .into_iter()
        .collect()
}

#[derive(Debug, Args)]
pub(super) struct DownloadArgs {
    /// Model backend whose data to download.
    #[arg(value_enum)]
    pub(super) backend: Backend,
    /// Dataset to fetch: `all` (the full AlphaFold2 set) or a single db name (e.g. `uniref90`,
    /// `pdb70`, `bfd`, `alphafold_params`), mapped to `downloaders/<backend>/download_<name>.sh`.
    #[arg(default_value = "all")]
    pub(super) dataset: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(super) enum Backend {
    Openfold,
    Esmfold,
}

/// What a backend is called and where its scripts live -- the whole of what the two differ by,
/// in one place, so adding a third is a row rather than a sweep through five `match` arms.
struct Names {
    slug: &'static str,
    label: &'static str,
    /// Installer script, relative to the checkout: each backend owns one under `backends/<name>/install/`.
    installer: &'static str,
    /// Downloader dir, relative to the checkout. `None` for a backend that fetches at run time.
    downloader_dir: Option<&'static str>,
}

const OPENFOLD: Names = Names {
    slug: "openfold",
    label: "OpenFold",
    installer: config::INSTALLER,
    downloader_dir: Some("downloaders/openfold"),
};

const ESMFOLD: Names = Names {
    slug: "esmfold",
    label: "ESMFold",
    installer: "backends/esmfold/install/install.sh",
    downloader_dir: None,
};

impl Backend {
    fn names(self) -> &'static Names {
        match self {
            Self::Openfold => &OPENFOLD,
            Self::Esmfold => &ESMFOLD,
        }
    }

    pub(super) fn slug(self) -> &'static str {
        self.names().slug
    }

    pub(super) fn label(self) -> &'static str {
        self.names().label
    }

    pub(super) fn installer(self) -> &'static str {
        self.names().installer
    }

    pub(super) fn downloader_dir(self) -> Option<&'static str> {
        self.names().downloader_dir
    }

    pub(super) fn env_prefix(self) -> PathBuf {
        match self {
            Self::Openfold => config::openfold_env_prefix(),
            Self::Esmfold => config::esmfold_env_prefix(),
        }
    }

    pub(super) fn is_installed(self) -> bool {
        self.env_prefix().is_dir()
    }

    pub(super) fn dir(self, home: &Path) -> PathBuf {
        home.join("backends").join(self.slug())
    }

    /// Exactly what `install <backend>` puts back. Fold outputs are results, never install state.
    pub(super) fn install_paths(self, prefix: &Path, home: &Path) -> Vec<PathBuf> {
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
pub(super) struct UninstallArgs {
    /// Part to remove. Omit to remove every part, the config, and the run database too.
    #[arg(value_enum)]
    pub(super) part: Option<Part>,
    /// Remove without the confirmation prompt.
    #[arg(long, short = 'y')]
    pub(super) yes: bool,
}

#[derive(Debug, Args)]
pub(super) struct UpdateArgs {
    /// What to bring current: the checkout, or a backend reinstalled from it.
    #[arg(value_enum)]
    pub(super) part: Part,
    /// Tag or branch to move the checkout to. Defaults to this binary's own release tag. `repo` only.
    #[arg(long, value_name = "REF")]
    pub(super) r#ref: Option<String>,
    /// Reinstall without the confirmation prompt.
    #[arg(long, short = 'y')]
    pub(super) yes: bool,
}

#[derive(Debug, Args)]
pub(super) struct SelfUpdateArgs {
    /// Release to install (e.g. v0.5.1). Defaults to the latest published release.
    #[arg(long, value_name = "TAG")]
    pub(super) version: Option<String>,
    /// Re-download even when this binary already is that release.
    #[arg(long)]
    pub(super) force: bool,
}

#[derive(Debug, Args)]
pub(super) struct ServeArgs {
    /// Backends the dashboard folds with and lists runs for. Defaults to every one installed.
    #[arg(value_enum)]
    pub(super) backends: Vec<Backend>,
    /// Port for the dashboard dev server. Defaults to 3000.
    #[arg(long)]
    pub(super) port: Option<u16>,
}

impl ServeArgs {
    /// For the dashboard: the named backends in order, else all installed -- never one that cannot run.
    pub(super) fn backends_env(&self) -> String {
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
pub(super) struct ListArgs {
    #[command(subcommand)]
    pub(super) resource: ListResource,
}

#[derive(Debug, Args)]
pub(super) struct RunArgs {
    /// What to fold: bundled protein ids (`vizfold list proteins`), FASTA files, or directories of
    /// FASTAs -- several fold in one execution. A lone run id replays that queued run instead.
    #[arg(required = true)]
    pub(super) targets: Vec<String>,
    /// Backend. Defaults to the only one installed, else openfold. A queued run carries its own.
    #[arg(long, value_enum)]
    pub(super) backend: Option<Backend>,
    /// Record the run and stop, without folding it. `vizfold run <id>` folds it later.
    #[arg(long)]
    pub(super) no_exec: bool,
    /// Name recorded for this run. Defaults to the folded tags joined with `+`, the only value
    /// preflight takes.
    #[arg(long)]
    pub(super) input_id: Option<String>,
    /// Print only the run as JSON, for tools driving the CLI.
    #[arg(long)]
    pub(super) json: bool,
    /// OpenFold data directory. Defaults to the config `OPENFOLD_DATA_DIR`.
    #[arg(long)]
    pub(super) data_dir: Option<String>,
    /// Precomputed alignments directory. Defaults to <OPENFOLD_HOME>/examples/monomer/alignments.
    #[arg(long)]
    pub(super) alignment_dir: Option<String>,
    /// Torch device. Defaults to cuda:0 when a GPU partition is configured to srun onto (the
    /// HPC flow) or a GPU is visible locally, otherwise cpu.
    #[arg(long)]
    pub(super) model_device: Option<String>,
    /// CPU threads. Defaults to this machine's core count, clamped to the execution target's maximum.
    #[arg(long)]
    pub(super) cpus: Option<i64>,
    /// Residue index offset passed through to the model (OpenFold).
    #[arg(long, default_value_t = 1)]
    pub(super) residue_idx: i64,
    /// Dump per-layer, per-head attention maps, under `attention/<tag>/` (OpenFold).
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub(super) attn: bool,
    /// Write the model's raw output tensors alongside the structure (OpenFold).
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub(super) save_outputs: bool,
    /// How many recycling iterations to keep outputs for (OpenFold).
    #[arg(long, default_value_t = 1)]
    pub(super) num_recycles_save: i64,
    /// Use the precomputed alignments in `--alignment-dir`. On by default only when every target is
    /// a bundled example; `--use-precomputed-alignments=false` forces the full MSA pipeline.
    #[arg(long, action = ArgAction::Set)]
    pub(super) use_precomputed_alignments: Option<bool>,
    /// HuggingFace model id (ESMFold).
    #[arg(long, default_value = "facebook/esmfold_v1")]
    pub(super) model: String,
    /// What to extract: none, attention, activations, or attention+activations (ESMFold).
    #[arg(long, default_value = "attention+activations")]
    pub(super) trace_mode: String,
    /// Layers to save: `all` or a comma/colon list (ESMFold).
    #[arg(long, default_value = "all")]
    pub(super) layers: String,
    /// Model dtype (ESMFold).
    #[arg(long, default_value = "float32")]
    pub(super) dtype: String,
    /// Save trace tensors in fp16 to reduce size (ESMFold).
    #[arg(long)]
    pub(super) save_fp16: bool,
    /// Capture IPA attention and per-recycle backbone from the structure module (ESMFold).
    #[arg(long)]
    pub(super) structure_traces: bool,
}

#[derive(Debug, Subcommand)]
pub(super) enum ListResource {
    /// List the proteins available to fold, and which carry precomputed alignments.
    Proteins {
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
pub(super) struct ShowArgs {
    #[command(subcommand)]
    pub(super) resource: ShowResource,
}

#[derive(Debug, Subcommand)]
pub(super) enum ShowResource {
    /// Show a run and its artifacts.
    Run { run_id: i32 },
}

#[cfg(test)]
mod tests {
    use super::super::test_support::run_args;
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_install_part() {
        for (arg, want) in [
            ("repo", Part::Repo),
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
            Cli::try_parse_from(["vizfold", "update", "repo", "--ref", "v0.1.0"])
                .expect("update repo --ref should parse")
                .command,
            Command::Update(UpdateArgs { part: Part::Repo, r#ref: Some(at), yes: false }) if at == "v0.1.0"
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

    /// Each backend reads its own key: a stray `ESMFOLD_ENV_PREFIX` cannot move OpenFold's env, and
    /// that reading is what bare `serve` hands the dashboard. One test, not two: they would race.
    #[test]
    fn backend_is_installed_tracks_its_own_env_prefix_key() {
        let _env = crate::core::test_support::env_lock();
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
            Cli::try_parse_from(["vizfold", "uninstall", "repo"])
                .expect("uninstall repo should parse")
                .command,
            Command::Uninstall(UninstallArgs {
                part: Some(Part::Repo),
                yes: false
            })
        ));
    }

    #[test]
    fn openfold_install_paths_cover_generated_trees_but_not_run_outputs() {
        let _env = crate::core::test_support::env_lock();
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
}
