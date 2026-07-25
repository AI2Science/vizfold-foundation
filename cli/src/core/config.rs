use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Path to the install-time config written by `lib/config.sh` (`config::save`).
/// This flat JSON map is the single source of storage, DB, and cluster-inferrable paths.
pub fn config_file() -> PathBuf {
    if let Ok(explicit) = std::env::var("VIZFOLD_CONFIG")
        && !explicit.is_empty()
    {
        return PathBuf::from(explicit);
    }
    let base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| format!("{}/.config", home_dir()));
    PathBuf::from(base).join("vizfold").join("vizfold.json")
}

pub fn is_initialized() -> bool {
    config_file().is_file()
}

fn home_dir() -> String {
    std::env::var("HOME").unwrap_or_else(|_| ".".to_owned())
}

fn vizfold_config() -> &'static Map<String, Value> {
    static CONFIG: OnceLock<Map<String, Value>> = OnceLock::new();
    CONFIG.get_or_init(|| {
        std::fs::read_to_string(config_file())
            .ok()
            .and_then(|c| serde_json::from_str::<Value>(&c).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default()
    })
}

/// Empty is unset. The config carries a fixed key set, so every name the install did not settle is
/// present with an empty value, and those must fall through to the caller's default like a missing
/// key -- exactly as an empty env var already does.
fn non_empty(value: Option<&str>) -> Option<String> {
    value.filter(|v| !v.is_empty()).map(str::to_owned)
}

/// inline env var of the same name > vizfold.json entry > None.
fn resolved(key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(key)
        && !v.is_empty()
    {
        return Some(v);
    }
    non_empty(vizfold_config().get(key).and_then(Value::as_str))
}

pub fn openfold_home() -> PathBuf {
    resolved("OPENFOLD_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(repository_root)
}

/// Repo checkout holding `backends/openfold/install/install.sh` (what `vizfold install` runs, cloning it if absent).
/// `OPENFOLD_HOME` -- the config's own name for it -- else the default clone location (`$HOME/vizfold-src`).
pub fn vizfold_src() -> PathBuf {
    resolved("OPENFOLD_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(default_src)
}

/// Where `vizfold install` clones the checkout when nothing points at an existing one --
/// the only checkout `vizfold uninstall` may delete.
pub fn default_src() -> PathBuf {
    PathBuf::from(format!("{}/vizfold-src", home_dir()))
}

pub fn data_dir() -> PathBuf {
    resolved("OPENFOLD_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| openfold_home().join("data"))
}

/// The one directory holding every environment the install creates. Mirrors `vizfold::env_base`
/// in `lib/config.sh`.
pub fn env_base() -> PathBuf {
    resolved("VIZFOLD_ENV_BASE")
        .map(PathBuf::from)
        .unwrap_or_else(|| prefix().join("envs"))
}

/// `<env base>/vizfold-<backend>` — a fixed name per backend, conda env and venv alike, so nothing
/// has to be told where any of them is.
pub fn env_dir(name: &str) -> PathBuf {
    env_base().join(format!("vizfold-{name}"))
}

/// micromamba env prefix for local OpenFold execution. Every OpenFold install records it
/// (`setup::config_save` writes `OPENFOLD_ENV_PREFIX=$CONDA_PREFIX`), so the config normally
/// answers; the `<env base>/vizfold-openfold` fallback covers a config that left the key empty --
/// only the ESMFold backend was installed.
pub fn openfold_env_prefix() -> PathBuf {
    resolved("OPENFOLD_ENV_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| env_dir("openfold"))
}

/// venv prefix for the ESMFold backend, same story as `openfold_env_prefix`.
pub fn esmfold_env_prefix() -> PathBuf {
    resolved("ESMFOLD_ENV_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| env_dir("esmfold"))
}

/// The install-resolved config map as sorted (key, value) string pairs, for `vizfold status`.
pub fn config_entries() -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = vizfold_config()
        .iter()
        .filter_map(|(key, value)| Some((key.clone(), value.as_str()?.to_owned())))
        .collect();
    entries.sort();
    entries
}

pub fn prefix() -> PathBuf {
    resolved("OPENFOLD_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(openfold_home)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlurmContext {
    InStep,
    InAllocation,
    None,
}

impl SlurmContext {
    pub fn detect() -> Self {
        if std::env::var_os("SLURM_STEP_ID").is_some() {
            Self::InStep
        } else if std::env::var_os("SLURM_JOB_ID").is_some() {
            Self::InAllocation
        } else {
            Self::None
        }
    }
}

/// Empty string counts as absent, same as an unset env var.
fn or_default<'a>(value: Option<&'a str>, default: &'a str) -> &'a str {
    value.filter(|v| !v.is_empty()).unwrap_or(default)
}

/// SLURM launch prefix for a fold, mirroring `setup::fold_vars`. Empty means run bare --
/// either we are already on the node, or no GPU partition is configured (the workstation case).
pub fn gpu_launch(
    context: SlurmContext,
    partition: Option<&str>,
    account: Option<&str>,
    gres: Option<&str>,
    resources: Option<&str>,
    time: Option<&str>,
) -> Vec<String> {
    match context {
        SlurmContext::InStep => return Vec::new(),
        SlurmContext::InAllocation => return vec!["srun".to_owned(), "--ntasks=1".to_owned()],
        SlurmContext::None => {}
    }
    let Some(partition) = partition.filter(|p| !p.is_empty()) else {
        return Vec::new();
    };
    let mut args = vec!["srun".to_owned()];
    if let Some(account) = account.filter(|a| !a.is_empty()) {
        args.push("-A".to_owned());
        args.push(account.to_owned());
    }
    args.push("-p".to_owned());
    args.push(partition.to_owned());
    args.push(format!("--gres={}", or_default(gres, "gpu:1")));
    // Several space-separated flags in one value: setup::fold_vars relies on word splitting too.
    args.extend(
        or_default(resources, "--cpus-per-task=8 --mem=32G")
            .split_whitespace()
            .map(str::to_owned),
    );
    args.push("-t".to_owned());
    args.push(or_default(time, "02:00:00").to_owned());
    args
}

pub fn gpu_launch_args() -> Vec<String> {
    gpu_launch(
        SlurmContext::detect(),
        gpu_partition().as_deref(),
        resolved("OPENFOLD_GPU_ACCOUNT").as_deref(),
        resolved("OPENFOLD_GPU_GRES").as_deref(),
        resolved("OPENFOLD_GPU_RESOURCES").as_deref(),
        resolved("OPENFOLD_GPU_TIME").as_deref(),
    )
}

/// The GPU partition `gpu_launch_args` would srun onto, resolved the same env-var-or-config way.
pub fn gpu_partition() -> Option<String> {
    resolved("OPENFOLD_GPU_PARTITION")
}

pub fn database_url() -> String {
    // DATABASE_URL used to win here. It names nothing the install writes, so it is no longer a
    // source -- but README shipped it, and silently moving someone's database is the worst way to
    // say so. VIZFOLD_DB takes a full sqlite: URL, so it covers every use.
    if std::env::var("DATABASE_URL").is_ok_and(|u| !u.is_empty()) {
        eprintln!("warning: DATABASE_URL is ignored; set VIZFOLD_DB instead");
    }
    if let Some(db) = resolved("VIZFOLD_DB") {
        return if db.starts_with("sqlite:") {
            db
        } else {
            format!("sqlite://{db}?mode=rwc")
        };
    }
    if let Some(p) = resolved("OPENFOLD_PREFIX") {
        return format!("sqlite://{p}/vizfold.db?mode=rwc");
    }
    let dh = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| format!("{}/.local/share", home_dir()));
    format!("sqlite://{dh}/vizfold/vizfold.db?mode=rwc")
}

/// File behind `database_url()`, when it is a file-backed sqlite URL.
pub fn database_path() -> Option<PathBuf> {
    let url = database_url();
    let path = url.strip_prefix("sqlite://")?.split('?').next()?;
    (!path.is_empty() && path != ":memory:").then(|| PathBuf::from(path))
}

/// Repository root for the local development layout: this crate is `<root>/cli`, so the root is one
/// level up. Baked in at build time, so for a released binary it names the machine that built it --
/// use it only when it is actually present, else the checkout `vizfold install` clones.
fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .filter(|root| root.join("backends/openfold/install/install.sh").is_file())
        .map_or_else(default_src, Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::{SlurmContext, env_base, env_dir, gpu_launch, non_empty};

    /// Compiled in from the build machine, so it has to name a checkout that is actually here --
    /// a released binary otherwise reports its CI workspace as the install root.
    #[test]
    fn repository_root_names_a_real_checkout() {
        let root = super::repository_root();
        assert!(
            root.join("backends/openfold/install/install.sh").is_file(),
            "repository_root() must be a checkout, got {}",
            root.display()
        );
    }

    /// The fixed key set writes "" for every name the install did not settle, so a consumer must
    /// not tell those apart from a missing key.
    #[test]
    fn an_empty_config_value_reads_as_unset() {
        assert_eq!(non_empty(Some("")), None);
        assert_eq!(non_empty(None), None);
        assert_eq!(non_empty(Some("gpu:a100:1")), Some("gpu:a100:1".to_owned()));
    }

    /// Every environment is `<base>/vizfold-<backend>` — one directory, a fixed name each. Keeps
    /// the Rust side in step with `vizfold::env` in lib/config.sh.
    #[test]
    fn every_env_is_a_fixed_name_under_one_base() {
        for backend in ["openfold", "esmfold", "workbench"] {
            assert_eq!(
                env_dir(backend),
                env_base().join(format!("vizfold-{backend}"))
            );
        }
        assert_eq!(env_base().file_name().unwrap(), "envs");
    }

    // (name, context, partition, account, gres, resources, time, expected args)
    #[test]
    #[rustfmt::skip]
    fn gpu_launch_cases() {
        let defaults = vec!["srun", "-p", "gpu", "--gres=gpu:1", "--cpus-per-task=8", "--mem=32G", "-t", "02:00:00"];
        let cases = [
            ("in_step_runs_bare", SlurmContext::InStep, Some("gpuA100x4"), None, None, None, None, vec![]),
            ("in_allocation_runs_a_plain_step", SlurmContext::InAllocation, Some("gpuA100x4"), None, None, None, None, vec!["srun", "--ntasks=1"]),
            ("no_partition_runs_bare", SlurmContext::None, None, Some("acct"), None, None, None, vec![]),
            ("empty_partition_runs_bare", SlurmContext::None, Some(""), None, None, None, None, vec![]),
            ("resources_word_split_into_separate_arguments", SlurmContext::None, Some("gpuA100x4"), Some("bbkg-delta-gpu"), Some("gpu:a100:1"), Some("--cpus-per-task=8 --mem=32G"), Some("04:00:00"),
                vec!["srun", "-A", "bbkg-delta-gpu", "-p", "gpuA100x4", "--gres=gpu:a100:1", "--cpus-per-task=8", "--mem=32G", "-t", "04:00:00"]),
            ("none_gres_resources_and_time_fall_back_to_defaults", SlurmContext::None, Some("gpu"), None, None, None, None, defaults.clone()),
            ("empty_gres_and_resources_fall_back_to_the_same_defaults_as_none", SlurmContext::None, Some("gpu"), None, Some(""), Some(""), None, defaults.clone()),
            ("empty_time_falls_back_to_default", SlurmContext::None, Some("gpu"), None, None, None, Some(""), defaults),
        ];

        for (name, context, partition, account, gres, resources, time, want) in cases {
            assert_eq!(
                gpu_launch(context, partition, account, gres, resources, time),
                want,
                "case: {name}"
            );
        }
    }
}
