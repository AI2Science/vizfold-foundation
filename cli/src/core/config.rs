use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The install-time config `config::save` writes: one flat JSON map, the source of every resolved path.
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

/// Mirrors `VIZFOLD_CONFIG_KEYS` in `lib/config.sh`; `tests/vocabulary.sh` fails if the two differ.
pub const CONFIG_KEYS: &[&str] = &[
    "ESMFOLD_ENV_PREFIX",
    "OPENFOLD_ACCOUNT",
    "OPENFOLD_AF2_ROOT",
    "OPENFOLD_DATA_DIR",
    "OPENFOLD_DRIVER_CUDA",
    "OPENFOLD_ENV_PREFIX",
    "OPENFOLD_EXAMPLE",
    "OPENFOLD_GPU_ACCOUNT",
    "OPENFOLD_GPU_GRES",
    "OPENFOLD_GPU_PARTITION",
    "OPENFOLD_GPU_RESOURCES",
    "OPENFOLD_GPU_TIME",
    "OPENFOLD_HOME",
    "OPENFOLD_MAX_CUDA",
    "OPENFOLD_PARTITION",
    "OPENFOLD_PREFIX",
    "OPENFOLD_SITE",
    "VIZFOLD_DB",
    "VIZFOLD_ENV_BASE",
];

pub fn config_keys() -> Vec<String> {
    vizfold_config().keys().cloned().collect()
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

/// Empty is unset: the key set is fixed, so an unsettled name is present-but-empty and must fall through.
fn non_empty(value: Option<&str>) -> Option<String> {
    value.filter(|v| !v.is_empty()).map(str::to_owned)
}

/// inline env var of the same name > vizfold.json entry > None.
pub fn resolved(key: &str) -> Option<String> {
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

/// The file whose presence makes a directory a vizfold checkout.
pub const INSTALLER: &str = "backends/openfold/install/install.sh";

/// Checkout holding `INSTALLER`: `OPENFOLD_HOME`, else the default clone location.
pub fn vizfold_src() -> PathBuf {
    resolved("OPENFOLD_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(default_src)
}

/// Where `vizfold install` clones -- the only checkout `vizfold uninstall` may delete.
pub fn default_src() -> PathBuf {
    PathBuf::from(format!("{}/vizfold-src", home_dir()))
}

/// The one root every OpenFold data path resolves under, weights included. Mirrors `setup::config`.
pub fn data_dir() -> PathBuf {
    resolved("OPENFOLD_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_data_dir(&prefix()))
}

/// The install's own default for it, kept separate so the literal can be pinned.
fn default_data_dir(prefix: &Path) -> PathBuf {
    prefix.join("openfold/data")
}

/// The one directory holding every environment. Mirrors `vizfold::env_base`.
pub fn env_base() -> PathBuf {
    resolved("VIZFOLD_ENV_BASE")
        .map(PathBuf::from)
        .unwrap_or_else(|| prefix().join("envs"))
}

/// `<env base>/vizfold-<backend>`: a fixed name per backend, so nothing has to be told where one is.
pub fn env_dir(name: &str) -> PathBuf {
    env_base().join(format!("vizfold-{name}"))
}

/// OpenFold's env. The install records it; the fallback covers a config where only ESMFold was installed.
pub fn openfold_env_prefix() -> PathBuf {
    resolved("OPENFOLD_ENV_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| env_dir("openfold"))
}

/// Environment prefix for the ESMFold backend, same story as `openfold_env_prefix`.
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

/// Mirrors `vizfold::prefix`, fallback included -- otherwise `status` describes a directory no install uses.
pub fn prefix() -> PathBuf {
    resolved("OPENFOLD_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_prefix(&home_dir()))
}

/// `vizfold::prefix`'s own default, kept separate so the literal can be pinned.
fn default_prefix(home: &str) -> PathBuf {
    PathBuf::from(format!("{home}/openfold"))
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

/// SLURM launch prefix, mirroring `setup::fold_vars`. Empty means run bare, here or on a workstation.
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
    let data_home = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| format!("{}/.local/share", home_dir()));
    database_url_from(
        resolved("VIZFOLD_DB"),
        resolved("OPENFOLD_PREFIX"),
        &data_home,
    )
}

/// VIZFOLD_DB > OPENFOLD_PREFIX > XDG data home; a `sqlite:` value passes through, a bare path gets
/// `?mode=rwc`. Callers resolve, as `gpu_launch` does.
fn database_url_from(db: Option<String>, prefix: Option<String>, data_home: &str) -> String {
    if let Some(db) = db {
        return if db.starts_with("sqlite:") {
            db
        } else {
            format!("sqlite://{db}?mode=rwc")
        };
    }
    match prefix {
        Some(prefix) => format!("sqlite://{prefix}/vizfold.db?mode=rwc"),
        None => format!("sqlite://{data_home}/vizfold/vizfold.db?mode=rwc"),
    }
}

/// File behind `database_url()`, when it is a file-backed sqlite URL.
pub fn database_path() -> Option<PathBuf> {
    let url = database_url();
    let path = url.strip_prefix("sqlite://")?.split('?').next()?;
    (!path.is_empty() && path != ":memory:").then(|| PathBuf::from(path))
}

/// The dev checkout, one level up from this crate. Baked in at build time, so use it only if it exists.
fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .filter(|root| root.join(INSTALLER).is_file())
        .map_or_else(default_src, Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::{SlurmContext, env_base, env_dir, gpu_launch, non_empty};

    /// Compiled in from the build machine, so a released binary would report its CI workspace.
    #[test]
    fn repository_root_names_a_real_checkout() {
        let root = super::repository_root();
        assert!(
            root.join(super::INSTALLER).is_file(),
            "repository_root() must be a checkout, got {}",
            root.display()
        );
    }

    /// The fixed key set writes "" for what the install did not settle; that must read as missing.
    #[test]
    fn an_empty_config_value_reads_as_unset() {
        assert_eq!(non_empty(Some("")), None);
        assert_eq!(non_empty(None), None);
        assert_eq!(non_empty(Some("gpu:a100:1")), Some("gpu:a100:1".to_owned()));
    }

    /// Keeps `env_dir` in step with `vizfold::env` in lib/config.sh.
    #[test]
    fn every_env_is_a_fixed_name_under_one_base() {
        assert_eq!(env_dir("openfold").file_name().unwrap(), "vizfold-openfold");
        assert_eq!(env_dir("openfold").parent().unwrap(), env_base());
        assert_eq!(env_base().file_name().unwrap(), "envs");
    }

    /// Mirrors `setup::config`'s `DATA=${OPENFOLD_DATA_DIR:-$STATE/data}`. It moved once already, and
    /// drift makes `status` and `uninstall` name a directory no install uses.
    #[test]
    fn the_data_dir_default_sits_under_the_backend_state_dir() {
        assert_eq!(
            super::default_data_dir(std::path::Path::new("/work/p")),
            std::path::PathBuf::from("/work/p/openfold/data")
        );
        assert_eq!(
            super::default_prefix("/home/me"),
            std::path::PathBuf::from("/home/me/openfold")
        );
    }

    /// The three sources in order. A configured database being ignored does not fail loudly -- it
    /// silently opens a different file, and the run history looks gone.
    #[test]
    #[rustfmt::skip]
    fn the_database_url_prefers_vizfold_db_then_the_prefix() {
        let url = |db: Option<&str>, prefix: Option<&str>| {
            super::database_url_from(db.map(str::to_owned), prefix.map(str::to_owned), "/xdg")
        };

        assert_eq!(url(Some("/db/mine.db"), Some("/p")), "sqlite:///db/mine.db?mode=rwc");
        assert_eq!(url(Some("sqlite://x?mode=ro"), Some("/p")), "sqlite://x?mode=ro");
        assert_eq!(url(None, Some("/p")), "sqlite:///p/vizfold.db?mode=rwc");
        assert_eq!(url(None, None), "sqlite:///xdg/vizfold/vizfold.db?mode=rwc");
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
