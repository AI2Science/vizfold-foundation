use serde_json::{Map, Value};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Path to the install-time config written by `install/config.sh` (`config::save`).
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

/// inline env var of the same name > vizfold.json entry > None.
fn resolved(key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(key)
        && !v.is_empty()
    {
        return Some(v);
    }
    vizfold_config()
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
}

pub fn openfold_home() -> PathBuf {
    resolved("OPENFOLD_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(repository_root)
}

/// Repo checkout holding `install/init.sh` (what `vizfold install` runs, cloning it if absent).
/// `VIZFOLD_SRC` env > vizfold.json `OPENFOLD_HOME` > the default clone location (`$HOME/vizfold-src`).
pub fn vizfold_src() -> PathBuf {
    if let Ok(v) = std::env::var("VIZFOLD_SRC")
        && !v.is_empty()
    {
        return PathBuf::from(v);
    }
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

/// micromamba env prefix for local OpenFold execution (matches fold.sh's
/// `${OPENFOLD_ENV_PREFIX:-$PREFIX/mamba/envs/openfold-env}`).
pub fn openfold_env_prefix() -> PathBuf {
    resolved("OPENFOLD_ENV_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| prefix().join("mamba/envs/openfold-env"))
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

/// SLURM launch prefix for a fold, mirroring `install/setup.sh:212`. Empty means run bare --
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
    // Holds several space-separated flags and must split, as setup.sh:212 relies on word splitting.
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
    if let Ok(u) = std::env::var("DATABASE_URL")
        && !u.is_empty()
    {
        return u;
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

/// Repository root for the local development layout. DEV FALLBACK ONLY: relies on
/// the executor crate being nested under the repository root, not suitable for an
/// installed binary (where `openfold_home()` resolves from config instead).
fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("executor manifest should be nested under the repository root")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::{SlurmContext, gpu_launch};

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
