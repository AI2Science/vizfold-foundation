use serde_json::{Map, Value};
use std::path::PathBuf;
use std::sync::OnceLock;

pub const DEFAULT_DATABASE_URL: &str = "sqlite://data/vizfold.db?mode=rwc";

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

/// Repository root for the local development layout. DEV FALLBACK ONLY: relies on
/// the executor crate being nested under the repository root, not suitable for an
/// installed binary (where `openfold_home()` resolves from config instead).
pub fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("executor manifest should be nested under the repository root")
        .to_path_buf()
}
