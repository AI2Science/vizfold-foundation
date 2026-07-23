use std::path::PathBuf;

pub const DEFAULT_DATABASE_URL: &str = "sqlite://data/vizfold.db?mode=rwc";

pub fn database_url() -> String {
    std::env::var("DATABASE_URL").unwrap_or_else(|_| DEFAULT_DATABASE_URL.to_owned())
}

/// Returns the repository root for the current MVP local development layout.
///
/// This intentionally relies on the executor crate being nested under the
/// repository root and is not suitable for a general installed binary.
pub fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("executor manifest should be nested under the repository root")
        .to_path_buf()
}
