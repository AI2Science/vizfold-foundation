pub const DEFAULT_DATABASE_URL: &str = "sqlite://data/vizfold.db?mode=rwc";

pub fn database_url() -> String {
    std::env::var("DATABASE_URL").unwrap_or_else(|_| DEFAULT_DATABASE_URL.to_owned())
}
