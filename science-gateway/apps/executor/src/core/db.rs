use std::path::Path;

use sea_orm::{ConnectOptions, ConnectionTrait, Database, DatabaseConnection, DbErr, Statement};

use crate::core::{config, migrations::MigratorTrait};

pub async fn connect_and_migrate() -> Result<DatabaseConnection, DbErr> {
    let database_url = config::database_url();

    ensure_sqlite_parent_dir(&database_url)?;

    let mut options = ConnectOptions::new(database_url);
    options.sqlx_logging(false);

    let db = Database::connect(options).await?;

    // SQLite needs foreign keys explicitly enabled per connection.
    db.execute(Statement::from_string(
        db.get_database_backend(),
        "PRAGMA foreign_keys = ON".to_owned(),
    ))
    .await?;

    crate::core::migrations::Migrator::up(&db, None).await?;

    Ok(db)
}

fn ensure_sqlite_parent_dir(database_url: &str) -> Result<(), DbErr> {
    if !database_url.starts_with("sqlite://") {
        return Ok(());
    }

    let path_part = database_url
        .trim_start_matches("sqlite://")
        .split('?')
        .next()
        .unwrap_or_default();

    if path_part == ":memory:" || path_part.is_empty() {
        return Ok(());
    }

    let path = Path::new(path_part);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| DbErr::Custom(error.to_string()))?;
    }

    Ok(())
}
