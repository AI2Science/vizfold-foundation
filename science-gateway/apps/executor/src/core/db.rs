use std::path::Path;

use sea_orm::{ConnectOptions, ConnectionTrait, Database, DatabaseConnection, DbErr, Statement};

use crate::core::{config, migrations::MigratorTrait};

pub async fn connect_and_migrate() -> Result<DatabaseConnection, DbErr> {
    if let Some(parent) = config::database_path().as_deref().and_then(Path::parent) {
        std::fs::create_dir_all(parent).map_err(|error| DbErr::Custom(error.to_string()))?;
    }

    let mut options = ConnectOptions::new(config::database_url());
    options.sqlx_logging(false);

    let db = Database::connect(options).await?;

    // SQLite needs foreign keys explicitly enabled per connection.
    db.execute(Statement::from_string(
        db.get_database_backend(),
        "PRAGMA foreign_keys = ON".to_owned(),
    ))
    .await?;

    migrate_database(&db).await?;

    Ok(db)
}

pub async fn migrate_database(db: &DatabaseConnection) -> Result<(), DbErr> {
    crate::core::migrations::Migrator::up(db, None).await
}
