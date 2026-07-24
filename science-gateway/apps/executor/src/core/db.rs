use std::path::Path;

use sea_orm::{ConnectOptions, ConnectionTrait, Database, DatabaseConnection, DbErr, Statement};

use crate::core::{config, migrations::MigratorTrait, seed, services::model_backends};

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
    crate::core::migrations::Migrator::up(db, None)
        .await
        .map_err(name_pre_baseline_database)
}

/// The migrator's own error for a database predating the migration collapse ("Migration file of
/// version '...' is missing...") names no remedy. Point at the actual fix: the baseline
/// migration recreates the schema from scratch once the stale file is gone.
fn name_pre_baseline_database(error: DbErr) -> DbErr {
    match &error {
        DbErr::Custom(message) if message.contains("Migration file of version") => {
            let path = config::database_path().map_or_else(
                || "the executor database".to_owned(),
                |path| path.display().to_string(),
            );
            DbErr::Custom(format!(
                "this executor database predates the 2026-07-23 baseline schema; delete {path} and re-run"
            ))
        }
        _ => error,
    }
}

pub struct ExecutionCore {
    db: DatabaseConnection,
}

impl ExecutionCore {
    pub async fn bootstrap() -> Result<Self, DbErr> {
        let db = connect_and_migrate().await?;
        seed::seed_defaults(&db).await?;
        Ok(Self { db })
    }

    pub async fn check_readiness(&self) -> Result<(), DbErr> {
        let _ = model_backends::list_model_backends(&self.db).await?;
        Ok(())
    }

    pub fn db(&self) -> &DatabaseConnection {
        &self.db
    }
}

#[cfg(test)]
mod tests {
    use sea_orm::{ActiveModelTrait, ActiveValue::Set};
    use sea_orm_migration::seaql_migrations;

    use super::*;

    #[tokio::test]
    async fn pre_baseline_database_gets_an_actionable_error() -> Result<(), DbErr> {
        let db = Database::connect("sqlite::memory:").await?;
        migrate_database(&db).await?;

        // Simulate a database migrated before the collapse: a version recorded as applied that
        // no longer has a corresponding migration file in this binary.
        seaql_migrations::ActiveModel {
            version: Set("m20200101_000001_stale".to_owned()),
            applied_at: Set(0),
        }
        .insert(&db)
        .await?;

        let error = migrate_database(&db)
            .await
            .expect_err("stale migration record should error");
        let message = error.to_string();
        assert!(message.contains("predates the 2026-07-23 baseline schema"));
        assert!(message.contains("delete"));
        Ok(())
    }
}
