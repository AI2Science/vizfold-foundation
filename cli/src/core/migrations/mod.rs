mod m20260723_000001_create_schema;

pub use sea_orm_migration::prelude::MigratorTrait;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn sea_orm_migration::MigrationTrait>> {
        vec![Box::new(m20260723_000001_create_schema::Migration)]
    }
}
