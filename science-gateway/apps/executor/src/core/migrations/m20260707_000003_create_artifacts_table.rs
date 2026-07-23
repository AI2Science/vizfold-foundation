use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Artifacts::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Artifacts::Id)
                            .integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(Artifacts::RunId).integer().not_null())
                    .col(ColumnDef::new(Artifacts::ArtifactType).string().not_null())
                    .col(ColumnDef::new(Artifacts::Format).string().not_null())
                    .col(ColumnDef::new(Artifacts::StorageUri).text().not_null())
                    .col(ColumnDef::new(Artifacts::MetadataJson).text().not_null())
                    .col(
                        ColumnDef::new(Artifacts::CreatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_artifacts_run_id")
                            .from(Artifacts::Table, Artifacts::RunId)
                            .to(Runs::Table, Runs::Id)
                            .on_delete(ForeignKeyAction::Cascade)
                            .on_update(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Artifacts::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum Artifacts {
    Table,
    Id,
    RunId,
    ArtifactType,
    Format,
    StorageUri,
    MetadataJson,
    CreatedAt,
}

#[derive(DeriveIden)]
enum Runs {
    Table,
    Id,
}
