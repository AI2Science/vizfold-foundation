use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(ArtifactTypes::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(ArtifactTypes::Id)
                            .integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(
                        ColumnDef::new(ArtifactTypes::Slug)
                            .string()
                            .not_null()
                            .unique_key(),
                    )
                    .col(ColumnDef::new(ArtifactTypes::Label).string().not_null())
                    .col(
                        ColumnDef::new(ArtifactTypes::DefaultFormat)
                            .string()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ArtifactTypes::DisplayMode)
                            .string()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ArtifactTypes::ViewerKind)
                            .string()
                            .not_null(),
                    )
                    .col(ColumnDef::new(ArtifactTypes::Description).text().not_null())
                    .col(
                        ColumnDef::new(ArtifactTypes::MetadataSchemaJson)
                            .text()
                            .not_null()
                            .default("{}"),
                    )
                    .col(
                        ColumnDef::new(ArtifactTypes::CreatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .col(
                        ColumnDef::new(ArtifactTypes::UpdatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .to_owned(),
            )
            .await?;

        // Artifacts are not yet registered in production. Rebuilding the table keeps the
        // migration small and produces a non-null foreign key on SQLite as well.
        manager
            .drop_table(Table::drop().table(Artifacts::Table).to_owned())
            .await?;
        create_artifacts_table(manager).await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Artifacts::Table).to_owned())
            .await?;
        create_legacy_artifacts_table(manager).await?;
        manager
            .drop_table(Table::drop().table(ArtifactTypes::Table).to_owned())
            .await
    }
}

async fn create_artifacts_table(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(Artifacts::Table)
                .col(
                    ColumnDef::new(Artifacts::Id)
                        .integer()
                        .not_null()
                        .auto_increment()
                        .primary_key(),
                )
                .col(ColumnDef::new(Artifacts::RunId).integer().not_null())
                .col(
                    ColumnDef::new(Artifacts::ArtifactTypeId)
                        .integer()
                        .not_null(),
                )
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
                .foreign_key(
                    ForeignKey::create()
                        .name("fk_artifacts_artifact_type_id")
                        .from(Artifacts::Table, Artifacts::ArtifactTypeId)
                        .to(ArtifactTypes::Table, ArtifactTypes::Id)
                        .on_delete(ForeignKeyAction::Restrict)
                        .on_update(ForeignKeyAction::Cascade),
                )
                .to_owned(),
        )
        .await
}

async fn create_legacy_artifacts_table(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(Artifacts::Table)
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

#[derive(DeriveIden)]
enum ArtifactTypes {
    Table,
    Id,
    Slug,
    Label,
    DefaultFormat,
    DisplayMode,
    ViewerKind,
    Description,
    MetadataSchemaJson,
    CreatedAt,
    UpdatedAt,
}

#[derive(DeriveIden)]
enum Artifacts {
    Table,
    Id,
    RunId,
    ArtifactType,
    ArtifactTypeId,
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
