use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Runs::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Runs::Id)
                            .integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(Runs::JobName).string().not_null())
                    .col(ColumnDef::new(Runs::InputText).text().not_null())
                    .col(ColumnDef::new(Runs::Status).string().not_null())
                    .col(ColumnDef::new(Runs::OutputJson).text())
                    .col(ColumnDef::new(Runs::ModelBackendId).integer().not_null())
                    .col(
                        ColumnDef::new(Runs::CreatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .col(
                        ColumnDef::new(Runs::UpdatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_runs_model_backend_id")
                            .from(Runs::Table, Runs::ModelBackendId)
                            .to(ModelBackends::Table, ModelBackends::Id)
                            .on_delete(ForeignKeyAction::Restrict)
                            .on_update(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Runs::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum Runs {
    Table,
    Id,
    JobName,
    InputText,
    Status,
    OutputJson,
    ModelBackendId,
    CreatedAt,
    UpdatedAt,
}

#[derive(DeriveIden)]
enum ModelBackends {
    Table,
    Id,
}
