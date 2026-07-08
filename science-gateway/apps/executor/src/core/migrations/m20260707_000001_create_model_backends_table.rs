use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(ModelBackends::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(ModelBackends::Id)
                            .integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(
                        ColumnDef::new(ModelBackends::Slug)
                            .string()
                            .not_null()
                            .unique_key(),
                    )
                    .col(ColumnDef::new(ModelBackends::Label).string().not_null())
                    .col(ColumnDef::new(ModelBackends::Summary).text().not_null())
                    .col(
                        ColumnDef::new(ModelBackends::CapabilitiesJson)
                            .text()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ModelBackends::CreatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .col(
                        ColumnDef::new(ModelBackends::UpdatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(ModelBackends::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum ModelBackends {
    Table,
    Id,
    Slug,
    Label,
    Summary,
    CapabilitiesJson,
    CreatedAt,
    UpdatedAt,
}
