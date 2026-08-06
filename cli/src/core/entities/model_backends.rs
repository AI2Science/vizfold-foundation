use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq)]
#[sea_orm(table_name = "model_backends")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    #[sea_orm(unique)]
    pub slug: String,
    pub label: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub artifact_capabilities_json: String,
    pub parameter_schema_json: String,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter)]
pub enum Relation {}

impl RelationTrait for Relation {
    fn def(&self) -> RelationDef {
        // The foreign keys live in the migration, which is what enforces them; nothing in the
        // crate joins through the ORM, so there is no relation to hand back.
        panic!("no ORM relations are declared for {}", Entity.table_name())
    }
}

impl ActiveModelBehavior for ActiveModel {}
