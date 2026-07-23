use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq)]
#[sea_orm(table_name = "artifact_types")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    #[sea_orm(unique)]
    pub slug: String,
    pub label: String,
    pub default_format: String,
    pub display_mode: String,
    pub viewer_kind: String,
    pub description: String,
    pub metadata_schema_json: String,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::artifacts::Entity")]
    Artifacts,
}

impl Related<super::artifacts::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Artifacts.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
