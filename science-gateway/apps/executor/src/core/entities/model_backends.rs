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

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::runs::Entity")]
    Runs,
    #[sea_orm(has_many = "super::model_invocation_profiles::Entity")]
    ModelInvocationProfiles,
}

impl Related<super::runs::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Runs.def()
    }
}

impl Related<super::model_invocation_profiles::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ModelInvocationProfiles.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
