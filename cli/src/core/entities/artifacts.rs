use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq)]
#[sea_orm(table_name = "artifacts")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub run_id: i32,
    pub artifact_type_id: i32,
    pub format: String,
    pub storage_uri: String,
    pub metadata_json: String,
    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::runs::Entity",
        from = "Column::RunId",
        to = "super::runs::Column::Id",
        on_update = "Cascade",
        on_delete = "Cascade"
    )]
    Run,
    #[sea_orm(
        belongs_to = "super::artifact_types::Entity",
        from = "Column::ArtifactTypeId",
        to = "super::artifact_types::Column::Id",
        on_update = "Cascade",
        on_delete = "Restrict"
    )]
    ArtifactType,
}

impl Related<super::runs::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Run.def()
    }
}

impl Related<super::artifact_types::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ArtifactType.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
