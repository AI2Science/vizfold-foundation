use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq)]
#[sea_orm(table_name = "runs")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub job_name: String,
    pub input_text: String,
    pub status: String,
    pub output_json: Option<String>,
    pub model_backend_id: i32,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::model_backends::Entity",
        from = "Column::ModelBackendId",
        to = "super::model_backends::Column::Id",
        on_update = "Cascade",
        on_delete = "Restrict"
    )]
    ModelBackend,
    #[sea_orm(has_many = "super::artifacts::Entity")]
    Artifacts,
}

impl Related<super::model_backends::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ModelBackend.def()
    }
}

impl Related<super::artifacts::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Artifacts.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
