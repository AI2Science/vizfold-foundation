use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq)]
#[sea_orm(table_name = "model_invocation_profiles")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub model_backend_id: i32,
    pub execution_target_id: i32,
    pub invocation_kind: String,
    pub config_json: String,
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
    #[sea_orm(
        belongs_to = "super::execution_targets::Entity",
        from = "Column::ExecutionTargetId",
        to = "super::execution_targets::Column::Id",
        on_update = "Cascade",
        on_delete = "Restrict"
    )]
    ExecutionTarget,
    #[sea_orm(has_many = "super::runs::Entity")]
    Runs,
}

impl Related<super::model_backends::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ModelBackend.def()
    }
}

impl Related<super::execution_targets::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ExecutionTarget.def()
    }
}

impl Related<super::runs::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Runs.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
