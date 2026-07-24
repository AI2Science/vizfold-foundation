use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq)]
#[sea_orm(table_name = "runs")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub model_backend_id: i32,
    pub execution_target_id: i32,
    pub invocation_profile_id: i32,
    pub status: String,
    pub input_id: String,
    pub input_sequence: String,
    pub model_parameters_json: String,
    pub execution_parameters_json: String,
    pub provenance_json: Option<String>,
    pub submitted_at: DateTimeUtc,
    pub started_at: Option<DateTimeUtc>,
    pub completed_at: Option<DateTimeUtc>,
    pub error_message: Option<String>,
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
    #[sea_orm(
        belongs_to = "super::model_invocation_profiles::Entity",
        from = "Column::InvocationProfileId",
        to = "super::model_invocation_profiles::Column::Id",
        on_update = "Cascade",
        on_delete = "Restrict"
    )]
    ModelInvocationProfile,
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

impl Related<super::execution_targets::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ExecutionTarget.def()
    }
}

impl Related<super::model_invocation_profiles::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ModelInvocationProfile.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
