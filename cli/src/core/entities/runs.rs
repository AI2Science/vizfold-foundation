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
