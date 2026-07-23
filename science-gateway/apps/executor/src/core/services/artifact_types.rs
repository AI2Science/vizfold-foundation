use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{entities::artifact_types, repositories};

use super::validation::require_json_object;

#[derive(Clone, Debug)]
pub struct RegisterArtifactTypeInput {
    pub slug: String,
    pub label: String,
    pub default_format: String,
    pub display_mode: String,
    pub viewer_kind: String,
    pub description: String,
    pub metadata_schema_json: String,
}

pub async fn register_artifact_type(
    db: &DatabaseConnection,
    input: RegisterArtifactTypeInput,
) -> Result<artifact_types::Model, DbErr> {
    require_json_object("artifact type metadata_schema", &input.metadata_schema_json)?;
    repositories::artifact_types::create(db, input).await
}

pub async fn list_artifact_types(
    db: &DatabaseConnection,
) -> Result<Vec<artifact_types::Model>, DbErr> {
    repositories::artifact_types::list(db).await
}

pub async fn get_artifact_type_by_slug(
    db: &DatabaseConnection,
    slug: &str,
) -> Result<Option<artifact_types::Model>, DbErr> {
    repositories::artifact_types::find_by_slug(db, slug).await
}
