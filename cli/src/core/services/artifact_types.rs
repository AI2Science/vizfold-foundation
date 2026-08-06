use sea_orm::{ActiveModelTrait, DatabaseConnection, DbErr, EntityTrait, Set};

use crate::core::entities::artifact_types;

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
    artifact_types::ActiveModel {
        slug: Set(input.slug),
        label: Set(input.label),
        default_format: Set(input.default_format),
        display_mode: Set(input.display_mode),
        viewer_kind: Set(input.viewer_kind),
        description: Set(input.description),
        metadata_schema_json: Set(input.metadata_schema_json),
        ..Default::default()
    }
    .insert(db)
    .await
}

pub async fn list_artifact_types(
    db: &DatabaseConnection,
) -> Result<Vec<artifact_types::Model>, DbErr> {
    artifact_types::Entity::find().all(db).await
}

/// Bring a catalog row to what the code now says the kind is. The slug identifies the kind; the
/// label, format and hints are the current definition of it.
pub async fn update_artifact_type(
    db: &DatabaseConnection,
    id: i32,
    input: RegisterArtifactTypeInput,
) -> Result<artifact_types::Model, DbErr> {
    require_json_object("artifact type metadata_schema", &input.metadata_schema_json)?;
    let mut row: artifact_types::ActiveModel = artifact_types::Entity::find_by_id(id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("artifact type {id} does not exist")))?
        .into();
    row.label = Set(input.label);
    row.default_format = Set(input.default_format);
    row.display_mode = Set(input.display_mode);
    row.viewer_kind = Set(input.viewer_kind);
    row.description = Set(input.description);
    row.metadata_schema_json = Set(input.metadata_schema_json);
    row.update(db).await
}
