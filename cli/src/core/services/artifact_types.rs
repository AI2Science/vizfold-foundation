use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, Set,
};

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

pub async fn get_artifact_type_by_slug(
    db: &DatabaseConnection,
    slug: &str,
) -> Result<Option<artifact_types::Model>, DbErr> {
    artifact_types::Entity::find()
        .filter(artifact_types::Column::Slug.eq(slug))
        .one(db)
        .await
}
