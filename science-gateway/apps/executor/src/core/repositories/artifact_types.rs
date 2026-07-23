use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, Set,
};

use crate::core::{entities::artifact_types, services::artifact_types::RegisterArtifactTypeInput};

pub async fn list(db: &DatabaseConnection) -> Result<Vec<artifact_types::Model>, DbErr> {
    artifact_types::Entity::find().all(db).await
}

pub async fn find_by_slug(
    db: &DatabaseConnection,
    slug: &str,
) -> Result<Option<artifact_types::Model>, DbErr> {
    artifact_types::Entity::find()
        .filter(artifact_types::Column::Slug.eq(slug))
        .one(db)
        .await
}

pub async fn create(
    db: &DatabaseConnection,
    input: RegisterArtifactTypeInput,
) -> Result<artifact_types::Model, DbErr> {
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
