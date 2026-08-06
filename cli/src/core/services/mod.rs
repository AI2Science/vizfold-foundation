pub mod artifact_types;
pub mod artifacts;
pub mod execution_targets;
pub mod model_backends;
pub mod model_invocation_profiles;
pub mod run_artifacts;
pub mod run_execution;
pub mod runs;
pub(crate) mod validation;

use sea_orm::{DatabaseConnection, DbErr, EntityTrait};

/// The catalog rows a run is assembled from. Every service that takes an id has to refuse the one
/// that names nothing, and said so in its own words until this existed.
macro_rules! require_row {
    ($name:ident, $entity:path, $what:literal) => {
        pub(crate) async fn $name(
            db: &DatabaseConnection,
            id: i32,
        ) -> Result<<$entity as EntityTrait>::Model, DbErr> {
            <$entity>::find_by_id(id)
                .one(db)
                .await?
                .ok_or_else(|| DbErr::Custom(concat!($what, " does not exist").into()))
        }
    };
}

require_row!(
    require_backend,
    crate::core::entities::model_backends::Entity,
    "model backend"
);
require_row!(
    require_target,
    crate::core::entities::execution_targets::Entity,
    "execution target"
);
require_row!(
    require_profile,
    crate::core::entities::model_invocation_profiles::Entity,
    "model invocation profile"
);
