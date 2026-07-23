use std::env;

use executor::core::{
    config, db,
    entities::{execution_targets, model_backends, model_invocation_profiles, runs},
    output_locations::resolve_output_location,
    repositories, seed,
    services::openfold_artifacts::register_known_openfold_artifacts,
};
use sea_orm::{ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), DbErr> {
    let paths = DemoPaths::from_environment();
    let database_url = config::database_url();
    let db = db::connect_and_migrate().await?;
    seed::seed_defaults(&db).await?;

    let profile = find_demo_invocation_profile(&db).await?;
    let run = find_or_create_demo_run(&db, &profile, &paths).await?;
    let output_dir = resolve_output_location(&profile, &run)?;
    let artifacts = register_known_openfold_artifacts(&db, run.id).await?;

    println!("== OpenFold demo artifact registration ==");
    println!("database: {database_url}");
    println!("run id used: {}", run.id);
    println!("output_dir scanned/registered: {}", output_dir.display());
    println!(
        "attn_map_dir scanned/registered: {}",
        output_dir.join("attention").display()
    );
    println!("artifacts currently registered: {}", artifacts.len());
    Ok(())
}

struct DemoPaths {
    input_id: String,
}

impl DemoPaths {
    fn from_environment() -> Self {
        let input_id = env_or_value("VIZFOLD_OPENFOLD_INPUT_ID", "6KWC_1");
        Self { input_id }
    }
}

fn env_or_value(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.into())
}

async fn find_or_create_demo_run(
    db: &DatabaseConnection,
    profile: &model_invocation_profiles::Model,
    paths: &DemoPaths,
) -> Result<runs::Model, DbErr> {
    // Runs have numeric IDs and no external run-key column. This example therefore
    // reuses a placeholder by input_id and invocation profile.
    if let Some(run) = runs::Entity::find()
        .filter(runs::Column::InputId.eq(&paths.input_id))
        .filter(runs::Column::InvocationProfileId.eq(profile.id))
        .all(db)
        .await?
        .into_iter()
        .next()
    {
        return Ok(run);
    }

    let backend = model_backends::Entity::find()
        .filter(model_backends::Column::Slug.eq("openfold"))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded OpenFold backend is missing".into()))?;
    let target = execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq("local-openfold"))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded local target is missing".into()))?;
    // This record is solely an artifact-indexing placeholder; it is never submitted
    // to the OpenFold planner or executor.
    repositories::runs::create(
        db,
        executor::core::services::runs::SubmitRunInput {
            model_backend_id: backend.id,
            execution_target_id: target.id,
            invocation_profile_id: profile.id,
            status: "artifact_indexed".into(),
            input_id: paths.input_id.clone(),
            input_sequence: "artifact-indexing-placeholder".into(),
            model_parameters_json: json!({}).to_string(),
            execution_parameters_json: json!({}).to_string(),
        },
    )
    .await
}

async fn find_demo_invocation_profile(
    db: &DatabaseConnection,
) -> Result<model_invocation_profiles::Model, DbErr> {
    let backend = model_backends::Entity::find()
        .filter(model_backends::Column::Slug.eq("openfold"))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded OpenFold backend is missing".into()))?;
    let target = execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq("local-openfold"))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded local target is missing".into()))?;

    model_invocation_profiles::Entity::find()
        .filter(model_invocation_profiles::Column::ModelBackendId.eq(backend.id))
        .filter(model_invocation_profiles::Column::ExecutionTargetId.eq(target.id))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded invocation profile is missing".into()))
}
