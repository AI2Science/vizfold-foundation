use crate::core::services::{execution_targets, model_backends, model_invocation_profiles};
use std::collections::{BTreeMap, HashMap};

use sea_orm::{DbErr, EntityTrait};
use serde_json::json;

use crate::core::{entities::artifact_types, examples, services::runs};

/// Filesystem-only, so the dashboard can draw its dropdown without a connect and migrate.
pub(super) fn list_proteins(json: bool) -> Result<(), DbErr> {
    let found = examples::scan_default();
    if json {
        println!(
            "{}",
            serde_json::Value::Array(
                found
                    .iter()
                    .map(|example| json!({
                        "id": example.id,
                        "residues": example.residues,
                        "description": example.description,
                        "sequence": example.sequence,
                        "alignments": example.alignments,
                    }))
                    .collect()
            )
        );
        return Ok(());
    }
    if found.is_empty() {
        println!(
            "No proteins under {}. Re-run `vizfold install openfold`.",
            examples::monomer_dir().display()
        );
        return Ok(());
    }
    // ALIGNMENTS is what a chooser needs: N means this one pays for the full MSA search.
    print_table(
        &["ID", "RESIDUES", "ALIGNMENTS", "DESCRIPTION"],
        found.iter().map(|example| {
            vec![
                example.id.clone(),
                example.residues.to_string(),
                if example.alignments { "Y" } else { "N" }.to_owned(),
                example.description.clone(),
            ]
        }),
    );
    Ok(())
}

pub(super) async fn list_models(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let models = model_backends::list_model_backends(database).await?;
    print_table(
        &["ID", "SLUG", "LABEL", "VERSION"],
        models.iter().map(|model| {
            vec![
                model.id.to_string(),
                model.slug.clone(),
                model.label.clone(),
                model.version.clone().unwrap_or_else(|| "-".into()),
            ]
        }),
    );
    Ok(())
}

pub(super) async fn list_targets(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let targets = execution_targets::list_execution_targets(database).await?;
    print_table(
        &["ID", "SLUG", "TYPE", "DESCRIPTION"],
        targets.iter().map(|target| {
            vec![
                target.id.to_string(),
                target.slug.clone(),
                target.target_type.clone(),
                target.description.clone().unwrap_or_else(|| "-".into()),
            ]
        }),
    );
    Ok(())
}

pub(super) async fn list_profiles(database: &sea_orm::DatabaseConnection) -> Result<(), DbErr> {
    let profiles = model_invocation_profiles::list_model_invocation_profiles(database).await?;
    print_table(
        &["ID", "MODEL ID", "TARGET ID", "INVOCATION KIND"],
        profiles.iter().map(|profile| {
            vec![
                profile.id.to_string(),
                profile.model_backend_id.to_string(),
                profile.execution_target_id.to_string(),
                profile.invocation_kind.clone(),
            ]
        }),
    );
    Ok(())
}

pub(super) async fn list_runs(
    database: &sea_orm::DatabaseConnection,
    status: Option<&str>,
) -> Result<(), DbErr> {
    let runs = runs::list_runs(database).await?;
    print_table(
        &[
            "ID",
            "STATUS",
            "MODEL ID",
            "TARGET ID",
            "PROFILE ID",
            "INPUT ID",
            "SUBMITTED AT",
        ],
        runs.iter()
            .filter(|run| status.is_none_or(|value| run.status == value))
            .map(|run| {
                vec![
                    run.id.to_string(),
                    run.status.clone(),
                    run.model_backend_id.to_string(),
                    run.execution_target_id.to_string(),
                    run.invocation_profile_id.to_string(),
                    run.input_id.clone(),
                    run.submitted_at.to_rfc3339(),
                ]
            }),
    );
    Ok(())
}

pub(super) async fn show_run(
    database: &sea_orm::DatabaseConnection,
    run_id: i32,
) -> Result<(), DbErr> {
    let Some(result) = runs::get_run_with_artifacts(database, run_id).await? else {
        return Err(DbErr::Custom(format!("run {run_id} does not exist")));
    };
    let run = result.run;

    println!("Run {}", run.id);
    println!("status: {}", run.status);
    println!("input_id: {}", run.input_id);
    println!("model_backend_id: {}", run.model_backend_id);
    println!("execution_target_id: {}", run.execution_target_id);
    println!("invocation_profile_id: {}", run.invocation_profile_id);
    println!("submitted_at: {}", run.submitted_at.to_rfc3339());
    println!("started_at: {}", format_time(run.started_at));
    println!("completed_at: {}", format_time(run.completed_at));
    if let Some(error_message) = run.error_message {
        println!("error_message: {error_message}");
    }

    // A run's outputs are worth reading as kinds first: how many of each, then the instances.
    // One read of the catalog and one of the artifacts serves both tables.
    let kinds: HashMap<i32, String> = artifact_types::Entity::find()
        .all(database)
        .await?
        .into_iter()
        .map(|artifact_type| (artifact_type.id, artifact_type.slug))
        .collect();
    let slug_of = |artifact: &crate::core::entities::artifacts::Model| {
        kinds
            .get(&artifact.artifact_type_id)
            .cloned()
            .unwrap_or_else(|| artifact.artifact_type_id.to_string())
    };

    let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
    for artifact in &result.artifacts {
        *by_kind.entry(slug_of(artifact)).or_default() += 1;
    }
    println!("artifacts by kind:");
    print_table(
        &["KIND", "COUNT"],
        by_kind
            .iter()
            .map(|(slug, count)| vec![slug.clone(), count.to_string()]),
    );
    println!("\nartifacts:");
    print_table(
        &["ID", "KIND", "FORMAT", "STORAGE URI"],
        result.artifacts.iter().map(|artifact| {
            vec![
                artifact.id.to_string(),
                slug_of(artifact),
                artifact.format.clone(),
                artifact.storage_uri.clone(),
            ]
        }),
    );
    Ok(())
}

pub(super) fn format_time(value: Option<chrono::DateTime<chrono::Utc>>) -> String {
    value
        .map(|time| time.to_rfc3339())
        .unwrap_or_else(|| "-".into())
}

pub(super) fn print_table(headers: &[&str], rows: impl IntoIterator<Item = Vec<String>>) {
    let rows: Vec<Vec<String>> = rows.into_iter().collect();
    let mut widths: Vec<usize> = headers.iter().map(|header| header.len()).collect();
    for row in &rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(cell.len());
        }
    }

    print_row(headers.iter().copied(), &widths);
    let separator = widths
        .iter()
        .map(|width| "-".repeat(*width))
        .collect::<Vec<_>>();
    print_row(separator.iter().map(String::as_str), &widths);
    for row in rows {
        print_row(row.iter().map(String::as_str), &widths);
    }
}

pub(super) fn print_row<'a>(cells: impl IntoIterator<Item = &'a str>, widths: &[usize]) {
    let rendered = cells
        .into_iter()
        .zip(widths)
        .map(|(cell, width)| format!("{cell:<width$}", width = width))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{rendered}");
}
