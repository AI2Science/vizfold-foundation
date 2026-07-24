#![cfg(test)]

use std::{
    env, fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use serde_json::json;

use crate::core::commands::CommandSpec;

static NEXT_TEST_DIR: AtomicUsize = AtomicUsize::new(0);

/// Local-filesystem fixture shared by OpenFold planning/preflight/execution tests: a working
/// directory with a test script, a FASTA directory with one matching record, and empty data /
/// output-location / alignment directories.
pub(crate) struct TestLayout {
    pub root: PathBuf,
    pub working_dir: PathBuf,
    pub fasta_dir: PathBuf,
    pub data_dir: PathBuf,
    pub alignment_dir: PathBuf,
    pub output_location: PathBuf,
}

impl TestLayout {
    /// `fasta_header` is written verbatim after `>` on the FASTA record's header line, so callers
    /// control both the tag a test's run `input_id` must match and any trailing header text.
    pub fn new(fasta_header: &str) -> Self {
        let root = env::temp_dir().join(format!(
            "executor-test-layout-{}-{}",
            std::process::id(),
            NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let working_dir = root.join("workspace");
        let fasta_dir = root.join("fasta");
        let data_dir = root.join("data");
        let alignment_dir = root.join("alignments");
        let output_location = root.join("outputs");

        fs::create_dir_all(&working_dir).expect("working directory should be created");
        fs::create_dir_all(&fasta_dir).expect("fasta directory should be created");
        fs::create_dir_all(&data_dir).expect("data directory should be created");
        fs::create_dir_all(&output_location).expect("output location should be created");
        fs::write(working_dir.join("run_openfold.py"), "# test script")
            .expect("script should be created");
        fs::write(
            fasta_dir.join("input.fasta"),
            format!(">{fasta_header}\nMSTNPKPQRITF\n"),
        )
        .expect("matching FASTA should be created");

        Self {
            root,
            working_dir,
            fasta_dir,
            data_dir,
            alignment_dir,
            output_location,
        }
    }

    pub fn command(&self) -> CommandSpec {
        CommandSpec {
            program: "python3".into(),
            args: vec!["-u".into(), "run_openfold.py".into()],
            current_dir: Some(self.working_dir.clone()),
            ..Default::default()
        }
    }

    pub fn execution_parameters(&self) -> serde_json::Value {
        json!({
            "fasta_dir": self.fasta_dir,
            "data_dir": self.data_dir,
        })
    }
}

impl Drop for TestLayout {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
