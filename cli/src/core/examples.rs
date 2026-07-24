//! The bundled monomer examples -- the sequences that can fold without an MSA search.
//!
//! `<OPENFOLD_HOME>/examples/monomer` holds `fasta_dir_<stem>/<stem>.fasta` beside
//! `alignments/<id>`. An example is offerable only when both exist: without the alignment
//! directory the fold falls back to the full MSA pipeline, which needs the complete databases and
//! runs for hours.

use std::path::{Path, PathBuf};

use crate::core::config;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Example {
    pub id: String,
    pub residues: usize,
    /// The FASTA header's molecule-name field, empty when the header carries only an id.
    pub description: String,
    pub sequence: String,
}

pub fn monomer_dir() -> PathBuf {
    config::openfold_home().join("examples/monomer")
}

pub fn scan_default() -> Vec<Example> {
    scan(&monomer_dir())
}

pub fn find(id: &str) -> Option<Example> {
    scan_default().into_iter().find(|example| example.id == id)
}

/// Every example in `dir` that has both a FASTA and a matching alignment directory, cheapest
/// first -- residue count is what someone picking an example is actually choosing on.
pub fn scan(dir: &Path) -> Vec<Example> {
    let alignments = dir.join("alignments");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<Example> = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("fasta_dir_"))
        .filter_map(|entry| first_fasta(&entry.path()))
        .filter_map(|fasta| parse(&std::fs::read_to_string(fasta).ok()?))
        .filter(|example| alignments.join(&example.id).is_dir())
        .collect();
    found.sort_by(|a, b| a.residues.cmp(&b.residues).then_with(|| a.id.cmp(&b.id)));
    found
}

fn first_fasta(dir: &Path) -> Option<PathBuf> {
    let mut fastas: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext == "fasta" || ext == "fa")
        })
        .collect();
    fastas.sort();
    fastas.into_iter().next()
}

/// Parse the first record of a FASTA. `>1UBQ_1|Chain A|UBIQUITIN|Homo sapiens (9606)` yields id
/// `1UBQ_1` and description `UBIQUITIN`; a bare `>6KWC_1` yields an empty description.
/// ponytail: first record only -- these directories hold one monomer each. Loop the records if a
/// multi-chain example ever ships.
fn parse(text: &str) -> Option<Example> {
    let mut lines = text.lines().skip_while(|line| !line.starts_with('>'));
    let header = lines.next()?.trim_start_matches('>');
    let mut fields = header.split('|');
    let id = fields.next()?.split_whitespace().next()?.to_owned();
    let description = fields.nth(1).unwrap_or("").trim().to_owned();
    let sequence: String = lines
        .take_while(|line| !line.starts_with('>'))
        .flat_map(str::chars)
        .filter(|c| !c.is_whitespace())
        .collect();
    (!id.is_empty() && !sequence.is_empty()).then_some(Example {
        residues: sequence.len(),
        id,
        description,
        sequence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lay out a monomer tree: (fasta stem, contents, whether `alignments/<id>` exists).
    fn tree(name: &str, entries: &[(&str, &str, bool)]) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "vizfold-examples-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        for (stem, contents, aligned) in entries {
            let fasta_dir = base.join(format!("fasta_dir_{stem}"));
            std::fs::create_dir_all(&fasta_dir).unwrap();
            std::fs::write(fasta_dir.join(format!("{stem}.fasta")), contents).unwrap();
            if *aligned {
                let id = contents
                    .trim_start_matches('>')
                    .split(['|', '\n'])
                    .next()
                    .unwrap();
                std::fs::create_dir_all(base.join("alignments").join(id)).unwrap();
            }
        }
        base
    }

    #[test]
    fn reads_id_description_and_sequence_from_a_piped_header() {
        let dir = tree(
            "piped",
            &[(
                "1UBQ",
                ">1UBQ_1|Chain A|UBIQUITIN|Homo sapiens (9606)\nMQIFVKTL\nTGKTITL\n",
                true,
            )],
        );
        assert_eq!(
            scan(&dir),
            vec![Example {
                id: "1UBQ_1".into(),
                residues: 15,
                description: "UBIQUITIN".into(),
                sequence: "MQIFVKTLTGKTITL".into(),
            }]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_bare_header_has_no_description() {
        let dir = tree("bare", &[("6KWC", ">6KWC_1\nGSTIQPGTGY\n", true)]);
        let found = scan(&dir);
        assert_eq!(found[0].id, "6KWC_1");
        assert_eq!(found[0].description, "");
        assert_eq!(found[0].residues, 10);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Without `alignments/<id>` the fold falls back to the full MSA pipeline, so it is not offerable.
    #[test]
    fn excludes_an_example_with_no_alignment_directory() {
        let dir = tree(
            "unaligned",
            &[
                ("2OMF", ">2OMF_1|Chain A|PORIN\nAEIYNKDG\n", false),
                ("1UBQ", ">1UBQ_1|Chain A|UBIQUITIN\nMQIFVKTL\n", true),
            ],
        );
        let ids: Vec<String> = scan(&dir).into_iter().map(|e| e.id).collect();
        assert_eq!(ids, vec!["1UBQ_1"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sorts_cheapest_first() {
        let dir = tree(
            "order",
            &[
                ("2OMF", ">2OMF_1|Chain A|PORIN\nAEIYNKDGNK\n", true),
                ("1G1J", ">1G1J_1|Chains A, B|NSP4\nIEKQ\n", true),
            ],
        );
        let ids: Vec<String> = scan(&dir).into_iter().map(|e| e.id).collect();
        assert_eq!(ids, vec!["1G1J_1", "2OMF_1"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_directory_yields_nothing() {
        assert!(scan(Path::new("/nonexistent/examples/monomer")).is_empty());
    }
}
